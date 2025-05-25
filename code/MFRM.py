import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
import glob
from collections import defaultdict


# Seed Fixing Function
def set_seed(seed=42):
    """Sets the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Get Score Boundary
def get_score_boundary(essay_set: int):
    """Returns (min_score, max_score) for each essay set
    """
    # Score range for essay sets
    known_ranges = {
        1: (1, 6),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 3),
        8: (1, 6)
    }
    
    # Return fixed range
    if essay_set in known_ranges:
        return known_ranges[essay_set]
    
    # Raise exception if essay set does not exist
    raise ValueError(f"Essay set {essay_set} does not exist")


# Dataset Definition
class MFRMDataset(Dataset):
    """Dataset for MFRM model training"""
    def __init__(self, df):
        self.p = torch.tensor(df["essay_idx"].values, dtype=torch.long)
        self.r = torch.tensor(df["rater_idx"].values, dtype=torch.long)
        self.st = torch.tensor(df["st_idx"].values, dtype=torch.long)
        self.k = torch.tensor(df["k_idx"].values, dtype=torch.long)
        self.m = torch.tensor(df["m_size"].values, dtype=torch.long)
        self.t = torch.tensor(df["trait_idx"].values, dtype=torch.long)
    
    def __len__(self):
        return len(self.k)
    
    def __getitem__(self, i):
        return self.p[i], self.r[i], self.st[i], self.k[i], self.m[i], self.t[i]


# MFRM Model
class MFRM(nn.Module):
    """Implementation of Many-Facet Rasch Model"""
    def __init__(self, N_person, N_trait, N_rater, N_strat, K_max, 
                 threshold_scale_factor=0.2, ability_scale_factor=1.0, severity_scale_factor=0.5):
        super().__init__()
        # Model parameters
        self.person_trait = nn.Embedding(N_person * N_trait, 1)  # Essay ability B_{n,t}
        
        # Rater severity
        # Original plan: rater_set_trait but implemented safely as rater_trait
        self.rater_set_trait = nn.Embedding(N_rater * N_strat, 1)  # Rater severity ρ_{j,st}
        
        # Output debugging information
        print(f"MFRM model initialization:")
        print(f"  N_person * N_trait: {N_person * N_trait}")
        print(f"  N_rater * N_strat: {N_rater * N_strat}")
        
        self.threshold_raw = nn.Embedding(N_strat, K_max)  # Threshold values (raw)
        
        # Initialization
        nn.init.normal_(self.threshold_raw.weight, mean=0.0, std=0.01)
        
        # Hyperparameters
        self.threshold_scale = threshold_scale_factor
        self.ability_scale = ability_scale_factor
        self.severity_scale = severity_scale_factor
        
        # Dimension information
        self.N_person = N_person
        self.N_trait = N_trait
        self.N_rater = N_rater
        self.N_strat = N_strat
        self.K_max = K_max
    
    def forward(self, p, r, st, k, m, t):
        """Model forward pass and loss calculation"""
        batch_size = p.size(0)
        
        # Essay ability (B)
        idx_pt = p * self.N_trait + t
        B = self.person_trait(idx_pt).squeeze(-1)
        
        # Rater severity (R)
        idx_rst = r * self.N_strat + st
        R = self.rater_set_trait(idx_rst).squeeze(-1)
        
        # Threshold calculation (T)
        T_raw = self.threshold_raw(st)  # [batch_size, K_max]
        T_cum = torch.cumsum(F.softplus(T_raw * self.threshold_scale), dim=1)  # [batch_size, K_max]
        # anchor first threshold to zero for identification
        T_cum = T_cum - T_cum[:, :1]
        
        # Partial score logits (B - R)
        logits = B - R  # [batch_size]
        
        # vectorized partial credit loss calculation
        device = logits.device
        # construct full threshold matrix with anchored zero step
        T_all = torch.cat([torch.zeros(batch_size, 1, device=device), T_cum], dim=1)  # [batch_size, K_max+1]
        # log numerator: P(X=k)
        step_k = torch.gather(T_all, 1, k.unsqueeze(1)).squeeze(1)
        log_numer = k.float() * logits - step_k
        # compute log denominator across valid categories
        G = torch.arange(self.K_max+1, device=device).float().unsqueeze(0)  # [1, K_max+1]
        logits_exp = logits.unsqueeze(1)  # [batch_size,1]
        log_terms = G * logits_exp - T_all  # [batch_size, K_max+1]
        mask = G < m.unsqueeze(1)
        log_terms = log_terms.masked_fill(~mask, float('-inf'))
        log_denom = torch.logsumexp(log_terms, dim=1)
        loss = torch.mean(-(log_numer - log_denom))
        return loss


# Checkpoint Management Class
class Trainer:
    """Model training and checkpoint management"""
    def __init__(self, model, optimizer, checkpoint_path='best_model.pt',
                max_epochs=90, patience=3, target_loss=0.5):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_loss = target_loss
        self.loss_history = []
        self.best_loss = float('inf')
        self.threshold_warmup_epochs = 5
        self.threshold_final_factor = 0.5
        
    def load_checkpoint(self):
        """Load saved checkpoint"""
        try:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_history = checkpoint.get('loss_history', [])
            epoch = checkpoint.get('epoch', -1)
            loss = checkpoint.get('loss', float('inf'))
            print(f"Loaded saved model. (Epoch {epoch+1}, loss: {loss:.4f})")
            return True
        except FileNotFoundError:
            print("No saved model found. Starting new training.")
            return False
    
    def save_checkpoint(self, epoch, loss):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_history': self.loss_history,
        }, self.checkpoint_path)
    
    def train(self, loader, skip_training=False):
        """Perform model training"""
        if skip_training:
            return
            
        no_improve = 0
        threshold_scale_backup = self.model.threshold_scale
        
        for epoch in range(self.max_epochs):
            # Warm-up threshold_scale_factor
            if epoch == self.threshold_warmup_epochs:
                self.model.threshold_scale = self.threshold_final_factor
                
            total_loss = 0.0
            for p, r, st, k, m, t in loader:
                # Forward pass
                loss = self.model(p, r, st, k, m, t)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # enforce sum-to-zero constraint for identification
                with torch.no_grad():
                    # ability parameters (person_trait)
                    W = self.model.person_trait.weight.view(self.model.N_person, self.model.N_trait)
                    W = W - W.mean(dim=0, keepdim=True)
                    self.model.person_trait.weight.copy_(W.view(-1,1))
                    # severity parameters (rater_set_trait)
                    V = self.model.rater_set_trait.weight.view(self.model.N_rater, self.model.N_strat)
                    V = V - V.mean(dim=0, keepdim=True)
                    self.model.rater_set_trait.weight.copy_(V.view(-1,1))
                
                total_loss += loss.item()
                
            # Epoch average loss
            mean_loss = total_loss / len(loader)
            self.loss_history.append(mean_loss)
            print(f"Epoch {epoch+1}: loss={mean_loss:.4f}")
            
            # Target loss reached, stop training
            if mean_loss <= self.target_loss:
                print(f"Target loss({self.target_loss}) reached. Stopping training.")
                self.save_checkpoint(epoch, mean_loss)
                break
                
            # Early stopping check
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                no_improve = 0
                self.save_checkpoint(epoch, mean_loss)
            else:
                no_improve += 1
                
            if no_improve >= self.patience:
                print(f"Early stopping: No improvement for {epoch+1} epochs")
                break
        
        # Load last checkpoint
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()


def get_available_peer_raters():
    """Get the list of available peer raters."""
    peer_raters = set()
    for set_num in range(1, 9):
        try:
            peer_path = f"data/peer_data/peer_{set_num}.csv"
            if os.path.exists(peer_path):
                df = pd.read_csv(peer_path)
                if 'rater' in df.columns:
                    peer_raters.update(df['rater'].unique())
        except Exception as e:
            print(f"Warning: peer_{set_num}.csv file load error: {e}")
    return sorted(list(peer_raters))

def select_peer_raters():
    """Interface for peer rater selection"""
    available_raters = get_available_peer_raters()
    if not available_raters:
        print("No available peer raters.")
        return []
    
    print("\nAvailable peer raters:")
    for i, rater in enumerate(available_raters, 1):
        print(f"{i}. {rater}")
    print(f"{len(available_raters) + 1}. Select all")
    
    while True:
        try:
            choice = input("\nEnter the number of the rater to select (separate multiple selections with commas, e.g., 1,3,5): ")
            if choice.strip() == str(len(available_raters) + 1):
                return available_raters
            
            selected_indices = [int(x.strip()) for x in choice.split(',')]
            if all(1 <= idx <= len(available_raters) for idx in selected_indices):
                return [available_raters[idx-1] for idx in selected_indices]
            print(f"Please enter a number between 1 and {len(available_raters)}.")
        except ValueError:
            print("Please enter the numbers in the correct format (e.g., 1,3,5)")


# Data Loading and Preprocessing
def load_data(selected_sets=None, selected_raters=None, selected_peer_raters=None, human_option='human_1'):
    """Load and preprocess essay data
    
    Args:
        selected_sets (list, optional): List of essay set numbers to analyze
        selected_raters (list, optional): List of rater types to analyze ('human', 'meta', 'peer')
        selected_peer_raters (list, optional): List of selected peer raters
        human_option (str, optional): Option for human data processing ('human_1', 'human_2', 'both')
    """
    # List of available essay sets
    available_sets = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Available essay sets: {available_sets}")
    
    # List of available rater types
    available_raters = ['human', 'meta', 'peer']
    if selected_raters is None:
        selected_raters = available_raters
    else:
        # Check if the selected rater types are valid
        if not all(r in available_raters for r in selected_raters):
            raise ValueError(f"Invalid rater type. Available types are {available_raters}.")
    
    print(f"Selected rater types: {selected_raters}")
    
    # Select essay sets to analyze
    if selected_sets is None:
        selected_sets = available_sets
    
    # Check if the selected sets are valid
    if not all(s in available_sets for s in selected_sets):
        raise ValueError(f"Invalid set number. Available sets are {available_sets}.")
    
    print(f"\nSelected essay sets: {selected_sets}")
    
    # Load data
    dfs = []
    
    for set_num in selected_sets:
        # Load human data
        if 'human' in selected_raters:
            if set_num in [7, 8]:
                if human_option in ['human_1', 'both']:
                    # Load human_1
                    human_path = f"data/human_data/human_{set_num}_1.csv"
                    if os.path.exists(human_path):
                        human_df = pd.read_csv(human_path)
                        human_df['rater'] = 'human_1' if human_option == 'both' else 'human'
                        dfs.append(human_df)
                
                if human_option in ['human_2', 'both']:
                    # Load human_2
                    human_path = f"data/human_data/human_{set_num}_2.csv"
                    if os.path.exists(human_path):
                        human_df = pd.read_csv(human_path)
                        human_df['rater'] = 'human_2' if human_option == 'both' else 'human'
                        dfs.append(human_df)
            else:
                # For sets 1-6, duplicate the data if 'both' is selected
                human_path = f"data/human_data/human_{set_num}.csv"
                if os.path.exists(human_path):
                    if human_option == 'both':
                        # Load twice with different labels
                        human_df = pd.read_csv(human_path)
                        human_df1 = human_df.copy()
                        human_df2 = human_df.copy()
                        human_df1['rater'] = 'human_1'
                        human_df2['rater'] = 'human_2'
                        dfs.extend([human_df1, human_df2])
                    else:
                        # Load once with 'human' label
                        human_df = pd.read_csv(human_path)
                        human_df['rater'] = 'human'
                        dfs.append(human_df)
        
        # Load meta data
        if 'meta' in selected_raters:
            meta_path = f"data/meta_data/meta_{set_num}.csv"
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path)
                meta_df['rater'] = 'meta'
                dfs.append(meta_df)
        
        # Load peer data
        if 'peer' in selected_raters:
            peer_path = f"data/peer_data/peer_{set_num}.csv"
            if os.path.exists(peer_path):
                peer_df = pd.read_csv(peer_path)
                if selected_peer_raters:
                    # Filter peer raters
                    peer_df = peer_df[peer_df['rater'].isin(selected_peer_raters)]
                dfs.append(peer_df)
    
    if not dfs:
        raise ValueError("No data found for the selected rater types.")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert essay ID to string
    df["essay_id"] = df["essay_id"].astype(str)
    
    # Rename columns: 'essay_set' to 'set'
    if 'essay_set' in df.columns and 'set' not in df.columns:
        df = df.rename(columns={'essay_set': 'set'})
    
    # Add score boundaries
    df["score_min"] = df["set"].apply(lambda s: get_score_boundary(int(s))[0])
    df["score_max"] = df["set"].apply(lambda s: get_score_boundary(int(s))[1])
    
    # 0-based category index and category size
    df["k_idx"] = df["score"] - df["score_min"]
    df["m_size"] = df["score_max"] - df["score_min"] + 1
    
    # Index mapping
    df["essay_idx"] = df["essay_id"].astype("category").cat.codes
    df["rater_idx"] = df["rater"].astype("category").cat.codes
    df["trait_idx"] = df["trait"].astype("category").cat.codes
    df["set_trait"] = df["set"].astype(str) + "_" + df["trait"]
    df["st_idx"] = df["set_trait"].astype("category").cat.codes
    
    # Dimension information
    N_person = df["essay_idx"].nunique()
    N_trait = df["trait_idx"].nunique()
    N_rater = df["rater_idx"].nunique()
    N_strat = df["st_idx"].nunique()
    K_max = int(df["m_size"].max() - 1)
    
    # Mapping information
    mappings = {
        'rater_map': df[["rater_idx", "rater"]].drop_duplicates().set_index("rater_idx")["rater"],
        'trait_map': df[["trait_idx", "trait"]].drop_duplicates().set_index("trait_idx")["trait"],
        'essay_meta': df[["essay_idx", "essay_id", "set"]].drop_duplicates().set_index("essay_idx"),
        'm_counts': torch.tensor(df.groupby("st_idx")["m_size"].first().values, dtype=torch.long)
    }
    
    dimensions = {
        'N_person': N_person,
        'N_trait': N_trait,
        'N_rater': N_rater,
        'N_strat': N_strat,
        'K_max': K_max
    }
    
    # Print data information
    print("\nData load information:")
    print(f"Total number of essays: {N_person}")
    print(f"Total number of traits: {N_trait}")
    print(f"Total number of raters: {N_rater}")
    print(f"Total number of set-trait combinations: {N_strat}")
    print(f"Maximum category size: {K_max + 1}")
    
    return df, mappings, dimensions


# Extract and save essay ability
def export_ability(model, df, mappings):
    """Extract and save essay ability"""
    print("\n1. ability.csv creation in progress...")
    os.makedirs("MFRM", exist_ok=True)
    N_person = model.N_person
    N_trait = model.N_trait
    
    # Extract ability
    abilities = model.person_trait.weight.detach().view(N_person, N_trait)
    abilities = abilities * model.ability_scale
    
    # Remove trait-wise mean
    trait_means = torch.mean(abilities, dim=0, keepdim=True)
    abilities = abilities - trait_means
    
    # Create data frame
    ability_rows = []
    for essay_idx in range(N_person):
        for trait_idx in range(N_trait):
            ability_rows.append({
                "essay_id": mappings['essay_meta'].loc[essay_idx, "essay_id"],
                "essay_set": mappings['essay_meta'].loc[essay_idx, "set"],
                "trait": mappings['trait_map'][trait_idx],
                "ability": abilities[essay_idx, trait_idx].item()
            })
    
    ability_df = pd.DataFrame(ability_rows)
    ability_df.to_csv(os.path.join("MFRM", "ability.csv"), index=False, encoding="utf-8", float_format="%.6f")
    print("1. ability.csv creation complete")
    
    return abilities


# Extract and save rater severity
def export_severity(model, df, mappings):
    """Extract and save rater severity"""
    print("\n2. severity.csv creation in progress...")
    os.makedirs("MFRM", exist_ok=True)
    N_rater = model.N_rater
    N_trait = model.N_trait
    N_strat = model.N_strat
    
    # Dimension debugging information output
    print(f"  N_rater: {N_rater}, N_trait: {N_trait}, N_strat: {N_strat}")
    
    # Extract severity (restore original form)
    rater_set_trait_weights = model.rater_set_trait.weight.detach().squeeze()
    
    # Calculate severity for each trait (separated by set)
    trait_set_map = {}  # trait와 set을 합쳐서 st_idx로 매핑
    for _, row in df[["trait_idx", "set", "st_idx"]].drop_duplicates().iterrows():
        trait_idx, set_num, st_idx = row["trait_idx"], row["set"], row["st_idx"]
        if trait_idx not in trait_set_map:
            trait_set_map[trait_idx] = {}
        trait_set_map[trait_idx][set_num] = st_idx
    
    # Calculate severity for each (rater, set, trait) combination
    severity_rows = []
    for j in range(N_rater):
        for t in range(N_trait):
            trait_name = mappings['trait_map'][t]
            
            # Search for set-specific data for this trait
            for set_num, st_idx in trait_set_map.get(t, {}).items():
                if st_idx >= len(rater_set_trait_weights):
                    continue
                    
                # Calculate severity value for this (rater, st_idx) combination
                idx = j * N_strat + st_idx
                if idx >= len(rater_set_trait_weights):
                    continue
                
                severity_val = rater_set_trait_weights[idx].item()
                
                # Calculate average severity for this st_idx (rater-wise average)
                rater_st_vals = []
                for r in range(N_rater):
                    r_idx = r * N_strat + st_idx
                    if r_idx < len(rater_set_trait_weights):
                        rater_st_vals.append(rater_set_trait_weights[r_idx].item())
                
                if not rater_st_vals:
                    continue
                    
                st_mean = sum(rater_st_vals) / len(rater_st_vals)
                
                # Center severity around mean and scale
                centered_severity = severity_val - st_mean
                scaled_severity = centered_severity * model.severity_scale
                
                severity_rows.append({
                    "rater": mappings['rater_map'][j],
                    "set": set_num,
                    "trait": trait_name,
                    "severity": scaled_severity
                })
    
    severity_df = pd.DataFrame(severity_rows)
    
    # ----- infit/outfit calculation logic  -----
    # Calculate predicted scores
    T_raw = model.threshold_raw.weight.detach()
    T_cum = torch.cumsum(F.softplus(T_raw * model.threshold_scale), dim=1)
    
    # Calculate predicted scores for each (essay, trait) combination
    expected_scores = {}
    
    # Prepare st_idx mapping information for essay_idx and trait_idx
    st_idx_map = df[['essay_idx', 'trait_idx', 'st_idx', 'score_min', 'score_max']].drop_duplicates()
    
    # Extract ability
    abilities = model.person_trait.weight.detach().view(model.N_person, model.N_trait)
    abilities = abilities * model.ability_scale
    
    for essay_idx in range(model.N_person):
        for trait_idx in range(model.N_trait):
            essay_id = mappings['essay_meta'].loc[essay_idx, "essay_id"]
            trait = mappings['trait_map'][trait_idx]
            
            # Find information for this essay and trait
            mask = (st_idx_map['essay_idx'] == essay_idx) & (st_idx_map['trait_idx'] == trait_idx)
            if not mask.any():
                continue
                
            info = st_idx_map[mask].iloc[0]
            st_idx = info['st_idx']
            score_min = info['score_min']
            score_max = info['score_max']
            m_i = score_max - score_min + 1
            
            # Calculate predicted scores
            theta_val = abilities[essay_idx, trait_idx]
            steps = T_cum[st_idx, :m_i-1]
            
            p_ge_list = [torch.tensor(1.0)]
            for step in steps:
                p_ge_list.append(torch.sigmoid(theta_val - step))
            p_ge_list.append(torch.tensor(0.0))
            
            p_ge = torch.stack(p_ge_list)
            p_vals = p_ge[:-1] - p_ge[1:]
            
            raw_cats = torch.arange(score_min, score_min + m_i, dtype=torch.float32)
            exp_score = torch.dot(raw_cats, p_vals).item()
            
            expected_scores[(essay_id, trait)] = exp_score
    
    # Calculate residuals and fit statistics
    fit_stats = defaultdict(lambda: defaultdict(lambda: {'residuals': [], 'variances': []}))
    
    for _, row in df.iterrows():
        essay_id = row['essay_id']
        trait = row['trait']
        rater = row['rater']
        set_num = row['set']
        observed = row['score']
        
        key = (essay_id, trait)
        if key in expected_scores:
            expected = expected_scores[key]
            residual = observed - expected
            
            # Model-based variance calculation
            score_min = row['score_min']
            score_max = row['score_max']
            m_i = score_max - score_min + 1
            st_idx = row['st_idx']
            
            theta_val = abilities[row['essay_idx'], row['trait_idx']]
            steps = T_cum[st_idx, :m_i-1]
            
            p_ge_list = [torch.tensor(1.0)]
            for step in steps:
                p_ge_list.append(torch.sigmoid(theta_val - step))
            p_ge_list.append(torch.tensor(0.0))
            
            p_ge = torch.stack(p_ge_list)
            p_vals = p_ge[:-1] - p_ge[1:]
            
            raw_cats = torch.arange(score_min, score_min + m_i, dtype=torch.float32)
            sq_diff = (raw_cats - expected).pow(2)
            model_variance = torch.sum(sq_diff * p_vals).item()
            
            std_residual = residual / torch.sqrt(torch.tensor(max(0.1, model_variance))).item()
            fit_stats[rater][(set_num, trait)]['residuals'].append(std_residual**2)
            fit_stats[rater][(set_num, trait)]['variances'].append(model_variance)
    
    # Calculate fit statistics and add to severity_df
    fit_rows = []
    for rater, set_trait_stats in fit_stats.items():
        for (set_num, trait), stats in set_trait_stats.items():
            residuals = stats['residuals']
            variances = stats['variances']
            
            if len(residuals) >= 5: 
                n_obs = len(residuals)
                
                # Outfit MSQ
                outfit_msq = sum(residuals) / n_obs
                
                # Infit MSQ
                weights = [1.0 / max(0.1, var) for var in variances]
                weighted_residuals = [r * w for r, w in zip(residuals, weights)]
                infit_msq = sum(weighted_residuals) / sum(weights)
                
                fit_rows.append({
                    "rater": rater,
                    "set": set_num,
                    "trait": trait,
                    "infit_msq": infit_msq,
                    "outfit_msq": outfit_msq,
                    "n_observation": n_obs
                })
    
    # Combine severity and fit statistics
    fit_df = pd.DataFrame(fit_rows)
    severity_df = pd.merge(severity_df, fit_df, on=['rater', 'set', 'trait'], how='left')

    # Compute standard error of severity per (set, trait) across raters
    severity_df['severity_se'] = severity_df.groupby(['set', 'trait'])['severity'] \
        .transform(lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    
    severity_df.to_csv(os.path.join("MFRM", "severity.csv"), index=False, encoding="utf-8", float_format="%.6f")
    print("2. severity.csv 생성 완료")
    
    # Statistics output
    print("\n=== 평가자 심사 정도 및 적합도 통계 ===")
    print("\n1. 특성별 전체 통계:")
    print(severity_df.groupby('trait')['severity'].agg(['mean', 'std', 'min', 'max']).round(4))
    print("\n2. 평가자별 세트-특성 상세 통계 (일부):")
    print(severity_df.head().to_string(index=False))
    
    return severity_df


# Threshold extraction and storage
def export_threshold(model, df):
    """Threshold extraction and storage"""
    print("\n3. threshold.csv 생성 중...")
    os.makedirs("MFRM", exist_ok=True)
    N_strat = model.N_strat
    
    # Extract threshold
    T_raw = model.threshold_raw.weight.detach()
    T_cum = torch.cumsum(F.softplus(T_raw * model.threshold_scale), dim=1)
    
    # Create dataframe
    threshold_rows = []
    for st_idx in range(N_strat):
        set_trait = df[df["st_idx"] == st_idx]["set_trait"].iloc[0]
        steps = T_cum[st_idx]
        for step, val in enumerate(steps, start=1):
            threshold_rows.append({
                "set_trait": set_trait,
                "step": step,
                "threshold": val.item()
            })
    
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(os.path.join("MFRM", "threshold.csv"), index=False, encoding="utf-8", float_format="%.6f")
    print("3. threshold.csv 생성 완료")
    
    return T_cum


# Training loss history storage
def export_loss_history(loss_history):
    """Training loss history storage"""
    print("\n4. training_loss.csv 생성 중...")
    os.makedirs("MFRM", exist_ok=True)
    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(loss_history)+1)),
        "loss": loss_history
    })
    
    loss_df.to_csv(os.path.join("MFRM", "training_loss.csv"), index=False, encoding="utf-8")
    print("4. training_loss.csv 생성 완료")
    
    return loss_df


# Main Function
def main():
    # Seed setting
    set_seed(2025)
    
    # User input - multiple selection possible
    print("\nSelect rater types to analyze (multiple selection possible, separated by commas):")
    print("1. Human")
    print("2. Meta")
    print("3. Peer")
    print("4. 모두")
    
    while True:
        try:
            choice = input("\n선택 (예: 1,3 또는 4): ")
            if choice.strip() == "4":
                selected_raters = ['human', 'meta', 'peer']
                break
            
            selected_indices = [int(x.strip()) for x in choice.split(',')]
            if all(1 <= idx <= 3 for idx in selected_indices):
                selected_raters = []
                if 1 in selected_indices:
                    selected_raters.append('human')
                if 2 in selected_indices:
                    selected_raters.append('meta')
                if 3 in selected_indices:
                    selected_raters.append('peer')
                break
            print("1부터 3 사이의 번호를 입력해주세요.")
        except ValueError:
            print("올바른 형식으로 입력해주세요 (예: 1,3 또는 4)")
    
    print(f"\n선택된 평가자 유형: {selected_raters}")
    
    # Human option selection
    human_option = 'human_1'  # default
    if 'human' in selected_raters:
        print("\nHuman 데이터 처리 옵션을 선택하세요:")
        print("1. Human_1만 사용 (Human_1을 Human으로 처리)")
        print("2. Human_2만 사용 (Human_2를 Human으로 처리)")
        print("3. 둘 다 사용 (Human_1과 Human_2를 별도로 처리)")
        
        while True:
            try:
                choice = input("\n선택 (1-3): ").strip()
                if choice == '1':
                    human_option = 'human_1'
                    break
                elif choice == '2':
                    human_option = 'human_2'
                    break
                elif choice == '3':
                    human_option = 'both'
                    break
                print("1, 2, 또는 3을 입력해주세요.")
            except Exception as e:
                print(f"오류: {e}")
    
    # Peer rater selection
    selected_peer_raters = None
    if 'peer' in selected_raters:
        selected_peer_raters = select_peer_raters()
        if selected_peer_raters:
            print(f"\n선택된 peer rater: {selected_peer_raters}")
    
    # Load and preprocess data
    selected_sets = [1, 2, 3, 4, 5, 6, 7, 8]
    df, mappings, dimensions = load_data(selected_sets, selected_raters, selected_peer_raters, human_option)
    
    # Set and dataloader configuration
    ds = MFRMDataset(df)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    
    # Debugging information output
    print(f"\nDimension information:")
    for key, value in dimensions.items():
        print(f"  {key}: {value}")
    
    # Model initialization
    model = MFRM(
        dimensions['N_person'], 
        dimensions['N_trait'], 
        dimensions['N_rater'], 
        dimensions['N_strat'], 
        dimensions['K_max']
    )
    
    # Optimizer configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    # Trainer initialization and training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_epochs=50,
        patience=3,
        target_loss=0.5
    )
    
    # Checkpoint loading attempt
    skip_training = trainer.load_checkpoint()
    
    # Training execution
    trainer.train(loader, skip_training)
    
    # Model update, use trainer.model
    model = trainer.model
    
    # Result extraction and storage
    print("\n=== Result file generation ===")
    
    # Ability extraction and storage
    abilities = export_ability(model, df, mappings)
    
    # Severity extraction and storage - dimension information confirmation
    severity_df = export_severity(model, df, mappings)
    
    # Threshold extraction and storage
    thresholds = export_threshold(model, df)
    
    # Training loss history storage
    loss_df = export_loss_history(trainer.loss_history)
    
    print("\nAll result files generated successfully!")


if __name__ == '__main__':
    main()
