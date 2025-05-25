import pandas as pd
import os
import time
import json
import requests

ESSAY_FILE_PATH = './data/asap_data/asap_2.csv'
RUBRICS_PATH = './data/rubrics/2_rubrics.txt'
PROMPT_PATH = './data/rubrics/2_prompt.txt'
PEER_REVIEWS_PATH = './data/peer_data/csv_data/essay_scores_set_2.csv'
SEVERITY_PATH = './data/ranking/ranking/set_2_ranking.txt'
OUTPUT_JSON_PATH = './results/peer_results/output_e2_peer_result.json'

# TARGET_IDS = [1,2,3,4,5]

API_CALL_DELAY = 1
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = '당신의 키를 넣으시오!!'
LLM_MODEL = "meta-llama/llama-4-maverick"

def load_prompt_components():
    with open(RUBRICS_PATH, 'r', encoding='utf-8') as f:
        rubric_content = f.read().strip()
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    with open(SEVERITY_PATH, 'r', encoding='utf-8') as f:
        severity = f.read().strip()
    peer_df = pd.read_csv(PEER_REVIEWS_PATH)
    peer_df = peer_df[peer_df['rater'] != 'llama']
    
    return rubric_content, prompt_text, severity, peer_df

def format_peer_reviews_text(peer_df, essay_id):
    df = peer_df[peer_df['essay_id'] == essay_id]
    traits = df['trait'].unique()
    output_lines = []
    for trait in traits:
        output_lines.append(f"Trait: {trait}")
        sub_df = df[df['trait'] == trait]
        for _, row in sub_df.iterrows():
            line = f"- {row['rater']} | score: {row['score']} | rationale: {row['rationale']}"
            output_lines.append(line)
    return "\n".join(output_lines)

def build_system_prompt(rubric_content):
    return f"""
You are a rater for essays written by students from grades 10.

Use the following scoring criteria when evaluating the essays:
{rubric_content}
""".strip()

def build_user_prompt(essay_id, essay_set, prompt_text, essay_text, formatted_peer_reviews, severity):
    sample_format = f'''
{{
  "essay_id": "{int(essay_id)}",
  "essay_set": "{int(essay_set)}",
  "scores": {{
    "contents": {{"rationale": "Your rationale here.", "score": 0}},
    "organization": {{"rationale": "Your rationale here.", "score": 0}},
    "word choice": {{"rationale": "Your rationale here.", "score": 0}},
    "sentence fluency": {{"rationale": "Your rationale here.", "score": 0}},
    "conventions": {{"rationale": "Your rationale here.", "score": 0}}
  }}
}}
'''
    return f"""
You will be sent with an essay prompt, student's essay response,
peer reviews for the essay, and the peers’ severity rankings.

You are provided with the following information:
- Essay Prompt
- Student's Essay 
- Rubric Guidelines: the grading criteria as defined in the system message
- Peer Reviews:
  - score: the peer model’s score for that trait
  - rationale: explanation of why the score was given
- Severity Rankings: peers’ strictness ranking when evaluating each trait (e.g. content: claude>gemini>gpt)

Evaluation Steps:
Use the severity ranking as a reference point to critically analyze each evaluator’s score and accompanying rationale.
1. Write a justification for each trait: Contents, Organization, Word choice, Sentence fluency, and Conventions.
2. Assign a score from 1 to 6 for each trait.

Notes: Remember that these essays are written by grade 10 students.
Avoid interpreting the rubric too strictly; consider what is developmentally appropriate for grade 10 students.

You should respond with your justification and give your score in the subsequent Python dictionary-like structure:
- No triple backticks (```)
- No "json" label
- No markdown formatting

Severity Rankings:
{severity}

Peer Reviews:
{formatted_peer_reviews}

Essay Prompt:
{prompt_text}

Student's Essay:
{essay_text}

Follow exactly this structure:
{sample_format}
""".strip()


def evaluate_essay_openrouter(essay_id, essay_set, essay_text, prompt_text, rubric_content, peer_df, severity):
    formatted_peer_reviews = format_peer_reviews_text(peer_df, essay_id)
    system_prompt = build_system_prompt(rubric_content)
    user_prompt = build_user_prompt(essay_id, essay_set, prompt_text, essay_text, formatted_peer_reviews, severity)
    # print(user_prompt) ## 프롬프트 확인

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 2048
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result_text = response.json()['choices'][0]['message']['content'].strip()
        print("⚡ result_text 확인:", result_text)
        parsed_result = json.loads(result_text)
        time.sleep(API_CALL_DELAY)
        return parsed_result
    
    except Exception as e:
        print(f"⚠️ 오류: OpenRouter 평가 중 문제 발생 - {e}")
        return {
            "essay_id": essay_id,
            "essay_set": essay_set,
            "scores": {trait: {"score": 0, "rationale": f"Error: {e}"} for trait in ['contents', 'organization', 'word choice', 'sentence fluency', 'conventions']}
        }

print(f"에세이 파일 로딩 중: {ESSAY_FILE_PATH}")
try:
    df = pd.read_csv(ESSAY_FILE_PATH, usecols=['essay_id', 'essay_set', 'essay'])
    # df = df[df['essay_id'].isin(TARGET_IDS)]
    print(f"총 {len(df)}개의 에세이 로드 완료.")
except Exception as e:
    print(f"❌ 에세이 파일 로딩 실패: {e}")
    exit()

rubric_content, prompt_text, severity, peer_df = load_prompt_components()

all_results = []
processed_count = 0

for _, row in df.iterrows():
    essay_id = row['essay_id']
    essay_set = row['essay_set']
    essay_text = row['essay']
    processed_count += 1

    print(f"\n--- 에세이 {processed_count}/{len(df)} 평가 시작 (ID: {essay_id}, Set: {essay_set}) ---")

    try:
        evaluation_result = evaluate_essay_openrouter(
            essay_id, essay_set, essay_text, prompt_text,
            rubric_content, peer_df, severity
        )
        all_results.append(evaluation_result)
    except Exception as e:
        print(f"⚠️ {essay_id} 처리 중 오류 발생: {e}")
        continue

print(f"\n--- 모든 에세이 평가 완료. 총 {len(all_results)}개 결과 ---")
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 평가 결과가 '{OUTPUT_JSON_PATH}' 파일로 저장되었습니다!")
except Exception as e:
    print(f"❌ 결과 저장 실패: {e}")

print("스크립트 실행 완료.")
