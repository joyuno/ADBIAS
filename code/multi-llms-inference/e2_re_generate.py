#!/usr/bin/env python3
# ================================================================
#  0. 원본 import / 설정 / 함수 (기존 코드 그대로)
# ================================================================
import pandas as pd
import os, time, json, requests, argparse, re

# ────────────────────────────────────────────────
# 데이터 경로
# ────────────────────────────────────────────────
ESSAY_FILE_PATH   = './data/asap_data/asap_2.csv'
RUBRICS_PATH      = './data/rubrics/2_rubrics.txt'
PROMPT_PATH       = './data/rubrics/2_prompt.txt'
# OUTPUT_JSON_PATH  = './results/results2/output_e2_results_claude.json'
# OUTPUT_JSON_PATH  = './results/results2/output_e2_results_gemini_merged_fixed_full.json'
# OUTPUT_JSON_PATH  = './results/results2/output_e2_results_gpt.json'
OUTPUT_JSON_PATH  = './results/results2/output_e2_results_llama.json'

# ────────────────────────────────────────────────
# OpenRouter 환경설정
# ────────────────────────────────────────────────
API_CALL_DELAY    = 1
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = '당신의 키를 넣으시오!!'
# LLM_MODEL = "anthropic/claude-3.5-sonnet" 
# LLM_MODEL = "google/gemini-2.5-flash-preview" 
# LLM_MODEL = "openai/gpt-4o" 
LLM_MODEL = "meta-llama/llama-4-maverick" 

# ────────────────────────────────────────────────
# (루브릭·프롬프트 로딩 / build_system_prompt / build_user_prompt
#  evaluate_essay_openrouter 함수는 그대로, 생략 없이 유지)
# ────────────────────────────────────────────────
def load_rubric():
    with open(RUBRICS_PATH, 'r', encoding='utf-8') as f:
        rubric_content = f.read().strip()
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    return rubric_content, prompt_text

def build_system_prompt(rubric_content):
    return f"""
You are a rater for essays written by students from grades 10.

Use the following scoring criteria when evaluating the essays:
{rubric_content}
""".strip()

def build_user_prompt(essay_id, essay_set, prompt_text, essay_text):
    sample_format = f'''
{{
  "essay_id": "{essay_id}",
  "essay_set": "{essay_set}",
  "scores": {{
    "content": {{"rationale": "Your rationale here.", "score": 0}},
    "organization": {{"rationale": "Your rationale here.", "score": 0}},
    "word choice": {{"rationale": "Your rationale here.", "score": 0}},
    "sentence fluency": {{"rationale": "Your rationale here.", "score": 0}},
    "conventions": {{"rationale": "Your rationale here.", "score": 0}}
  }}
}}
'''
    return f"""
You will be sent with an essay prompt and a student's essay response.

First, write a justification for each trait: Content, Organization, Word choice, Sentence fluency, and Conventions.
Then, assign a score from 1 to 6 for each trait.

Ratings are based on the rubric guidelines provided in the system message.

Notes: Remember that these essays are written by grade 10 students.
Avoid interpreting the rubric too strictly; consider what is developmentally appropriate for grade 10 students.

You should respond with your justification and give your score in the subsequent Python dictionary-like structure:
- No triple backticks (```)
- No "json" label
- No markdown formatting

Follow exactly this structure:
{sample_format}

Essay Prompt:
{prompt_text}

Student Essay:
{essay_text}
""".strip()

def evaluate_essay_openrouter(essay_id, essay_set, essay_text,
                              prompt_text, rubric_content, failed_set=None):
    system_prompt = build_system_prompt(rubric_content)
    user_prompt   = build_user_prompt(essay_id, essay_set, prompt_text, essay_text)

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
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result_text = resp.json()['choices'][0]['message']['content'].strip()
        parsed_result = json.loads(result_text)
        time.sleep(API_CALL_DELAY)
        return parsed_result
    except Exception as e:
        print(f"⚠️ 오류(ID {essay_id}): {e}")
        if failed_set is not None:
            failed_set.add(str(essay_id))
        return {
            "essay_id": essay_id,
            "essay_set": essay_set,
            "scores": {
                trait: {"score": 0, "rationale": f"Error: {e}"}
                for trait in ['content', 'organization', 'word choice',
                              'sentence fluency', 'conventions']
            }
        }

# ================================================================
# 1. 실패 ID 추출 (score==0 & 'Error:' 포함 기준)
# ================================================================
def extract_failed_ids(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    fail_ids = {
        str(item["essay_id"])
        for item in data
        for v in item["scores"].values()
        if v["score"] == 0 and "Error" in v["rationale"]
    }
    print(f"[+] 기존 결과에서 실패 ID {len(fail_ids)}개 감지")
    return fail_ids

# ================================================================
# 2. 병합 함수 (정렬 키 안전화)  ← 패치
# ================================================================
def merge_results(orig_json_path, retry_json_path, merged_path):
    with open(orig_json_path, encoding='utf-8') as f:
        orig  = {str(o["essay_id"]): o for o in json.load(f)}
    with open(retry_json_path, encoding='utf-8') as f:
        retry = json.load(f)
    for item in retry:
        orig[str(item["essay_id"])] = item

    def sort_key(x):
        eid = str(x["essay_id"])
        return (0, int(eid)) if eid.isdigit() else (1, eid.lower())

    merged = sorted(orig.values(), key=sort_key)
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"✅ 병합 완료 → {merged_path}")

# ================================================================
# 3. 재시도 모드
# ================================================================
def retry_mode():
    RETRY_JSON  = OUTPUT_JSON_PATH.replace(".json", "_retry.json")
    MERGED_JSON = OUTPUT_JSON_PATH.replace(".json", "_merged.json")

    fail_ids = extract_failed_ids(OUTPUT_JSON_PATH)
    if not fail_ids:
        print("🎉 실패 ID가 없습니다. 종료.")
        return

    df = pd.read_csv(ESSAY_FILE_PATH, usecols=['essay_id', 'essay_set', 'essay'])
    retry_df = df[df["essay_id"].astype(str).isin(fail_ids)]
    print(f"[+] 재평가 대상 {len(retry_df)}개")

    rubric_content, prompt_text = load_rubric()
    retry_results, still_fail = [], set()

    for _, row in retry_df.iterrows():
        res = evaluate_essay_openrouter(
            row['essay_id'], row['essay_set'], row['essay'],
            prompt_text, rubric_content, failed_set=still_fail
        )
        retry_results.append(res)

    os.makedirs(os.path.dirname(RETRY_JSON), exist_ok=True)
    with open(RETRY_JSON, 'w', encoding='utf-8') as f:
        json.dump(retry_results, f, indent=2, ensure_ascii=False)
    print(f"[√] 재평가 결과 저장 → {RETRY_JSON}")

    merge_results(OUTPUT_JSON_PATH, RETRY_JSON, MERGED_JSON)

    with open("failed_ids_claude.json", 'w') as f:
        json.dump(sorted(still_fail), f, indent=2)
    print(f"[+] 여전히 실패 {len(still_fail)}개 기록 완료")

# ================================================================
# 4. 메인 실행부
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", action="store_true",
                        help="기존 JSON의 오류 레코드만 재평가")
    args = parser.parse_args()

    if args.retry:
        retry_mode()
        exit()

    # ─── 1차 전체 평가 (기존 코드) ───
    print(f"에세이 파일 로딩 중: {ESSAY_FILE_PATH}")
    try:
        df = pd.read_csv(ESSAY_FILE_PATH, usecols=['essay_id', 'essay_set', 'essay'])
        print(f"총 {len(df)}개의 에세이 로드 완료.")
    except Exception as e:
        print(f"❌ 에세이 파일 로딩 실패: {e}")
        exit()

    rubric_content, prompt_text = load_rubric()
    all_results, processed_count = [], 0

    for _, row in df.iterrows():
        processed_count += 1
        print(f"\n--- 에세이 {processed_count}/{len(df)} 평가 시작 "
              f"(ID: {row['essay_id']}, Set: {row['essay_set']}) ---")
        res = evaluate_essay_openrouter(
            row['essay_id'], row['essay_set'], row['essay'],
            prompt_text, rubric_content
        )
        all_results.append(res)

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 평가 결과가 '{OUTPUT_JSON_PATH}' 파일로 저장되었습니다!")
