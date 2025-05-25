import pandas as pd
import os
import time
import json
import requests

# ────────────────────────────────────────────────
# 데이터 경로
# ────────────────────────────────────────────────
ESSAY_FILE_PATH = './data/asap_data/asap_4.csv'
RUBRICS_PATH = './data/rubrics/4_rubrics.txt'
PROMPT_PATH = './data/rubrics/4_prompt.txt'
OUTPUT_JSON_PATH = './results/output_e4_results_claude.json'

# ────────────────────────────────────────────────
# OpenRouter 환경설정
# ────────────────────────────────────────────────
API_CALL_DELAY = 1  
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions" 
OPENROUTER_API_KEY = '당신의 키를 넣으시오!!'

LLM_MODEL = "anthropic/claude-3.5-sonnet"  ####### 여기에 OpenRouter 모델 Path 

# ────────────────────────────────────────────────
# 루브릭과 프롬프트 로딩
# ────────────────────────────────────────────────
def load_rubric():
    with open(RUBRICS_PATH, 'r', encoding='utf-8') as f:
        rubric_content = f.read().strip()
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    return rubric_content, prompt_text

# ────────────────────────────────────────────────
# 시스템 프롬프트 빌드
# ────────────────────────────────────────────────
def build_system_prompt(rubric_content):
    return f"""
You are a rater for essays written by students from grades 10.

Use the following scoring criteria when evaluating the essays:
{rubric_content}
""".strip()

# ────────────────────────────────────────────────
# 유저 프롬프트 빌드
# ────────────────────────────────────────────────
def build_user_prompt(essay_id, essay_set, prompt_text, essay_text):
    sample_format = f'''
{{
  "essay_id": "{essay_id}",
  "essay_set": "{essay_set}",
  "scores": {{
    "content": {{"rationale": "Your rationale here.", "score": 0}},
    "prompt adherence": {{"rationale": "Your rationale here.", "score": 0}},
    "language": {{"rationale": "Your rationale here.", "score": 0}},
    "narrativity": {{"rationale": "Your rationale here.", "score": 0}}
  }}
}}
'''
    return f"""
You will be sent with an essay prompt and a student's essay response.

First, write a justification for each trait: Content, Prompt adherence, Language, and Narrativity.
Then, assign a score from 0 to 3 for each trait.

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


# ────────────────────────────────────────────────
# OpenRouter로 평가 요청
# ────────────────────────────────────────────────
def evaluate_essay_openrouter(essay_id, essay_set, essay_text, prompt_text, rubric_content):
    system_prompt = build_system_prompt(rubric_content)
    user_prompt = build_user_prompt(essay_id, essay_set, prompt_text, essay_text)

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
            "scores": {trait: {"score": 0, "rationale": f"Error: {e}"} for trait in ['content', 'prompt adherence', 'language', 'narrativity']}
        }

# ────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────
print(f"에세이 파일 로딩 중: {ESSAY_FILE_PATH}")
try:
    df = pd.read_csv(ESSAY_FILE_PATH, usecols=['essay_id', 'essay_set', 'essay'])
    # df = df[100:] 
    print(f"총 {len(df)}개의 에세이 로드 완료.")
except Exception as e:
    print(f"❌ 에세이 파일 로딩 실패: {e}")
    exit()

rubric_content, prompt_text = load_rubric()

all_results = []
processed_count = 0

for index, row in df.iterrows():
    essay_id = row['essay_id']
    essay_set = row['essay_set']
    essay_text = row['essay']
    processed_count += 1

    print(f"\n--- 에세이 {processed_count}/{len(df)} 평가 시작 (ID: {essay_id}, Set: {essay_set}) ---")
    evaluation_result = evaluate_essay_openrouter(
        essay_id, essay_set, essay_text, prompt_text, rubric_content
    )
    all_results.append(evaluation_result)

# 결과 저장
print(f"\n--- 모든 에세이 평가 완료. 총 {len(all_results)}개 결과 ---")
try:
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 평가 결과가 '{OUTPUT_JSON_PATH}' 파일로 저장되었습니다!")
except Exception as e:
    print(f"❌ 결과 저장 실패: {e}")

print("스크립트 실행 완료.")
