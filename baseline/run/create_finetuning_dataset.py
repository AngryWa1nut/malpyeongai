import json
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retriever import Retriever

# 설정
DOCX_PATH = "C:/workspace/malpyeong/data/국어 지식 기반 생성(RAG) 참조 문서.docx"
INPUT_PATH = "C:/workspace/malpyeong/data/korean_language_rag_V1.0_train.json"
OUTPUT_PATH = "C:/workspace/malpyeong/data/finetuning_dataset_with_context.json"


# 템플릿 함수
def create_prompt(question, context, answer):
    return f"""### 지시:
당신은 주어진 '문서'를 기반으로 '질문'에 올바르게 답하는 한국어 어문 규범 전문가입니다.
반드시 문서 내용을 충실하게 반영해, 아래 형식으로 답변하세요:

"○○○○이/가 옳다. 이유: …"

### 문서:
{context}

### 질문:
{question}

### 답변:
{answer}"""


def main():
    retriever = Retriever(DOCX_PATH)

    with open(INPUT_PATH, encoding="utf-8") as f:
        items = json.load(f)

    results = []

    for item in tqdm(items, desc="context 검색 및 프롬프트 생성"):
        question = item["input"]["question"]
        answer = item["output"]["answer"]

        try:
            context = retriever.search(question, top_n=1)[0]
        except Exception as e:
            print(f"[오류: {item['id']}] 검색 실패 - context 없음")
            context = "검색된 문서가 없습니다."

        prompt = create_prompt(question, context, answer)
        results.append({ "text": prompt })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"데이터셋 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
