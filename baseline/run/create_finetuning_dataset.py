# run/create_agentic_finetuning_dataset.py

import argparse
import json
from tqdm import tqdm
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 모듈 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.retriever import Retriever

# --- 프롬프트 템플릿 정의 ---

def create_keyword_extraction_prompt(query):
    """
    LLM이 질문의 핵심 문법 개념을 추출하도록 지시하는 프롬프트.
    """
    prompt = f"""다음 '질문'에서 다루는 핵심적인 한국어 어문 규범 개념을 1~3개의 키워드로 추출해줘. 
예를 들어, '띄어쓰기', '맞춤법', '표준어 규정', 'ㄷ 불규칙 활용', '사이시옷' 과 같이 간결하게 핵심만 요약해야 해.

[질문]:
{query}

[핵심 개념]:
"""
    return prompt

def create_training_text(question, context, answer):
    """
    파인튜닝 학습을 위한 단일 텍스트 데이터를 생성합니다.
    이 프롬프트 형식은 실제 추론(agentic_test.py)에서 사용하는 형식과 반드시 동일해야 합니다.
    """
    instruction_prompt = f"""### 지시:
당신은 주어진 '문서' 내용을 바탕으로 '질문'에 대해 답변하는 한국어 어문 규범 전문가입니다.
답변은 반드시 "{{선택/교정 문장}}이/가 옳다. {{이유}}" 형식이어야 합니다.
'문서' 내용을 근거로 하여 {{이유}} 부분을 상세하고 명확하게 설명하세요.
문서에서 답을 찾을 수 없는 경우, "문서에 관련 내용이 없습니다."라고 답변하세요.

### 문서:
{context}

### 질문:
{question}

### 답변:
"""
    # 모델은 instruction_prompt를 보고 answer 부분을 생성하도록 학습하게 됩니다.
    return instruction_prompt + answer


def main(args):
    # --- 1. Retriever 및 LLM 초기화 ---
    print("="*50)
    print("단계 1: Retriever 및 LLM 초기화")
    retriever = Retriever(args.knowledge_base)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_kwargs = {"device_map": args.device, "quantization_config": bnb_config}
    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    tokenizer_path = args.tokenizer if args.tokenizer else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    terminators = [t for t in terminators if t is not None]
    print("초기화 완료")
    print("="*50)

    # --- 2. 원본 학습 데이터 로딩 ---
    with open(args.input, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # --- 3. Agentic 데이터셋 생성 ---
    finetuning_dataset = []
    for item in tqdm(original_data, desc="Agentic 데이터셋 생성 중"):
        original_question = item['input']['question']
        ground_truth_answer = item['output']['answer']

        # 3-1. LLM으로 핵심 키워드 추출
        keyword_prompt = create_keyword_extraction_prompt(original_question)
        model_inputs = tokenizer(keyword_prompt, return_tensors="pt").to(args.device)
        
        keyword_outputs = model.generate(
            **model_inputs, max_new_tokens=50, eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )
        input_length = model_inputs.input_ids.shape[1]
        extracted_keywords = tokenizer.decode(keyword_outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # 3-2. 추출된 키워드로 문서 검색
        retrieved_docs = retriever.search(extracted_keywords, top_n=args.top_n)
        context = "\n\n".join(retrieved_docs)

        # 3-3. 최종 학습용 텍스트 생성
        training_text = create_training_text(original_question, context, ground_truth_answer)
        finetuning_dataset.append({ "text": training_text })

    # --- 4. 결과 저장 ---
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(finetuning_dataset, f, ensure_ascii=False, indent=2)
    
    print("="*50)
    print(f"Agentic 파인튜닝 데이터셋 생성 완료! 결과가 '{args.output}' 파일에 저장되었습니다.")
    print(f"총 {len(finetuning_dataset)}개의 학습 데이터가 생성되었습니다.")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 추론 기반의 Agentic RAG 파인튜닝 데이터셋을 생성합니다.")

    g = parser.add_argument_group("파일 경로")
    g.add_argument("--input", type=str, default="../data/korean_language_rag_V1.0_train.json", help="입력 원본 학습 파일 (JSON)")
    g.add_argument("--output", type=str, default="../data/finetuning_dataset_agentic.json", help="출력 파인튜닝 데이터셋 파일 (JSON)")
    g.add_argument("--knowledge_base", type=str, default="../data/국어 지식 기반 생성(RAG) 참조 문서.docx", help="참조할 지식 문서 (.docx)")

    g = parser.add_argument_group("모델 및 실행 환경")
    g.add_argument("--model_id", type=str, default="Qwen/Qwen1.5-7B-Chat", help="키워드 추출에 사용할 Hugging Face 모델 ID")
    g.add_argument("--tokenizer", type=str, help="Hugging Face 토크나이저 경로")
    g.add_argument("--device", type=str, default="cuda", help="모델을 로드할 장치")
    g.add_argument("--use_auth_token", type=str, help="Gated 모델 접근을 위한 Hugging Face 토큰")

    g = parser.add_argument_group("RAG 파라미터")
    g.add_argument("--top_n", type=int, default=1, help="검색기에서 가져올 관련 문서의 수")
    
    args = parser.parse_args()
    main(args)
