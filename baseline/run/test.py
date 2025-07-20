import argparse
import json
from tqdm import tqdm
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.retriever import Retriever  # src/retriever.py 에서 Retriever 클래스를 가져옵니다.


def create_prompt(query, context):
    prompt_template = f"""### 역할:
당신은 주어진 '참고 문서'를 기반으로 '질문'에 답변하는 한국어 어문 규범 전문가입니다.
다음 지시사항을 반드시 따르세요:
1.  **참고 문서에 근거하여 답변하세요.**
2.  **답변은 정확히 다음 형식으로 작성하세요:**
    "{{올바른 문장}}이/가 옳다. {{이유}}"
3.  **이유는 참고 문서 내용을 바탕으로 상세하고 명확하게 설명하세요.**
4.  참고 문서에서 답을 찾을 수 없으면 "문서에 관련 내용이 없습니다."라고만 답변하세요.

### 참고 문서:
{context}

### 질문:
{query}

### 답변:
"""
    return prompt_template

def main(args):

    retriever = Retriever(args.knowledge_base)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_kwargs = {
        "device_map": args.device,
        "quantization_config": bnb_config,
    }
    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    tokenizer_path = args.tokenizer if args.tokenizer else args.model_id
    tokenizer_kwargs = {}
    if args.use_auth_token:
        tokenizer_kwargs["use_auth_token"] = args.use_auth_token
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    terminators = [token_id for token_id in terminators if token_id is not None]

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)
        input_data = input_data[:5]

    for item in tqdm(input_data, desc="RAG 처리 중"):
        query = item['input']['question']

        # 3-1. Retrieve: 질문과 관련된 지식 검색
        retrieved_docs = retriever.search(query, top_n=args.top_n)
        context = "\n\n".join(retrieved_docs)

        # 3-2. Augment: 프롬프트 생성
        prompt = create_prompt(query, context)
        model_inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        # 3-3. Generate: LLM으로 답변 생성
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            temperature=0.7,
            top_p=0.8,
            do_sample=True
        )

        input_length = model_inputs.input_ids.shape[1]
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        item["output"] = {"answer": output_text}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)
    print(f"RAG 파이프라인 실행 완료! 결과가 '{args.output}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_rag", description="RAG 파이프라인을 실행합니다.")

    g = parser.add_argument_group("파일 경로")
    g.add_argument("--input", type=str, default="../data/korean_language_rag_V1.0_dev.json", help="입력 질문 파일 (JSON)")
    g.add_argument("--output", type=str, default="../result_rag_output.json", help="출력 답변 파일 (JSON)")
    g.add_argument("--knowledge_base", type=str, default="../data/국어 지식 기반 생성(RAG) 참조 문서.docx", help="참조할 지식 문서 (.docx)")

    g = parser.add_argument_group("모델 및 실행 환경")
    g.add_argument("--model_id", type=str, default=1, help="Hugging Face 모델 ID")
    g.add_argument("--tokenizer", type=str, help="Hugging Face 토크나이저 경로 (지정하지 않으면 model_id와 동일)")
    g.add_argument("--device", type=str, default="cuda", help="모델을 로드할 장치 (예: cuda, cpu)")
    g.add_argument("--use_auth_token", type=str, help="Gated 모델 접근을 위한 Hugging Face 토큰")

    g = parser.add_argument_group("RAG 파라미터")
    g.add_argument("--top_n", type=int, default=1, help="검색기에서 가져올 관련 문서의 수")
    
    args = parser.parse_args()
    main(args)