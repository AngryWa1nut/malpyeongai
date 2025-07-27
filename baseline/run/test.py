import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re  # 정규표현식 라이브러리

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
project_path = '/content/drive/MyDrive/malpyeong/baseline'
if project_path not in sys.path:
    sys.path.append(project_path)
from src.retriever import Retriever

# 수정: 모델이 간결한 답변을 생성하도록 프롬프트를 명확하고 강력하게 수정
def create_prompt(query: str, context: str) -> str:
    prompt_template = f"""### 역할:
당신은 주어진 '참고 문서'를 기반으로 '질문'에 답변하는 한국어 어문 규범 전문가입니다.

# 지시사항
1. '참고 문서'와 당신의 지식을 종합하여 질문에 대한 답을 찾으세요.
2. 아래 '좋은 답변의 예시'와 같이, 답변은 반드시 "{{올바른 문장}}이/가 옳다. {{이유}}" 형식으로 작성하세요.
3. **이유는 핵심적인 근거만 간결하게 1~2 문장으로 요약하여 설명해야 합니다.**
4. 불필요한 부연 설명이나 배경 지식을 나열하지 마세요.

# 좋은 답변의 예시
질문: "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\\n\\"오늘은 퍼즐 마추기를 해 볼 거예요.\\""
답변: "\"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."

---

### 참고 문서:
{context}

### 질문:
{query}

### 답변:
"""
    return prompt_template

# 수정: 안정성을 높인 최종 후처리 함수
def postprocess_answer(text: str) -> str:
    """
    후처리 기능의 영향을 확인하기 위한 최소 기능 버전입니다.
    """
    # 1. 불필요한 마커가 문장 시작에 오는 경우를 대비해 먼저 제거
    stop_phrases = ["###", "[지침]", "[상황", "참고 문서:", "질문:"]
    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase, 1)[0]
            
    return text.strip()


def load_dependencies(args: argparse.Namespace) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print(f"{args.precision}비트 정밀도로 모델과 토크나이저를 로드하는 중...")
    model_kwargs = {"device_map": args.device}
    if args.token:
        model_kwargs["token"] = args.token

    if args.precision == "16":
        model_kwargs["torch_dtype"] = torch.float16
    elif args.precision == "8":
        model_kwargs["load_in_8bit"] = True
    elif args.precision == "4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    tokenizer_path = args.tokenizer if args.tokenizer else args.model_id
    tokenizer_kwargs = {}
    if args.token:
        tokenizer_kwargs["token"] = args.token
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    print("모델과 토크나이저 로드 완료.")
    return model, tokenizer

def process_batch(
    batch: List[Dict[str, Any]], 
    retriever: Retriever, 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    args: argparse.Namespace
) -> None:
    prompts = []
    for item in batch:
        query = item['input']['question']
        retrieved_docs = retriever.search(query, top_n=args.top_n)
        context = "\n\n".join(retrieved_docs)
        prompts.append(create_prompt(query, context))

    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_token_type_ids=False
    ).to(args.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    terminators = [token_id for token_id in terminators if token_id is not None]
    
    # 수정: 모든 점수의 균형을 맞추기 위한 '통제된 샘플링' 방식 적용
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        do_sample=True,          # 자연스러운 문장 생성을 위해 True로 설정
        temperature=0.3,        # 매우 낮은 온도로 설정하여 사실상 결정적(deterministic)으로 작동하게 제어
        top_p=0.9
    )

    generated_tokens = outputs[:, model_inputs.input_ids.shape[1]:]
    output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for i, item in enumerate(batch):
        raw_answer = output_texts[i].strip()
        clean_answer = postprocess_answer(raw_answer)
        item["output"] = {"answer": clean_answer}


def main(args: argparse.Namespace) -> None:
    model, tokenizer = load_dependencies(args)
    retriever = Retriever(args.knowledge_base)

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: 입력 파일 '{input_path}'이(가) 유효한 JSON 형식이 아닙니다.")
        return

    total_steps = (len(input_data) + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(total=total_steps, desc="RAG 처리 중")

    for i in range(0, len(input_data), args.batch_size):
        batch = input_data[i:i + args.batch_size]
        process_batch(batch, retriever, model, tokenizer, args)
        progress_bar.update(1)
    
    progress_bar.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nRAG 파이프라인 실행 완료! 결과가 '{output_path}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_rag", description="배치 처리를 지원하는 RAG 파이프라인을 실행합니다.")

    g = parser.add_argument_group("파일 경로")
    g.add_argument("--input", type=str, default="../../data/korean_language_rag_V1.0_test.json", help="입력 질문 파일 (JSON)")
    g.add_argument("--output", type=str, default="../../data/result_rag_output.json", help="출력 답변 파일 (JSON)")
    g.add_argument("--knowledge_base", type=str, default="../../data/labeling.json", help="참조할 지식 문서 (.docx)")

    g = parser.add_argument_group("모델 및 실행 환경")
    g.add_argument("--model_id", type=str, default="../../data/qwen3-8B-korean-grammar-expert-merged2", help="Hugging Face 모델 ID 또는 파인튜닝된 모델 경로")
    g.add_argument("--tokenizer", type=str, help="Hugging Face 토크나이저 경로 (지정하지 않으면 model_id와 동일)")
    g.add_argument("--device", type=str, default="cuda", help="모델을 로드할 장치 (예: cuda, cpu)")
    g.add_argument("--token", type=str, help="Gated 모델 접근을 위한 Hugging Face 토큰")
    g.add_argument("--precision", type=str, default="16", choices=["16", "8", "4"], help="모델 로딩 정밀도 (16: float16, 8: int8, 4: nf4)")

    g = parser.add_argument_group("RAG 및 생성 파라미터")
    g.add_argument("--top_n", type=int, default=1, help="검색기에서 가져올 관련 문서의 수")
    g.add_argument("--batch_size", type=int, default=8, help="한 번에 처리할 데이터의 수")
    g.add_argument("--max_length", type=int, default=2048, help="토크나이저의 최대 입력 길이")
    
    args = parser.parse_args()
    main(args)