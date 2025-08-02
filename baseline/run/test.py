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

# `src` 디렉토리가 경로에 없으면 추가
project_path = str(Path(__file__).resolve().parent.parent)
if project_path not in sys.path:
    sys.path.append(project_path)
from src.retriever import Retriever

# --- 프롬프트 템플릿 정의 ---
# test.py 파일 상단의 PROMPT_TEMPLATES 딕셔너리를 아래 내용으로 전체 교체하세요.

PROMPT_TEMPLATES = PROMPT_TEMPLATES = {
    "교정형": """### 역할:
당신은 주어진 '참고 문서'를 기반으로 '질문'에 답변하는 한국어 어문 규범 전문가입니다.

# 지시사항
1. '참고 문서'와 당신의 지식을 종합하여 질문에 대한 답을 찾으세요.
2. 답변은 반드시 아래 '좋은 답변의 예시'와 같이 "{{올바른 문장}}이/가 옳다. {{이유}}" 형식으로 작성하세요.
3. 이유는 핵심적인 근거만 간결하게 1~2 문장으로 요약하여 설명해야 합니다.
4. '잘못된 답변의 예시'처럼 명백히 틀린 사실을 생성하지 마세요.
5. 만약 질문에 대한 답을 찾을 수 없다면, "정보를 찾을 수 없습니다."라고만 답변하세요.

# 좋은 답변의 예시
질문: "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\\n\\"해당 사업은 아직 진행중입니다.\\""
답변: "\"해당 사업은 아직 진행 중입니다.\"가 옳다. '진행 중'의 '중'은 의존 명사이므로 앞말과 띄어 써야 한다."

질문: "다음 문장이 어문 규범에 부합하도록 문장 부호를 추가하고, 그 이유를 설명하세요.\\n― 이번 추석 연휴(9. 16.(월)~9. 18.(수))에는 기숙사가 휴관합니다."
답변: "\"이번 추석 연휴[9. 16.(월)~9. 18.(수)]에는 기숙사가 휴관합니다.\"가 옳다. 괄호 안에 또 괄호를 써야 할 때 바깥쪽의 괄호는 대괄호를 사용한다."

---
# 잘못된 답변의 예시
질문: "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\\n\\"숫사자는 머리에 갈기가 있다.\\""
답변: "\"수수사는 머리에 갈기가 있다.\"가 옳다."  # (X) '숫사자'는 올바른 표기이며, '수사자'로 써야 합니다. '수수사'는 완전히 잘못된 단어입니다.

질문: "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\\n\\"할머니 집에 게신가요?\\""
답변: "\"해서 온 길목에 가신가요?가 옳다." # (X) 질문의 의도와 전혀 상관없는 문장을 생성한 잘못된 답변입니다. '계신가요'가 올바른 표현입니다.

---
### 참고 문서:
{context}

### 질문:
{query}

### 답변:
""",
    "선택형": """### 역할:
당신은 주어진 '참고 문서'를 기반으로 '질문'에 답변하는 한국어 어문 규범 전문가입니다.

# 지시사항
1. '참고 문서'와 당신의 지식을 종합하여 질문에 대한 답을 찾으세요.
2. 주어진 보기 중에서 가장 적절한 것을 선택하여 "~가 옳다." 형태로 답변하고, 그 이유를 설명하세요.
3. 이유는 핵심적인 근거만 간결하게 1~2 문장으로 요약하여 설명해야 합니다.
4. '잘못된 답변의 예시'처럼 명백히 틀린 사실을 이유로 들지 마세요.
5. 만약 질문에 대한 답을 찾을 수 없다면, "정보를 찾을 수 없습니다."라고만 답변하세요.

# 좋은 답변의 예시
질문: "\"그 집은 {{부모 자식간/부모 자식 간}} 정이 두터운 것 같더라.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
답변: "\"그 집은 부모 자식 간 정이 두터운 것 같더라.\"가 옳다. '간'은 '사이'의 뜻을 나타내는 의존 명사이므로 앞말과 띄어 쓴다."

질문: "\"나는 {{몰티즈/말티즈}}를 한 마리 키운다.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
답변: "\"나는 말티즈를 한 마리 키운다.\"가 옳다. 외래어 표기법에 따라 'Maltese'는 '말티즈'로 적는 것이 올바른 표기이다."

---
# 잘못된 답변의 예시
질문: "\"막내가 {{빈대떡/빈자떡}}을 둥글넓적하게 만들었다.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
답변: "\"빈자떡\"이 옳다."  # (X) '빈대떡'이 표준어이므로 사실과 다른 잘못된 답변입니다.

질문: "\"{{프라이팬/후라이팬}}을 사야 한다.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
답변: "\"후라이팬\"이 옳다." # (X) 외래어 표기법에 따라 'f'는 'ㅍ'으로 적으므로 '프라이팬'이 올바른 표기입니다.

---
### 참고 문서:
{context}

### 질문:
{query}

### 답변:
"""
}
def create_prompt(query: str, context: str, q_type: str) -> str:
    """질문 유형(q_type)에 따라 적절한 프롬프트를 반환합니다."""
    template = PROMPT_TEMPLATES.get(q_type, PROMPT_TEMPLATES["교정형"])
    return template.format(context=context, query=query)

def postprocess_answer(text: str) -> str:
    """
    개선된 후처리 함수: 정규표현식을 사용하여 답변 형식을 안정적으로 추출합니다.
    """
    text = text.strip()
    
    if "### 답변:" in text:
        text = text.split("### 답변:", 1)[-1].strip()
    if "답변:" in text:
        text = text.split("답변:", 1)[-1].strip()

    m = re.search(r'([\"“`])(.*?)\1\s*가\s*옳다', text)
    if m:
        quote_content = m.group(2).strip()
        corrected_sentence = f'"{quote_content}"가 옳다.'
        explanation = text[m.end():].strip(' .')
        if explanation:
            return f"{corrected_sentence} {explanation}"
        return corrected_sentence

    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return "답변을 생성하지 못했습니다." 
    return text

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
        q_type = item['input'].get('question_type', '교정형')
        
        retrieved_docs = retriever.search(query, top_n=args.top_n)
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            meta = doc.get('metadata') if isinstance(doc, dict) else {}
            content = doc.get('content') if isinstance(doc, dict) else str(doc)
            doc_info = f"[참고 문서 {i+1}: {meta.get('category_2', '')} {meta.get('regulation_id', '')} - {meta.get('content_type', '')}]"
            context_parts.append(f"{doc_info}\n{content}")
        
        context = "\n\n".join(context_parts)
        prompts.append(create_prompt(query, context, q_type))

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
    
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.05,
        temperature=0.7
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

    # 결과를 새로 저장할 리스트
    final_results = []
    for i in range(0, len(input_data), args.batch_size):
        # 원본 데이터의 복사본으로 배치 생성
        batch = [item.copy() for item in input_data[i:i + args.batch_size]]
        process_batch(batch, retriever, model, tokenizer, args)
        final_results.extend(batch)
        progress_bar.update(1)
    
    progress_bar.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # 새로 생성된 결과 리스트를 저장
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"\nRAG 파이프라인 실행 완료! 결과가 '{output_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_rag", description="배치 처리를 지원하는 RAG 파이프라인을 실행합니다.")

    g = parser.add_argument_group("파일 경로")
    g.add_argument("--input", type=str, default="../../data/korean_language_rag_V1.0_test.json", help="입력 질문 파일 (JSON)")
    g.add_argument("--output", type=str, default="../../data/result_rag_output.json", help="출력 답변 파일 (JSON)")
    g.add_argument("--knowledge_base", type=str, default="../../data/labeling.json", help="참조할 지식 문서 (JSON)")

    g = parser.add_argument_group("모델 및 실행 환경")
    g.add_argument("--model_id", type=str, default="../../data/qwen3-8B-korean-grammar-expert-merged2", help="Hugging Face 모델 ID 또는 파인튜닝된 모델 경로")
    g.add_argument("--tokenizer", type=str, help="Hugging Face 토크나이저 경로 (지정하지 않으면 model_id와 동일)")
    g.add_argument("--device", type=str, default="cuda", help="모델을 로드할 장치 (예: cuda, cpu)")
    g.add_argument("--token", type=str, help="Gated 모델 접근을 위한 Hugging Face 토큰")
    g.add_argument("--precision", type=str, default="16", choices=["16", "8", "4"], help="모델 로딩 정밀도 (16: float16, 8: int8, 4: nf4)")

    g = parser.add_argument_group("RAG 및 생성 파라미터")
    g.add_argument("--top_n", type=int, default=3, help="검색기에서 가져올 관련 문서의 수")
    g.add_argument("--batch_size", type=int, default=8, help="한 번에 처리할 데이터의 수")
    g.add_argument("--max_length", type=int, default=2048, help="토크나이저의 최대 입력 길이")
    
    args = parser.parse_args()
    main(args)