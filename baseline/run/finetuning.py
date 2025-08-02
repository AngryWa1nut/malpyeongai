import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# 수정: EarlyStoppingCallback을 사용하기 위해 import
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
# 수정: SFTTrainer와 함께 SFTConfig를 import 합니다.
from trl import SFTTrainer, SFTConfig

# 전역 변수 tokenizer 선언 (format_instruction 함수에서 사용)
tokenizer = None

def format_instruction(sample: dict) -> list[str]:
    """
    데이터셋의 각 항목을 모델 학습에 적합한 프롬프트 형식으로 변환합니다.
    SFTTrainer는 내부적으로 map을 사용하므로 배치 처리가 필요합니다.
    """
    # 답변 끝에 EOS 토큰을 추가하여 문장 종료 지점을 명확하게 학습시킵니다.
    return [
        f"""### Human:
{inp['question']}

### Assistant:
{outp['answer']}{tokenizer.eos_token}"""
        for inp, outp in zip(sample["input"], sample["output"])
    ]

def main(args):
    # ======================================================================================
    # 데이터셋 준비
    # ======================================================================================
    print("데이터셋을 로드하고 훈련/검증용으로 분리합니다...")
    
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split['train']
    eval_dataset = train_val_split['test']

    print(f"  - 훈련 데이터: {len(train_dataset)}개")
    print(f"  - 검증 데이터: {len(eval_dataset)}개")
    
    # ======================================================================================
    # 모델 및 토크나이저 로드
    # ======================================================================================
    print(f"'{args.base_model_id}' 모델과 토크나이저를 로드합니다...")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=args.hf_token,
    )
    
    # 전역 변수 tokenizer에 객체를 할당
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ======================================================================================
    # LoRA 설정
    # ======================================================================================
    print("PEFT를 위한 LoRA 설정을 구성합니다...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ======================================================================================
    # 훈련 설정 및 실행
    # ======================================================================================
    print("훈련을 시작합니다...")

    # 수정: TrainingArguments 대신 SFTConfig를 사용합니다.
    # 기존 TrainingArguments의 모든 인자와 SFTTrainer의 인자를 여기에 통합합니다.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        load_best_model_at_end=True,
        disable_tqdm=False,
        # SFTTrainer에 있던 인자들을 SFTConfig로 이동
    )

    # 수정: SFTTrainer 호출이 더 간결해집니다.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        args=training_args, # 여기에 SFTConfig 객체를 전달
        formatting_func=format_instruction, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        
    )

    trainer.train()

    # ======================================================================================
    # 훈련된 모델 저장
    # ======================================================================================
    print("훈련이 완료되었습니다. 최적의 모델을 저장합니다.")
    
    final_model_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_model_path))
    
    print(f"파인튜닝된 모델 어댑터가 '{final_model_path}'에 성공적으로 저장되었습니다.")
    print("이제 이 경로를 RAG 파이프라인의 모델 ID로 사용하여 로드할 수 있습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM을 LoRA로 파인튜닝하는 스크립트")

    parser.add_argument("--dataset_path", type=str, default="../../data/korean_language_rag_V1.0_train.json", help="훈련 데이터셋 JSON 파일 경로")
    parser.add_argument("--base_model_id", type=str, default="upstage/SOLAR-10.7B-Instruct-v1.0", help="파인튜닝할 기본 모델의 Hugging Face ID")
    parser.add_argument("--output_dir", type=str, default="../../data/SOLAR-10.7B-Instruct-v1.0-grammar-expert", help="훈련 결과물이 저장될 디렉토리")
    
    parser.add_argument("--epochs", type=int, default=5, help="훈련 에포크 수 (EarlyStopping을 사용하므로 넉넉하게 설정)")
    parser.add_argument("--batch_size", type=int, default=2, help="훈련 배치 크기")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="모델이 처리할 시퀀스의 최대 길이")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face 인증 토큰 (필요 시)")

    args = parser.parse_args()
    main(args)
