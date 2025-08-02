import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 기본 모델 ID와 저장된 LoRA 어댑터 경로 설정
base_model_id = "Qwen/Qwen3-8B"
adapter_path = "../../data/SOLAR-10.7B-Instruct-v1.0-grammar-expert/final"
save_path = "../../data/SOLAR-10.7B-Instruct-v1.0-grammar-expert-merged" # 병합된 모델을 저장할 새 경로

# 2. 기본 모델과 토크나이저 로드
print(f"'{base_model_id}' 에서 기본 모델을 로드합니다...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu", # 병합은 CPU에서 수행해도 충분합니다.
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. PeftModel을 로드하여 LoRA 어댑터 적용
print(f"'{adapter_path}' 에서 어댑터를 적용합니다...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 4. LoRA 어댑터를 기본 모델에 병합
print("모델과 어댑터를 병합하는 중...")
model = model.merge_and_unload()
print("병합 완료!")

# 5. 병합된 모델과 토크나이저를 새로운 경로에 저장
print(f"병합된 모델을 '{save_path}'에 저장합니다...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("성공적으로 완료되었습니다. 이제 RAG 스크립트에서 새 모델 경로를 사용할 수 있습니다.")
