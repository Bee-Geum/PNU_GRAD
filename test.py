import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 저장된 모델과 토크나이저 불러오기
model = AutoModelForSequenceClassification.from_pretrained('./final_model')
tokenizer = AutoTokenizer.from_pretrained('./final_model')

# 예시 입력 문장
input_text = input("지금의 기분을 문장으로 표현하면? : ")

# 토큰화
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 모델을 평가 모드로 전환
model.eval()

# 입력을 모델에 전달하고 예측 결과 얻기
with torch.no_grad():
    outputs = model(**inputs)

# 예측 결과
logits = outputs.logits
predicted_class = logits.argmax().item()

mood = None

if predicted_class == 0:
    mood = "공포가"
elif predicted_class == 1:
    mood = "놀람이"
elif predicted_class == 2:
    mood = "분노가"
elif predicted_class == 3:
    mood = "슬픔이"
elif predicted_class == 4:
    mood = "중립이"
elif predicted_class == 5:
    mood = "행복이"
elif predicted_class == 6:
    mood = "혐오가"

print(f">> 입력하신 내용에서 {mood} 느껴집니다.")