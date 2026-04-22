from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_id = "VolodymyrPugachov/BioClinicalBERT-Triage"

# load model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

texts = [
    "Patient reports chest pain and shortness of breath.",
    "Routine follow-up visit for blood pressure check.",
    "Mild headache for two days, no other symptoms."
]

for t in texts:
    print("\n", t)
    print(classifier(t))