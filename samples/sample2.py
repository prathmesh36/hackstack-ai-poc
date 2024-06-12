from transformers import AutoModel, AutoTokenizer

model_name = "deepset/roberta-base-squad2"  # Example model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")