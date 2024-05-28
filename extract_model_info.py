from transformers import AutoModelForSequenceClassification, OPTForSequenceClassification

# Load the model
model = OPTForSequenceClassification.from_pretrained('facebook/opt-6.7b')

# Calculate the number of total trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(trainable_params)

# for name, layer in model.named_modules():
#     print(name, layer)