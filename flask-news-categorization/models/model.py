import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# List available datasets
datasets_path = os.path.join(os.getcwd(), 'datasets')
datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

# Load and combine all datasets
dataframes = []
required_columns = {'news_headline', 'news_article', 'news_category'}
for dataset in datasets:
    dataset_path = os.path.join(datasets_path, dataset)
    data = pd.read_csv(dataset_path)
    if required_columns.issubset(data.columns):
        dataframes.append(data)
    else:
        print(f"Skipping {dataset} as it does not contain the required columns.")

if not dataframes:
    raise ValueError("No datasets with the required columns found.")

combined_data = pd.concat(dataframes, ignore_index=True)

# Preprocess data
def preprocess_data(data):
    data['text'] = data['news_headline'] + " " + data['news_article']
    data = data.rename(columns={'news_category': 'category'})
    return data[['text', 'category']]

processed_data = preprocess_data(combined_data)

# Encode labels
label_encoder = LabelEncoder()
processed_data['label'] = label_encoder.fit_transform(processed_data['category'])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(processed_data)

# Split into train and test sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # Limit to 1 epoch for faster training
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",  # Disable saving checkpoints to save time
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./news_classification_model")
tokenizer.save_pretrained("./news_classification_model")

# Test the model
def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_label])[0]

# Test the classification
#sample_article = "The World Cup final was a thrilling match, with both teams displaying exceptional skill and determination. The game ended in a dramatic penalty shootout."
#predicted_category = classify_news(sample_article)
#print(f"The news belongs to the {predicted_category} category.")