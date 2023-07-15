import random
from transformers import pipeline, TrainingArguments, Trainer

# Fine-tuning BERT
class SentimentDataset(Dataset):
    # ...

# Load pre-trained models and tokenizers
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

model_roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')

model_xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
tokenizer_xlnet = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Create an instance of the SentimentDataset for each model
dataset_bert = SentimentDataset(texts, labels, tokenizer_bert)
dataset_roberta = SentimentDataset(texts, labels, tokenizer_roberta)
dataset_xlnet = SentimentDataset(texts, labels, tokenizer_xlnet)

# ...

# Model Ensembling
def ensemble_predict(models, tokenizers, texts):
    all_predictions = []

    for text in texts:
        input_ids = []
        attention_masks = []

        for model, tokenizer in zip(models, tokenizers):
            encoding = tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(encoding['input_ids'].squeeze())
            attention_masks.append(encoding['attention_mask'].squeeze())

        input_ids = torch.stack(input_ids).to(device)
        attention_masks = torch.stack(attention_masks).to(device)

        predictions = []
        for model in models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                predicted_label_idx = torch.argmax(logits, dim=1).item()
                predicted_label = ['negative', 'positive'][predicted_label_idx]
                predictions.append(predicted_label)

        ensemble_prediction = Counter(predictions).most_common(1)[0][0]
        all_predictions.append(ensemble_prediction)

    return all_predictions

# Train and fine-tune multiple models with different architectures
model_bert_1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_bert_2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model_roberta_1 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model_roberta_2 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

model_xlnet_1 = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
model_xlnet_2 = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

models = [model_bert_1, model_bert_2, model_roberta_1, model_roberta_2, model_xlnet_1, model_xlnet_2]
tokenizers = [tokenizer_bert, tokenizer_bert, tokenizer_roberta, tokenizer_roberta, tokenizer_xlnet, tokenizer_xlnet]

# ...

# Perform ensembling on the validation set
val_texts, val_labels = read_dataset_from_file('validation.txt')
val_predictions = ensemble_predict(models, tokenizers, val_texts)
val_accuracy = accuracy_score(val_labels, val_predictions)

print(f"Ensemble Validation Acc: {val_accuracy:.2%}")

# Function to perform sentiment analysis using ensembled models
def analyze_sentiment_ensemble(text):
    predictions = ensemble_predict(models, tokenizers, [text])
    sentiment = predictions[0]
    return sentiment

# Example usage
text = "I really enjoyed the movie!"
sentiment = analyze_sentiment_ensemble(text)
print(f"Sentiment: {sentiment}")

# Advanced Technique: Hyperparameter Tuning
hyperparameters = {
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'num_train_epochs': [3, 4, 5],
    'per_device_train_batch_size': [16, 32],
    'weight_decay': [0.01, 0.1],
}

best_accuracy = 0
best_hyperparameters = {}

for learning_rate in hyperparameters['learning_rate']:
    for num_train_epochs in hyperparameters['num_train_epochs']:
        for per_device_train_batch_size in hyperparameters['per_device_train_batch_size']:
            for weight_decay in hyperparameters['weight_decay']:
                training_args = TrainingArguments(
                    output_dir='./results',
                    evaluation_strategy='epoch',
                    learning_rate=learning_rate,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=16,
                    num_train_epochs=num_train_epochs,
                    weight_decay=weight_decay,
                    load_best_model_at_end=True,
                )

                trainer = Trainer(
                    model=model_bert,
                    args=training_args,
                    train_dataset=dataset_bert,
                    eval_dataset=val_dataset,
                )

                # Train the model using the trainer
                trainer.train()

                # Evaluate the model on the validation set
                eval_result = trainer.evaluate(eval_dataset=val_dataset)
                val_accuracy = eval_result['eval_accuracy']

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_hyperparameters = {
                        'learning_rate': learning_rate,
                        'num_train_epochs': num_train_epochs,
                        'per_device_train_batch_size': per_device_train_batch_size,
                        'weight_decay': weight_decay,
                    }

print("Best Hyperparameters:")
print(best_hyperparameters)
print(f"Best Validation Accuracy: {best_accuracy:.2%}")

# Advanced Technique: Data Augmentation (Back-translation)
def backtranslate(text, target_lang='fr'):
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-' + target_lang)
    translated_text = translator(text, max_length=512)[0]['translation_text']
    backtranslated_text = translator(translated_text, max_length=512, source_lang=target_lang)[0]['translation_text']
    return backtranslated_text

augmented_texts = []
augmented_labels = []

for text, label in zip(texts, labels):
    augmented_text = backtranslate(text)
    augmented_texts.append(augmented_text)
    augmented_labels.append(label)

augmented_texts = texts + augmented_texts
augmented_labels = labels + augmented_labels

# Create an instance of the SentimentDataset for augmented data
augmented_dataset = SentimentDataset(augmented_texts, augmented_labels, tokenizer_bert)

# Combine the original and augmented datasets
combined_dataset = ConcatDataset([dataset_bert, augmented_dataset])

# ...

# Advanced Technique: Training Pipeline
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=best_hyperparameters['learning_rate'],
    per_device_train_batch_size=best_hyperparameters['per_device_train_batch_size'],
    per_device_eval_batch_size=16,
    num_train_epochs=best_hyperparameters['num_train_epochs'],
    weight_decay=best_hyperparameters['weight_decay'],
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=combined_dataset,  # Use the combined dataset for training
    eval_dataset=val_dataset,
)

# Train the model using the trainer
trainer.train()

# Evaluate the model on the validation set
eval_result = trainer.evaluate(eval_dataset=val_dataset)
val_accuracy = eval_result['eval_accuracy']

print(f"Trained Model Validation Acc: {val_accuracy:.2%}")
