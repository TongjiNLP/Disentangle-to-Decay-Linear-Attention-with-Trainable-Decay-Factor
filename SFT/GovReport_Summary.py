from transformers import Trainer, GPT2LMHeadModel, DataCollatorForLanguageModeling, \
    TrainingArguments, GPT2Tokenizer, TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset, Dataset, load_from_disk
import json
import os
import torch

data_path = "govreport"
model_name = "ALiBiElu"
tokenizer_path = "gpt"
model_path = "/data/hubblezcy/Projects/Pinwell@0818/model/MoonStoneALiBiElu/checkpoint-178507"
log_path = "LogFile"
exp_name = "ALiBiElu_SFT"
result_path = "govreport_train_result"
test_path = "./ALiBiElu_govreport_result.txt"
output_dir = "govreport_dataset_disk"
dataset_from_disk = True

tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


# Processing function for summary SFT
def process_summary_data(examples, max_length):
    # Extract document and summary
    documents = [json.loads(example)["document"] for example in examples["text"]]
    summaries = [json.loads(example)["summary"] for example in examples["text"]]
    
    # Combine document and summary into the required format
    inputs = [f"Summary the following document : '{doc}'. summary result: {summary}" 
              for doc, summary in zip(documents, summaries)]
    
    # Tokenize and truncate
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=max_length)


# Wrapper function for tokenization with max_length
def tokenize_function(max_length):
    return lambda examples: process_summary_data(examples, max_length=max_length)


# Model loading
if model_name == "D2DElu":
    model = MoonStoneEluPlusGPT2LMHeadModel.from_pretrained(model_path)
elif model_name == "ALiBiElu":
    model = MoonStoneEluALiBiGPT2LMHeadModel.from_pretrained(model_path)
elif model_name == "TrainingALiBi":
    model = TrainingEluALiBiGPT2LMHeadModel.from_pretrained(model_path)


# Load dataset from disk or file
if dataset_from_disk:
    train_dataset = load_from_disk(os.path.join(output_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(output_dir, "valid"))
    test_dataset = load_from_disk(os.path.join(output_dir, "test"))
else:
    datas = load_dataset('text', data_files={'train': f'{data_path}/train/train.txt',
                                             'validation': f'{data_path}/valid/valid.txt',
                                             'test': f'{data_path}/test/test.txt'})
    
    # Map function for tokenization
    train_dataset = datas["train"].map(
        tokenize_function(max_length=512),
        batched=True,
        remove_columns=["text"],
        num_proc=6
    )

    valid_dataset = datas["validation"].map(
        tokenize_function(max_length=512),
        batched=True,
        remove_columns=["text"],
        num_proc=6
    )

    test_dataset = datas["test"].map(
        tokenize_function(max_length=1024),
        batched=True,
        remove_columns=["text"],
        num_proc=6
    )

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    valid_dataset.save_to_disk(os.path.join(output_dir, "valid"))
    test_dataset.save_to_disk(os.path.join(output_dir, "test"))

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    num_train_epochs=1,  # Set to 1 for quick tests
    logging_dir=os.path.join(log_path, exp_name),
    logging_steps=1,
    save_strategy="epoch",
    output_dir=os.path.join(result_path, exp_name),
    eval_steps=40000000000,
    evaluation_strategy="steps",
    save_total_limit=1,
    gradient_accumulation_steps=2,
    report_to="tensorboard",
    logging_first_step=True,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    warmup_steps=4,
    weight_decay=0.1,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False
)


# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Trainer training
# trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
print(f"Perplexity: {perplexity}")

# Save evaluation result to a file
with open(test_path, 'w') as f:
    f.write(f"Perplexity: {perplexity}\n")

# Generate summaries for testing after training
def generate_summary(model, tokenizer, text, max_length=512):
    inputs = tokenizer(f"Summary the following document : '{text}'. summary result:", return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test the summary generation
sample_document = "This is a test document that needs to be summarized."
generated_summary = generate_summary(model, tokenizer, sample_document)
print(f"Generated Summary: {generated_summary}")

# Write test results to file
with open(test_path, 'a') as f:
    f.write(f"\nGenerated Summary: {generated_summary}\n")
