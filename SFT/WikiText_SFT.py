from transformers import Trainer, GPT2LMHeadModel, DataCollatorForLanguageModeling, \
    TrainingArguments, GPT2Tokenizer, TrainerCallback, TrainerState, TrainerControl
from JingyuanDeShenjun.MoonStoneEluPlusGPT import MoonStoneEluPlusGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneEluALiBi import MoonStoneEluALiBiGPT2LMHeadModel
from JingyuanDeShenjun.TrainingEluALiBi import TrainingEluALiBiGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneWithRopeWithoutWPE_Refactored import MoonStoneRopeRefactoredModel
from datasets import load_dataset
import json
import os
import torch
from datasets import Dataset, load_from_disk


data_path = "wikitext-103"
model_name = "ALiBiElu"
tokenizer_path = "gpt"
model_path = "/data/hubblezcy/Projects/Pinwell@0818/model/MoonStoneALiBiElu/checkpoint-178507"
log_path = "LogFile"
exp_name = "ALiBiElu_SFT"
result_path = "wiki_train_result"
test_path = "./ALiBiElu_wiki_result.txt"
output_dir = "wiki_dataset_disk"
dataset_from_disk = False


tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


def load_local_wikitext2(data_dir):
    # 文件路径
    file_paths = {
        "train": os.path.join(data_dir, "wiki.train.tokens"),
        "test": os.path.join(data_dir, "wiki.test.tokens"),
        "validation": os.path.join(data_dir, "wiki.valid.tokens")
    }
    datasets = {split: Dataset.from_text(file_path) for split, file_path in file_paths.items()}
    return datasets


if model_name == "D2DElu":
    model = MoonStoneEluPlusGPT2LMHeadModel.from_pretrained(model_path)
elif model_name == "ALiBiElu":
    model = MoonStoneEluALiBiGPT2LMHeadModel.from_pretrained(model_path)
elif model_name == "TrainingALiBi":
    model = TrainingEluALiBiGPT2LMHeadModel.from_pretrained(model_path)

if dataset_from_disk:
    train_dataset = load_from_disk(os.path.join(output_dir, "train"))
    valid_dataset = load_from_disk(os.path.join(output_dir, "valid"))
    test_dataset = load_from_disk(os.path.join(output_dir, "test"))
else:
    datas = load_local_wikitext2(data_path)
    # 重新生成训练集的映射
    train_dataset = datas["train"].map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
        batched=True, remove_columns=["text"], num_proc=6)

    # 重新生成验证集和测试集的映射
    valid_dataset = datas["validation"].map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
        batched=True, remove_columns=["text"], num_proc=6)

    test_dataset = datas["test"].map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
        batched=True, remove_columns=["text"], num_proc=6)

    os.makedirs(output_dir, exist_ok=True)
    # 保存映射后的数据集
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    valid_dataset.save_to_disk(os.path.join(output_dir, "valid"))
    test_dataset.save_to_disk(os.path.join(output_dir, "test"))

print("Length of train_dataset:", len(train_dataset))
print("Length of valid_dataset:", len(valid_dataset))
print("Length of test_dataset:", len(test_dataset))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
training_args = TrainingArguments(
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    num_train_epochs=1,  # ****set to 1
    logging_dir=os.path.join(log_path, exp_name),
    logging_steps=1,  # ****** set to
    save_strategy="epoch",
    output_dir=os.path.join(result_path, exp_name),
    eval_steps=40,
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


# Include callback in the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
print(f"Perplexity: {perplexity}")

with open(test_path, 'w') as f:
    f.write(f"Perplexity: {perplexity}\n")

trainer.train()

eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
print(f"Perplexity: {perplexity}")

with open(test_path, 'w') as f:
    f.write(f"Perplexity: {perplexity}\n")