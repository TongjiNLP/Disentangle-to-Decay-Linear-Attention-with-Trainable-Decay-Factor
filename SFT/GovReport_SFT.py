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


data_path = "govreport"
model_name = "ALiBiElu"
tokenizer_path = "gpt"
model_path = "/data/hubblezcy/Projects/Pinwell@0818/model/MoonStoneALiBiElu/checkpoint-178507"
# model_path = "/data/hubblezcy/Projects/Pinwell@0818/govreport_train_result/TrainingALiBi_SFT/checkpoint-109"
log_path = "LogFile"
exp_name = "ALiBiElu_SFT"
result_path = "govreport_train_result"
test_path = "./ALiBiElu_govreport_result.txt"
output_dir = "govreport_dataset_disk"
dataset_from_disk = True


tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


def process_data(examples, max_length):
    # 解析 JSON 数据
    reports = [json.loads(example)["report"] for example in examples["text"]]
    # 对 report 部分进行 tokenization，使用外部传入的 max_length
    return tokenizer(reports, truncation=True, padding="max_length", max_length=max_length)


# 定义一个 wrapper 函数，用来传递 max_length 参数
def tokenize_function(max_length):
    return lambda examples: process_data(examples, max_length=max_length)


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
    datas = load_dataset('text', data_files={'train': f'{data_path}/train/train.txt',
                                             'validation': f'{data_path}/valid/valid.txt',
                                             'test': f'{data_path}/test/test.txt'})
    # 重新生成训练集的映射
    train_dataset = datas["train"].map(
        tokenize_function(max_length=512),  # 例如，这里设置 max_length 为 256
        batched=True,
        remove_columns=["text"],
        num_proc=6
    )

    # 重新生成验证集和测试集的映射
    valid_dataset = datas["validation"].map(
        tokenize_function(max_length=512),  # 例如，这里设置 max_length 为 256
        batched=True,
        remove_columns=["text"],
        num_proc=6
    )
    test_dataset = datas["test"].map(
        tokenize_function(max_length=1024),  # 例如，这里设置 max_length 为 256
        batched=True,
        remove_columns=["text"],
        num_proc=6
    )
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


# Include callback in the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# trainer.train()

eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
print(f"Perplexity: {perplexity}")

with open(test_path, 'w') as f:
    f.write(f"Perplexity: {perplexity}\n")