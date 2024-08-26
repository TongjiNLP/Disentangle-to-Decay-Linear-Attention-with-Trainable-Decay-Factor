from accelerate.utils import LoggerType
from datasets import load_dataset, load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig
from transformers import Trainer, GPT2LMHeadModel, DataCollatorForLanguageModeling, \
    TrainingArguments, GPT2Tokenizer, TrainerCallback, TrainerState, TrainerControl
from torch.utils.data import DataLoader, Dataset, IterableDataset

loss_function = CrossEntropyLoss()


UseSpecialModels = True
Train_from_sctrach = True
# Each Exp,**modify this ! below**
model_type = "MoonStoneEluPlus" # ["Pinwell","LinearAtt","SuJianLin","GELUGPT","DecayGPT","BailuGPT"]
ExpName = "MoonStoneEluPlusFull"
# Each Exp,**modify this ! above**
descprition = "MoonStoneEluPlusFull with Elu kernel linear attention"
desc_path = "/share/home/tj07149/Pinwell/descs/"
log_path = "/share/home/tj07149/Pinwell/logs/"
result_path = "/share/home/tj07149/Pinwell/train_result/"

gpt_path = "/share/home/tj07149/Pinwell/gpt"

token_path = gpt_path

from JingyuanDeShenjun.EluGPT import EluGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneEluALiBi import MoonStoneEluALiBiGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneALiBiGPT import MoonStoneALiBiGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneEluPlusGPT import MoonStoneEluPlusGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneWithRopeWithoutWPE_Refactored import MoonStoneRopeRefactoredModel
from JingyuanDeShenjun.TrainingEluALiBi import TrainingEluALiBiGPT2LMHeadModel
save_path_train = "/share/home/tj07149/mapdata/train"
save_path_valid = "/share/home/tj07149/mapdata/valid/shard_0"

class DynamicShardedDataset(IterableDataset):
    def __init__(self, base_path, num_shards):
        self.base_path = base_path
        self.num_shards = num_shards
        self.shards_paths = [f"{base_path}/shard_{i}" for i in range(num_shards)]

    def __len__(self):
        return 91398757

    def __iter__(self):
        for shard_path in self.shards_paths:
            shard = load_from_disk(shard_path)
            for example in shard:
                # 直接yield预处理好的tokenized数据
                yield {"input_ids": example['input_ids'], "attention_mask": example['attention_mask']}


if __name__ == "__main__":
    """
    token donnot change in When using Jingyuanizer & dataset_prepare
    """
    tokenizer = GPT2Tokenizer.from_pretrained(token_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = DynamicShardedDataset(save_path_train, 10)
    valid_dataset = load_from_disk(save_path_valid)

    if Train_from_sctrach:
        config = AutoConfig.from_pretrained(gpt_path)
        if UseSpecialModels:
            if model_type == "EluGPT":
                model = EluGPT2LMHeadModel(config)
            elif model_type == "MoonStoneALiBiElu":
                model = MoonStoneEluALiBiGPT2LMHeadModel(config)
            elif model_type == "MoonStoneALiBi":
                model = MoonStoneALiBiGPT2LMHeadModel(config)
            elif model_type == "MoonStoneEluPlus":
                model = MoonStoneEluPlusGPT2LMHeadModel(config)
            elif model_type == "MoonStoneRopeRefactored":
                model = MoonStoneRopeRefactoredModel(config)
            elif model_type == "TrainingEluALiBi":
                model = TrainingEluALiBiGPT2LMHeadModel(config)
            else:
                raise NotImplementedError("Please specify models in list")
        else:
            model = GPT2LMHeadModel(config)
    else:
        if UseSpecialModels:
            raise NotImplementedError("Please specify models in list")
        else:
            model = GPT2LMHeadModel.from_pretrained(gpt_path)
    # 打印模型参数的数据类型
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    # 定义训练参数
    with open(desc_path + ExpName + ".txt", "w", encoding="utf-8") as fout:
        fout.write(descprition + "\n")

    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        logging_dir=log_path + ExpName,
        logging_steps=80,
        save_strategy="epoch",
        output_dir=result_path + ExpName,
        eval_steps=4000,
        evaluation_strategy="steps",
        save_total_limit=10,
        gradient_accumulation_steps=8,
        report_to="tensorboard",
        logging_first_step=True,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        warmup_steps=3_000,
        weight_decay=0.1,
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        bf16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    trainer.train()
