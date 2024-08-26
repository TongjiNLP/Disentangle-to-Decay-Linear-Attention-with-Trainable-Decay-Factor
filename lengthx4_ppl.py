import math
import datasets
from transformers import Trainer, GPT2LMHeadModel, DataCollatorForLanguageModeling, \
    TrainingArguments, GPT2Tokenizer, TrainerCallback, TrainerState, TrainerControl, \
    GPT2Model, GPT2Config
# 导入自定义模型
from model.LinearEluWithD2D import LinearEluWithD2DLMHeadModel
from model.LinearExpWithD2D import LinearExpWithD2DLMHeadModel
from model.LinearEluWithRoPE import LinearEluWithRoPELMHeadModel
from model.LinearExpWithRoPE import LinearExpWithRoPELMHeadModel
from model.LinearEluWithALiBi import LinearEluWithALiBiLMHeadModel
from model.LinearExpWithALiBi import LinearExpWithALiBiLMHeadModel
from model.LinearEluWithVanillaAPE import LinearEluWithVanillaAPELMHeadModel
from model.LinearExpWithVanillaAPE import LinearExpWithVanillaAPELMHeadModel


# 设置命令行参数
parser = argparse.ArgumentParser(description="Evaluate the model on a test dataset")
parser.add_argument("--save_path_test", type=str, required=True, help="Path to the saved test dataset")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
parser.add_argument("--gpt_path", type=str, required=True, help="Path to GPT model or tokenizer")
parser.add_argument("--ppl_file_path", type=str, default="ppl/Bailu_ppl.txt", help="Path to save the perplexity result")

args = parser.parse_args()

# 使用命令行参数
dataset_path = args.save_path_test
model_path = args.model_path
gpt_path = args.gpt_path
token_path = gpt_path
ppl_file_path = args.ppl_file_path

if __name__ == "__main__":
    test_dataset = datasets.load_from_disk(dataset_path)
    ten_percent = int(0.1 * len(test_dataset))
    subset = test_dataset.select(range(ten_percent))
    config = GPT2Config.from_pretrained(model_path)
    config.n_positions = 2048
    model = MoonStoneEluALiBiGPT2LMHeadModel.from_pretrained(model_path, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained(token_path)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    training_args = TrainingArguments(
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        save_strategy="epoch",
        output_dir="./results",
        evaluation_strategy="steps"
    )

    trainer = Trainer(
        data_collator=data_collator,
        model=model,
        args=training_args,
        eval_dataset=subset
    )

    output = trainer.evaluate()
    ppl = math.exp(output["eval_loss"])
    print(ppl)

    with open(ppl_file_path, 'w') as file:
        file.write(f'Model Path: {model_path}\n')
        file.write(f'Perplexity: {ppl}\n')

    print(f'Perplexity {ppl} has been saved to {ppl_file_path}')