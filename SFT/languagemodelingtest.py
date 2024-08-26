import argparse
import torch
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from JingyuanDeShenjun.MoonStoneEluPlusGPT import MoonStoneEluPlusGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneEluALiBi import MoonStoneEluALiBiGPT2LMHeadModel
from JingyuanDeShenjun.MoonStoneWithRopeWithoutWPE_Refactored import MoonStoneRopeRefactoredModel
from enwiki8.load_dataset import load as load_enwik8


def main():
    parser = argparse.ArgumentParser(description='Test a model on a specified dataset')
    parser.add_argument('--model_name', type=str, default='D2DElu', help='Name of the model to use')
    parser.add_argument('--model_path', type=str, help='Path to the pre-trained model')
    parser.add_argument('--dataset_name', type=str, default='enwik8', help='Name of the dataset to use')
    parser.add_argument('--result_path', type=str, default='./evaluation_results.txt', help='Path to save the evaluation results')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt")
    tokenizer.pad_token = tokenizer.eos_token

    if args.model_name == "D2DElu":
        model = MoonStoneEluPlusGPT2LMHeadModel.from_pretrained(args.model_path)
    elif args.model_name == "AliBiElu":
        model = MoonStoneEluALiBiGPT2LMHeadModel.from_pretrained(args.model_path)
    elif args.model_name == "RoPEElu":
        model = MoonStoneRopeRefactoredModel.from_pretrained(args.model_path)
    model.to(device)

    if args.dataset_name == "enwik8":
        dataset = load_enwik8()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=64,
        do_train=False,
        do_eval=True,
        logging_dir="./logs",
        logging_strategy="steps",  # 设置为每步都记录日志
        logging_steps=1,  # 每1步记录一次日志
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=dataset,
        tokenizer=tokenizer
    )

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
    print(f"Perplexity: {perplexity}")

    with open(args.result_path, 'w') as f:
        f.write(f"Perplexity: {perplexity}\n")


if __name__ == "__main__":
    main()
