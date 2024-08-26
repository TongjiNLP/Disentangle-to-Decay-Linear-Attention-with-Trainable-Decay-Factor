# Disentangle to Decay: Linear Attention with Trainable Positional Decay for Length Extrapolation :rocket:

This repository contains the implementation of the paper "Disentangle to Decay: Linear Attention with Trainable Positional Decay for Length Extrapolation".  We design a trainable positional encoding for linear attention that can freely transform between RPE and APE, and improve its stability by disentanglement. Model with this encoding achieves better performance than existing PE for linear attention.

## :white_check_mark: Abstract

Transformer architecture has significantly advanced Natural Language Processing (NLP) by delivering unparalleled performance. However, it faces challenges with efficiency and processing long sequences, attributed to its quadratic time complexity. Linear attention offers a more efficient linear time solution but falls short in language modeling and length extrapolation compared with traditional Transformer. To enhance the performance of linear attention and fully leverage its capability in modeling long sequences, we begin with positional encoding, specifying the constraints required for positional encoding by linear attention. Based on these constraints, we design a positional encoding for linear attention, named Disentangle to Decay (D2D), which allows for a seamless conversion between absolute positional encoding (APE) and relative positional encoding (RPE). To alleviate the instability of directly training D2D, we disentangle D2D into the combination of RPE and APE, which greatly improves the stability while ensuring the efficiency of model training. Experiments result shows that, application of D2D in linear attention significantly improves performance in language modeling and length extrapolation, demonstrating competitiveness with the vanilla Transformer and other positional encodings.

## :fuelpump: Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## :factory: Models

You can find our model code under the model folder. **LinearEluWithD2D.py** and **LinearExpWithD2D.py** implement the transformation from linear attention to RNN. When training, these two models will be parallelized using the transformer architecture, and when inference, these two models will be transformed into RNN serial predictions.

## :electric_plug: Usage

To train the model with the default configuration, run:

```bash
#!/bin/bash

#SBATCH -J xxxxx ##定义作业名称
#SBATCH -p xxxxx ##指定队列名称
#SBATCH --nodes=x
#SBATCH --cpus-per-task=x
#SBATCH --gres=xxxxxx ##每个节点使用 gpu 卡数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end # 作业结束时，邮件提醒
#SBATCH --mail-user=xxxxx #邮箱地址
accelerate launch Training.py
```

File **train.sh** is also provided for a quick training.

## :ledger: Dataset

We train and evaluate our model on OpenWebText.  You can download it at https://huggingface.co/datasets/Skylion007/openwebtext.

## :crystal_ball: Evaluation

To evaluate the model on a specific dataset, you can run code in SFT file.

## :black_nib: Test Generate

If you want to test the generation of your model after training, you can use  **test_generate.py** for model inference
