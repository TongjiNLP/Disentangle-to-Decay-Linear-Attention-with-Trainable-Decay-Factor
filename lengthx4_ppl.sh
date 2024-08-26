#!/bin/bash

# 设置参数变量
SAVE_PATH_TEST="/path/to/test/dataset"
MODEL_PATH="/path/to/model"
GPT_PATH="/path/to/gpt"
PPL_FILE_PATH="/path/to/save/ppl.txt"

# 调用 Python 脚本并传递参数
accelerate launch lengthx4_ppl.py \
  --save_path_test $SAVE_PATH_TEST \
  --model_path $MODEL_PATH \
  --gpt_path $GPT_PATH \
  --ppl_file_path $PPL_FILE_PATH
