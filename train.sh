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
