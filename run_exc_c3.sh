#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=electra-exc-12-16-large-tokentest
export OUTPUT_DIR=$CURRENT_DIR/check_points
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="exc_c3"
 
python exc.py \
  --model_type="large" \
  --gpu_ids="2" \
  --num_train_epochs=12 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --a_dropout_prob=0.1 \
  --h_dropout_prob=0.1 \
  --s_dropout_prob=0.1 \
  --learning_rate=1e-6 \
  --warmup_proportion=0.1 \
  --max_seq_length=512 \
  --do_train \
  --do_eval \
  --gradient_accumulation_steps=16 \
  --task_name=$TASK_NAME \
  --data_dir=$GLUE_DIR/c3/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
