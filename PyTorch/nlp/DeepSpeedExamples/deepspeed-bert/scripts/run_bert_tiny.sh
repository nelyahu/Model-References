#!/bin/bash

###############################################################################################
# Example: Pretraining phase 1 of BERT Tiny parameters on single HLS1 box with 1 device.
###############################################################################################

# Params: run_pretraining
DATA_BASE_PATH=/mnt/weka/data
DATA_DIR=${DATA_BASE_PATH}/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/
MODEL_CONFIG=./scripts/bert_tiny_config.json
ZERO_STAGE=${HL_ZERO_STAGE:-0}
DS_CONFIG=./scripts/deepspeed_bert_tiny_zero${ZERO_STAGE}.json
RESULTS_DIR=./results/bert_tiny_zero${ZERO_STAGE}
MAX_SEQ_LENGTH=128
NUM_STEPS_PER_CP=100000
MAX_STEPS=100000
RUN_STEPS=1000
LR=0.001
WARMUP=0.2843
LOG_FREQ=100

# Params: DeepSpeed
NUM_NODES=1
NGPU_PER_NODE=${HL_NUM_CARD_PER_NODE:-8}

CMD="python -u ./run_pretraining.py \
     --steps_this_run=$RUN_STEPS \
     --disable_progress_bar \
     --resume_from_checkpoint \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$RESULTS_DIR/checkpoints \
     --seed=12439 \
     --optimizer=adamw \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=20 \
     --max_steps=$MAX_STEPS \
     --warmup_proportion=$WARMUP \
     --num_steps_per_checkpoint=$NUM_STEPS_PER_CP \
     --learning_rate=$LR \
     --log_freq=$LOG_FREQ \
     --deepspeed \
     --deepspeed_config=$DS_CONFIG"

mkdir -p $RESULTS_DIR

deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $CMD |& tee $RESULTS_DIR/output.txt
