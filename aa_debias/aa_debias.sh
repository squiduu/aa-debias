MODEL_VERSION=bert-base-uncased
MODEL_NAME=bert
SEED=42
BATCH_SIZE=128
DEBIAS_TYPE=gender
RUN_NO=run02
DATA_DIR=./data/
MAX_PROMPT_LEN=5
TOP_K=100
NUM_WORKERS=4
LR=5e-6
ACCUMULATE_GRAD_BATCHES=1
GRADIENT_CLIP_VAL=1.0
MAX_EPOCHS=10
OUTPUT_DIR=./out/
GPUS=2
PATIENCE=10
CKPT_DIR=./ckpts/
PRECISION=16
FINETUNE_VOCAB=None
SAVE_DIR=aa_debiased/
MODE=aa-debias

python aa_debias.py \
    --model_version ${MODEL_VERSION} \
    --model_name ${MODEL_NAME} \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --debias_type ${DEBIAS_TYPE} \
    --run_no ${RUN_NO} \
    --data_dir ${DATA_DIR} \
    --max_prompt_len ${MAX_PROMPT_LEN} \
    --top_k ${TOP_K} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --max_epochs ${MAX_EPOCHS} \
    --output_dir ${OUTPUT_DIR} \
    --gpus ${GPUS} \
    --patience ${PATIENCE} \
    --ckpt_dir ${CKPT_DIR} \
    --precision ${PRECISION} \
    --finetune_vocab ${FINETUNE_VOCAB} \
    --save_dir ${SAVE_DIR} \
    --mode ${MODE}
