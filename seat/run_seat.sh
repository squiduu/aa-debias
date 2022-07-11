TESTS=seat3,seat3b,seat4,seat5,seat5b,seat6,seat6b,seat7,seat7b,seat8,seat8b,seat9,seat10
MODEL_NAME=bert
SAVE_DIR=./out/
SEED=42
DATA_DIR=./data/tests/
NUM_SAMPLES=100000
RUN_NO=run01
CKPT_DIR=./ckpts/aa_debiased/bert-base-uncased_gender_epoch:10
VERSION=bert-base-uncased

# run this file at my_xai directory
python ./seat/seat_main.py \
    --tests ${TESTS} \
    --model_name ${MODEL_NAME} \
    --seed ${SEED} \
    --log_dir ${SAVE_DIR} \
    --results_path ${SAVE_DIR}${MODEL_NAME}_seat_${RUN_NO}.csv \
    --ignore_cached_encs \
    --data_dir ${DATA_DIR} \
    --exp_dir ${SAVE_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --run_no ${RUN_NO} \
    --use_ckpt \
    --ckpt_dir ${CKPT_DIR} \
    --version ${VERSION}