python ada_debias_v2.py \
    --run_type debias \
    --data_dir ../data/debias/ \
    --debias_type gender \
    --model_name_or_path bert-base-uncased \
    --seed 42 \
    --per_device_train_batch_size 512 \
    --dataloader_num_workers 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 50 \
    --output_dir ./out/ \
    --run_name run00