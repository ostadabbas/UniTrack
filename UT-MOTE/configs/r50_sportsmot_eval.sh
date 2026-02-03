#!/bin/bash

EXP_DIR=exps/e2e_motr_r50_sportsmot

python3 eval_sportsmot.py \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --resume /home/nail/Documents/MOTR/model_weights/model_dance_final.pth \
    --mot_path /home/nail/Documents/MOTR/SportsMOT_example \
    --device cuda
