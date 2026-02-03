#!/bin/bash
# ------------------------------------------------------------------------
# Modified from r50_motr_eval.sh for SportsMOT dataset
# ------------------------------------------------------------------------

EXP_DIR=exps/sportsmot_results
python eval.py \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --with_box_refine \
    --resume exps/e2e_motr_r50_joint/motr_final.pth \
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
    --mot_path /home/nail/Documents/MOTR/SportsMOT_example \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --resume exps/e2e_motr_r50_joint/motr_final.pth
