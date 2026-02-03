# ------------------------------------------------------------------------
# Modified and add the copyrights as well to Institution (Author)
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.

# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

EXP_DIR=/home/author/ondemand/dev/MOTE/motr_exp_mot20
OUT_DIR=/work/Institution/author/results/MOTE_results
python3 submit20.py \
    --meta_arch mote \
    --dataset_file e2e_joint \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${EXP_DIR}/checkpoint.pth \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 50 90 150 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'ETEM' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/mot20.train \
    --data_txt_path_val ./datasets/data_path/mot20.train \
    --resume ${EXP_DIR}/checkpoint.pth \
    --exp_name pub_submit_20