# ------------------------------------------------------------------------
# Modified and add the copyrights as well to Institution (Author)
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.

# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

EXP_DIR=mote_exp_mot17
python3 demo.py \
    --meta_arch mote \
    --dataset_file e2e_joint \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${EXP_DIR}/mote_final.pth \
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
    --query_interaction_layer 'ETEM' \
    --extra_track_attn \
    --resume ${EXP_DIR}/mote_final.pth \
    --input_video figs/demo.avi