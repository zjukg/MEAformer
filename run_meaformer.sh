# 600 250
CUDA_VISIBLE_DEVICES=0,1,2,3 python  main.py \
            --gpu           $1    \
            --eval_epoch    1  \
            --only_test     0   \
            --model_name    MEAformer \
            --data_choice   $2 \
            --data_split    $3 \
            --data_rate     $4 \
            --epoch         1000 \
            --lr            5e-4  \
            --hidden_units  "300,300,300" \
            --save_model    0 \
            --batch_size    3500 \
            --semi_learn_step 5 \
	        --csls          \
	        --csls_k        3 \
	        --random_seed   42 \
            --exp_name      IJCAI_MEAformer_sf_$5 \
            --exp_id        v1_$3_$4 \
            --workers       12 \
            --dist          0 \
            --accumulation_steps 1 \
            --scheduler     cos \
            --il            \
	    --il_start      500 \
            --attr_dim      300     \
            --img_dim       300     \
            --name_dim      300     \
            --char_dim      300     \
            --hidden_size   300     \
            --tau           0.1     \
            --tau2          4.0     \
            --structure_encoder "gat" \
            --num_attention_heads 1 \
            --num_hidden_layers 1 \
            --use_surface   $5     \
            --use_intermediate 1   \
            --replay 0
            # --unsup \
            # --unsup_mode $8 \
            # --unsup_k 3000  \
