CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29504 main.py \
        --arch resnet34 --num_segments 8 --seed 1000 \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --training --testing \
        --train_batch-size 8 --test_batch-size 8 --exemplar_batch-size 8 -j 8 \
        --start_task 0 --exp 1128000 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 51 --nb_class 5 --K 5 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 1e-3 --cbf \
        --modality RGB --scaler 100 \
        --modality Zip --root_log /input/zhukai/ddp_tcp/log/ \
        --root_model /input/zhukai/ddp_tcp/checkpoint
# --fine_tune_epochs 1