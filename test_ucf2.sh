CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main.py \
        --arch resnet34 --num_segments 8 --seed 2000 \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --fine_tune_epochs 20 --training --testing \
        --train_batch-size 4 --test_batch-size 2 --exemplar_batch-size 2 -j 8 \
        --start_task 0 --exp 1214021114 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames entire \
        --cl_type DIST --cl_method OURS --init_task 51 --nb_class 2 --K 6 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --t_div --lambda_2 1e-3 --cbf \
        --hs_lr 1e-3 \
        --modality RGB --scaler 100 \
        --workers 8

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main.py \
#         --arch resnet34 --num_segments 8 --seed 2000 --wandb \
#         --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --fine_tune_epochs 201 --training --testing \
#         --train_batch-size 4 --test_batch-size 2 --exemplar_batch-size 2 -j 8 \
#         --start_task 0 --exp 1214025 --exemplar --nme \
#         --test_crops 5 --loss_type nll --store_frames uniform \
#         --cl_type DIST --cl_method OURS --init_task 51 --nb_class 10 --K 5 \
#         --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
#         --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
#         --t_div --lambda_2 1e-3 --cbf \
#         --hs_lr 1e-3 \
#         --modality RGB --scaler 100 \
#         --workers 8
