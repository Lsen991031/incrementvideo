CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main.py \
        --arch resnet50 --num_segments 8 --seed 1000 \
        --gd 20 --lr 1e-4 --lr_steps 20 30 --epochs 50 --training --testing \
        --train_batch-size 8 --test_batch-size 8 --exemplar_batch-size 8 -j 8 \
        --start_task 0 --exp 3 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 26 --nb_class 1 --K 5 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 1e-3 --cbf \
        --dataset hmdb51 \
        -scaler 100 \
        --modality Zip --root_log /input/zhukai/ddp_tcp/log/ \
        --root_model /input/zhukai/ddp_tcp/checkpoint

