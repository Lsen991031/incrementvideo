CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
        --arch resnet50 --num_segments 8 --seed 1993 \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --fine_tune_epochs 20 --training --testing \
        --train_batch-size 8 --test_batch-size 16 --exemplar_batch-size 16 -j 8 \
        --start_task 0 --exp 216002 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 84 --nb_class 5 --K 40 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --t_div --lambda_2 1e-3 --cbf \
        --hs_lr 1e-3 \
        --dataset somethingv2 \
        --modality RGB --scaler 100
        --workers 0 \
        --exemplar_path ./checkpoint/somethingv2/exemplar --num_per_class_list 84
