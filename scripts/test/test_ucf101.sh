CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main.py \
        --arch resnet34 --num_segments 8 --seed 1993 \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 1 --fine_tune_epochs 1 --training --testing \
        --train_batch-size 8 --test_batch-size 16 --exemplar_batch-size 16 -j 8 \
        --start_task 0 --exp 0215002 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 51 --nb_class 10 --K 40 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --t_div --lambda_2 1e-3 --cbf \
        --hs_lr 1e-3 \
        --modality RGB --scaler 100 \
        --workers 0 \
        --exemplar_path ./checkpoint/ucf101/exemplar --num_per_class_list 51 --exemplar_iteration 1
