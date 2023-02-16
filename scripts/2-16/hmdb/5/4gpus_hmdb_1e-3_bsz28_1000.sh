CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 25961 main.py \
        --arch resnet50 --num_segments 8 --seed 1000 \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --fine_tune_epochs 20 --training --testing \
        --train_batch-size 16 --test_batch-size 16 --exemplar_batch-size 64 -j 8 \
        --start_task 0 --exp 216001 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 26 --nb_class 5 --K 40 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --t_div --lambda_2 1e-3 --cbf \
        --hs_lr 1e-3 \
        --dataset hmdb51 \
        --modality RGB --scaler 100 \
        --workers 0 --weight-decay 5e-5 \
        --exemplar_path ./checkpoint/hmdb51/exemplar --num_per_class_list 26
