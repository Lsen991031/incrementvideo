# 重大进展
addp_tcp_trash1文件夹下运行
bash 8gpus_sthv2_1e-3_bsz88_hs100_1993.sh \
bash 8gpus_sthv2_1e-3_bsz88_hs100_19935.sh \
bash 4gpus_sthv2_1e-3_bsz88_hs100_19935.sh \
bash 4gpus_sthv2_1e-3_bsz88_hs100_19932-22.sh

# 2-20重要任务
分别将addp_tcp_trash1和addp_tcp_trash2文件夹下面的ops/dataset_config.py的 \
ROOT_DATASET = '/home/ls/mmaction2-old/data/'更换为存放数据的文件夹
### （1）测试sh
addp_tcp_trash1和addp_tcp_trash2文件夹分别是两个任务
分别运行两个文件夹下的ceshi.sh

### （2）对比试验
addp_tcp_trash1文件夹下运行
bash 8gpus_hmdb_1e-3_bsz88_hs100_1993_hslr1e-4.sh

addp_tcp_trash2文件夹下运行
8gpus_hmdb_1e-3_bsz88_hs100_1993_hslr1e-4_duibi.sh


# 环境配置
conda create -n ls python=3.8 \
conda activate ls \
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch \
pip install -r requirments

# 2-16重要任务
### （1）测试sh
bash scripts/test/test_hmdb.sh \
bash scripts/test/test_sthv2.sh \
bash scripts/test/test_ucf101.sh

### （2）ucf
bash scripts/2-16/ucf/2/4gpus_UCF101_1e-3_bsz28_2021.sh \
bash scripts/2-16/ucf/5/4gpus_UCF101_1e-3_bsz28_2021.sh \
bash scripts/2-16/sthv2/10/4gpus_sthv2_1e-3_bsz28_2021.sh \
bash scripts/2-16/hmdb/5/4gpus_hmdb_1e-3_bsz28_2021.sh


# 2-16日任务(所有任务在incrementvideo-git目录下打开命令行)

### （1）测试sh
bash scripts/test/test_hmdb.sh \
bash scripts/test/test_sthv2.sh \
bash scripts/test/test_ucf101.sh

### （2）ucf101数据集任务
bash scripts/2-16/ucf/2/4gpus_UCF101_1e-3_bsz28_1000.sh \
bash scripts/2-16/ucf/2/4gpus_UCF101_1e-3_bsz28_1993.sh \
bash scripts/2-16/ucf/2/4gpus_UCF101_1e-3_bsz28_2021.sh \
bash scripts/2-16/ucf/5/4gpus_UCF101_1e-3_bsz28_1000.sh \
bash scripts/2-16/ucf/5/4gpus_UCF101_1e-3_bsz28_1993.sh \
bash scripts/2-16/ucf/5/4gpus_UCF101_1e-3_bsz28_2021.sh \
bash scripts/2-16/ucf/10/4gpus_UCF101_1e-3_bsz28_1000.sh \
bash scripts/2-16/ucf/10/4gpus_UCF101_1e-3_bsz28_1993.sh \
bash scripts/2-16/ucf/10/4gpus_UCF101_1e-3_bsz28_2021.sh 
### （3）HMDB51数据集任务
bash scripts/2-16/hmdb/1/4gpus_hmdb_1e-3_bsz28_1000.sh \
bash scripts/2-16/hmdb/1/4gpus_hmdb_1e-3_bsz28_1993.sh \
bash scripts/2-16/hmdb/1/4gpus_hmdb_1e-3_bsz28_2021.sh \
bash scripts/2-16/hmdb/5/4gpus_hmdb_1e-3_bsz28_1000.sh \
bash scripts/2-16/hmdb/5/4gpus_hmdb_1e-3_bsz28_1993.sh \
bash scripts/2-16/hmdb/5/4gpus_hmdb_1e-3_bsz28_2021.sh 
### （4）sthv2数据集任务
bash scripts/2-16/sthv2/5/4gpus_sthv2_1e-3_bsz28_1000.sh \
bash scripts/2-16/sthv2/5/4gpus_sthv2_1e-3_bsz28_1993.sh \
bash scripts/2-16/sthv2/5/4gpus_sthv2_1e-3_bsz28_2021.sh \
bash scripts/2-16/sthv2/10/4gpus_sthv2_1e-3_bsz28_1000.sh \
bash scripts/2-16/sthv2/10/4gpus_sthv2_1e-3_bsz28_1993.sh \
bash scripts/2-16/sthv2/10/4gpus_sthv2_1e-3_bsz28_2021.sh


# 一、数据下载和处理（HMDB51）
## 1、下载mmaction2
下面的链接是处理数据的开源仓库，利用该开源库下载和处理数据（将这个仓库放在数据存储的目录下，下载和处理的数据会生成在这个仓库的某文件夹下）
仓库链接：https://github.com/open-mmlab/mmaction2
## 2、下载数据集
下载上述仓库后，解压缩生成MMACTION2文件夹
进入 MMACTION2/tools/data/hmdb51/ 文件夹，并打开终端
运行如下命令：
(1) 准备注释
```
bash download_annotations.sh
```
(2) 准备视频
```
bash download_videos.sh
```
## 3、抽取RGB
OpenCV 提取RGB 帧
```
bash extract_rgb_frames_opencv.sh
```
## 4、生成文件列表
```
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```
## 5、文件目录如下（最终的生成目录如下，mmaction2\data文件夹下就是数据）
```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── hmdb51
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi

│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0.avi
│   │   ├── rawframes
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0
│   │   │   │   ├── ...
│   │   │   │   ├── winKen_wave_u_cm_np1_ri_bad_1
```
## 6、修改代码中数据的路径
(1) 下载incrementvideo-git项目
把步骤5的hmdb51的上一层data路径添加到incrementvideo-git\ops\dataset_config.py文件的ROOT_DATASET变量上即可

# 二、数据下载和处理（Something-SomethingV2）
## 1、下载mmaction2
利用github开源库下载和处理数据（下载位置不限）
仓库链接：https://github.com/open-mmlab/mmaction2
## 2、下载数据集
在MMACTION2/data/下创建sthv2文件夹，并将下述文件下载到sthv2文件夹中 \
百度网盘链接：https://pan.baidu.com/s/1gCkSaQ5idG3aqgoiQlogEg?pwd=9d8u 提取码: 9d8u
在MMACTION2/data/sthv2文件夹下运行：
```
cat 20bn-something-something-v2-?? | tar zx
```
并把文件夹名字20bn-something-something-v2改为videos 

## 3、抽取RGB
进入MMACTION2/tools/data/sthv2/文件夹 \
OpenCV 提取RGB 帧
```
bash extract_rgb_frames_opencv.sh
```
## 4、生成文件列表
将tcd项目tools文件夹下的gen_label_sthv2.py文件复制到MMACTION2/data/sthv2文件夹下，打开命令行，运行：
```
python gen_label_sthv2.py
```
## 5、文件目录如下
```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv2
|   |   ├── 原文件
|   |   ├── val_videofolder.txt
│   │   ├── train_videofolder.txt
│   │   ├── test_videofolder.txt
│   │   ├── category.txt
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2
│   |   |   ├── ...

```
## 6、修改代码中数据的路径
把步骤5的hmdb51的上一层data路径添加到incrementvideo-git\ops\dataset_config.py文件的ROOT_DATASET变量上即可

# 训练注意事项
#### 1、batchsize调节
亲测batchsize=2的时候在第一个task达到4000MB显存占用，task=2的时候达到8000/9000MB
#### 2、ddp调节
多卡，在run.sh文件中调节
#### 3、训练
```
bash run.sh
```

# 三、数据下载和处理（UCF101）
## 1、下载mmaction2
利用github开源库下载和处理数据（下载位置不限）
仓库链接：https://github.com/open-mmlab/mmaction2
## 2、下载数据集
进入 MMACTION2/tools/data/ucf101/ 文件夹
运行如下命令：
(1) 准备注释
```
bash download_annotations.sh
```
(2) 准备视频
```
bash download_videos.sh
```
## 3、抽取RGB
OpenCV 提取RGB 帧
```
bash extract_rgb_frames_opencv.sh
```
## 4、生成文件列表
```
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```
## 5、文件目录如下
```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05

```
## 6、修改代码中数据的路径
把步骤5的hmdb51的上一层data路径添加到incrementvideo-git\ops\dataset_config.py文件的ROOT_DATASET变量上即可