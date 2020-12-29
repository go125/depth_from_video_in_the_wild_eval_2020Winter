
# depth_from_video_in_the_wild_eval_2020Winter

## Data Preparation (KITTI)

- [ImageNet Checkpoint Preparation](https://github.com/dalgu90/resnet-18-tensorflow)

### Input example (KITTI)

```script
nohup python GenDataKITTI_gray.py \
--HEIGHT 128 \
--WIDTH 256 \
--OUTPUT_DIR /home/ubuntu/data/kitti_result_all_20201228 \
--TEMP_DIR /home/ubuntu/data/train_data_example_all_20201228/ &
```

## Train example (KITTI)
- 白黒で訓練
- 下記データでAbs Rel Errorが0.1374まで下がるか確認

```script
nohup python -m depth_from_video_in_the_wild.train \
--img_height 128 \
--img_width 256 \
--data_dir /home/ubuntu/data/kitti_result_all_20201228 \
--checkpoint_dir=/home/ubuntu/data/kitti_experiment_checkpoint_20201228 \
--imagenet_ckpt=/home/ubuntu/data/ResNet18/model.ckpt \
--train_steps=1000000 &
```

## Inference Example (KITTI)

```shell
python inference_dfv.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_list_file /home/ubuntu/data/raw_data_KITTI/test_files_eigen.txt \
    --output_dir /home/ubuntu/data/result_20201225_143940_1229test/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20201225/model-143940
```

```shell
python inference_dfv.py \
    --img_height 128 \
    --img_width 256 \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_list_file /home/ubuntu/data/raw_data_KITTI/test_files_eigen_gray.txt \
    --output_dir /home/ubuntu/data/result_20201228_14394/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20201228/model-14394
```

### Getting Abs Rel Error (KITTI)

```shell
python kitti_eval/eval_depth.py --kitti_dir=/home/ubuntu/data/raw_data_KITTI/ --pred_file=/home/ubuntu/data/result_20201223_273486/result.npy
```

- アスペクト比変更オプション不使用
- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.1374,     0.9873,     5.5315,     0.2212,     0.0000,     0.8166,     0.9388,     0.9754 ,   12.3922
  - この出力は7月実行時と同じ結果である(このコードは過去のコードと等価)
  
```shell
python kitti_eval/eval_depth.py --kitti_dir=/home/ubuntu/data/raw_data_KITTI/ --pred_file=/home/ubuntu/data/result_20201225_143940/result.npy
```

- アスペクト比変更オプション不使用
- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.1305,     0.9316,     5.3069,     0.2099,     0.0000,     0.8309,     0.9460,     0.9788 ,    8.1981 

```shell
python kitti_eval/eval_depth.py --kitti_dir=/home/ubuntu/data/raw_data_KITTI/ --pred_file=/home/ubuntu/data/result_20201225_143940_1229test/result.npy
```

- アスペクト比変更オプション使用
- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.1324,     0.9724,     5.3728,     0.2126,     0.0000,     0.8290,     0.9444,     0.9781 ,    8.1621 

```shell
python kitti_eval/eval_depth.py --kitti_dir=/home/ubuntu/data/raw_data_KITTI/ --pred_file=/home/ubuntu/data/result_20201228_14394/result.npy
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.1537,     1.2137,     6.1761,     0.2229,     0.0000,     0.7847,     0.9323,     0.9779 ,   14.8086 
  
## Finetuning with the video taken in Saitama

## 1. Use StereoAVIToPNG.py

```
nohup python StereoAVIToPNG.py \
--path_avi /home/ubuntu/data/StereoVideo/V2-mv-20200716103312-ulrg.avi \
--path_output_png /home/ubuntu/data/Sayama/all_video/video1top_png/ \
--option top \
--fps 10 &
```

```
nohup python StereoAVIToPNG.py \
--path_avi /home/ubuntu/data/StereoVideo/V2-mv-20200716103312-ulrg.avi \
--path_output_png /home/ubuntu/data/Sayama/all_video/video1middle_png/ \
--option middle \
--fps 10 &
```

```
nohup python StereoAVIToPNG.py \
--path_avi /home/ubuntu/data/StereoVideo/V2-mv-20200716105152-ulrg.avi \
--path_output_png /home/ubuntu/data/Sayama/all_video/video2top_png/ \
--option top \
--fps 10 &
```

```
nohup python StereoAVIToPNG.py \
--path_avi /home/ubuntu/data/StereoVideo/V2-mv-20200716105152-ulrg.avi \
--path_output_png /home/ubuntu/data/Sayama/all_video/video2middle_png/ \
--option middle \
--fps 10 &
```

## 2 Use CropPNG.py

```script
nohup python CropPNG.py --base_path /home/ubuntu/data/Sayama/all_video/ \
--WIDTH 416 \
--HEIGHT 128 \
--OUTPUT_DIR /home/ubuntu/data/Sayama/out \
--TEMP_DIR /home/ubuntu/data/Sayama/tmpdir &
```

## 3 Use MakeMask.py
- "all video_training" dir should include only "video2top_png" dir.
  - "video2top_png"に対してのみトレーニング用のマスクを生成

```script
nohup python MakeMask.py --base_path /home/ubuntu/data/Sayama/all_video_training/ \
--ROOT_DIR ../Mask_RCNN \
--WIDTH 416 \
--HEIGHT 128 \
--OUTPUT_DIR /home/ubuntu/data/Sayama/training_data \
--TEMP_DIR /home/ubuntu/data/Sayama/tmpdir_training &
```

## 4. Training

```script
nohup python -m depth_from_video_in_the_wild.train \
--data_dir /home/ubuntu/data/Sayama/training_data \
--checkpoint_dir=/home/ubuntu/data/kitti_experiment_checkpoint_20201224 \
--imagenet_ckpt=/home/ubuntu/data/ResNet18/model.ckpt \
--train_steps=1000000 &
```

## Evaluation

### Before fine tuning

### Getting Predicted Depth

```shell
python inference_dfv.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_dir /home/ubuntu/data/Sayama/tmpdir/2020_08_04/video1top_png/image_02/data/ \
    --output_dir /home/ubuntu/Sayama/result_video1top_273486/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20200716/model-273486
```

### Getting Abs Rel Error

下記ファイルはステレオカメラ専用

```
python AbsRelError.py
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.2797,     3.7270,     9.6290,     0.3576,     0.0000,     0.5507,     0.8217,     0.9325 ,   14.2422
  - これは8月と同じ数値

### After fine tuning

### Getting Predicted Depth

```shell
python inference_dfv.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_dir /home/ubuntu/data/Sayama/tmpdir/2020_08_04/video1top_png/image_02/data/ \
    --output_dir /home/ubuntu/Sayama/result_video1top_279296/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20200716/model-279296
```

### Getting Abs Rel Error

以下2ファイルはステレオカメラ専用

```
python AbsRelError.py \
--depth_map_dir /home/ubuntu/Sayama/result_video1top_279296/
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.2258,     3.0248,     8.5670,     0.3028,     0.0000,     0.6844,     0.8930,     0.9543 ,   14.0935 
  - これは8月と同じ出力

```
python AbsRelError_NoScaleMatching.py \
--depth_map_dir /home/ubuntu/Sayama/result_video1top_279296/
```
- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.4028,     5.3890,    12.5421,     0.5733,     0.0000,     0.1633,     0.4877,     0.7393 ,    9.1144
  - これは8月と同じ出力

### Visualization

```
python Visualization.py 
```
