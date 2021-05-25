
# depth_from_video_in_the_wild_eval_2020Winter

## Data Preparation (KITTI)

- [ImageNet Checkpoint Preparation](https://github.com/go125/resnet-18-tensorflow)
- GenDataKITTI_gray.py requires [MaskRCNN](https://github.com/matterport/Mask_RCNN) in the same dir.

```script
nohup python GenDataKITTI_gray.py \
--HEIGHT 128 \
--WIDTH 256 \
--OUTPUT_DIR /home/ubuntu/data/kitti_result_all_20201228 \
--TEMP_DIR /home/ubuntu/data/train_data_example_all_20201228/ &
```

## Train example (KITTI)

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
    --img_height 128 \
    --img_width 256 \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_list_file /home/ubuntu/data/raw_data_KITTI/test_files_eigen_gray.txt \
    --output_dir /home/ubuntu/data/result_20201228_143940/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20201228/model-143940
```

## Getting Abs Rel Error (KITTI)

```shell
python kitti_eval/eval_depth.py --kitti_dir=/home/ubuntu/data/raw_data_KITTI/ --pred_file=/home/ubuntu/data/result_20201228_143940/result.npy
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.1253,     1.0206,     5.6762,     0.2004,     0.0000,     0.8414,     0.9482,     0.9808 ,   16.1354 


## Finetuning with the video taken in Saitama

## 1. Use StereoAVIToPNG.py

```
nohup python StereoAVIToPNG.py \
--path_avi /home/ubuntu/data/StereoVideo/V2-mv-20200716103312-ulrg.avi \
--path_output_png /home/ubuntu/data/Sayama/all_video/video1top_png/ \
--option top \
--fps 10 &
```

## 2 Use CropPNG.py

```script
nohup python CropPNG.py --base_path /home/ubuntu/data/Sayama/all_video/ \
--WIDTH 256 \
--HEIGHT 128 \
--OUTPUT_DIR /home/ubuntu/data/Sayama/out_128_256 \
--TEMP_DIR /home/ubuntu/data/Sayama/tmpdir_128_256 &
```

## 3 Use MakeMask.py
- "all video_training" dir should include only "video2top_png" dir.
  - Make training mask for images only in "video2top_png" dir.

```script
nohup python MakeMask.py --base_path /home/ubuntu/data/Sayama/all_video_training/ \
--ROOT_DIR ../Mask_RCNN \
--WIDTH 256 \
--HEIGHT 128 \
--OUTPUT_DIR /home/ubuntu/data/Sayama/training_data_128_256 \
--TEMP_DIR /home/ubuntu/data/Sayama/tmpdir_training_128_256 &
```


## 4. Training

```script
nohup python -m depth_from_video_in_the_wild.train \
--img_height 128 \
--img_width 256 \
--data_dir /home/ubuntu/data/Sayama/training_data_128_256 \
--checkpoint_dir=/home/ubuntu/data/kitti_experiment_checkpoint_20201228 \
--imagenet_ckpt=/home/ubuntu/data/ResNet18/model.ckpt \
--train_steps=1000000 &
```

## Evaluation

## Before fine tuning

## Getting Predicted Depth

```shell
python inference_dfv.py \
    --img_height 128 \
    --img_width 256 \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_dir /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1top_png/image_02/data/ \
    --output_dir /home/ubuntu/Sayama/result_video1top_143940_128_256/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20201228/model-143940
```

## Getting Abs Rel Error

```
python AbsRelError.py \
--save_path /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1top_png/image_02/data/ \
--depth_map_dir /home/ubuntu/Sayama/result_video1top_143940_128_256/ \
--ans_int_disp_map_dir /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1middle_png/image_02/data
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.2853,     3.5981,     9.5108,     0.3669,     0.0000,     0.5333,     0.8103,     0.9268 ,   12.7202 

## After fine tuning

## Getting Predicted Depth

```shell
python inference_dfv.py \
    --img_height 128 \
    --img_width 256 \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion false \
    --input_dir /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1top_png/image_02/data/ \
    --output_dir /home/ubuntu/Sayama/result_video1top_150558_128_256/ \
    --model_ckpt /home/ubuntu/data/kitti_experiment_checkpoint_20201228/model-150558
```


## Getting Abs Rel Error

```
python AbsRelError.py \
--save_path /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1top_png/image_02/data/ \
--depth_map_dir /home/ubuntu/Sayama/result_video1top_150558_128_256/ \
--ans_int_disp_map_dir /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1middle_png/image_02/data
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.2465,     3.6083,     9.2301,     0.3344,     0.0000,     0.6564,     0.8658,     0.9395 ,   14.5235



## Getting Abs Rel Error (With no scale matching)

```
python AbsRelError_NoScaleMatching.py \
--save_path /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1top_png/image_02/data/ \
--depth_map_dir /home/ubuntu/Sayama/result_video1top_150558_128_256/ \
--ans_int_disp_map_dir /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1middle_png/image_02/data
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.6569,    14.5743,    22.5303,     1.4669,     0.0000,     0.0662,     0.1630,     0.2452 ,    1.0000 
 

## Getting Abs Rel Error (Scale Matching with acceleration sensor)

```
python AbsRelError_NoScaleMatching.py \
--save_path /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1top_png/image_02/data/ \
--depth_map_dir /home/ubuntu/Sayama/result_video1top_150558_128_256/ \
--ans_int_disp_map_dir /home/ubuntu/data/Sayama/tmpdir_128_256/2020_08_04/video1middle_png/image_02/data
```

- abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3,     scalor 
- 0.3831,     6.7716,    11.9421,     0.4545,     0.0000,     0.4083,     0.6868,     0.8542 ,   13.7007 


## Visualization

```
python Visualization.py 
```
