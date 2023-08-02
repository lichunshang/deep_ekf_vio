# Deep EKF VIO
This is the continuation of using an EKF as part of an end-to-end learnable network (https://github.com/lichunshang/end_to_end_odometry) 

## Dependencies:
See `env.yaml` for list of dependencies

## Example Usage:
Change parameters "self.project_dir" to the appropriate directory in param.py


#### KITTI Folder Layout:
```
path_to_KITTI_dir/
    - dataset/
        - 2011_09_30/
            - 2011_09_30_drive_0034_extract
                - image_02
                - oxts
            - ...
        - 2011_10_03/
            - 2011_10_03_drive_0027_extract
                - image_02
                - oxts
            - ...
```

#### EUROC Folder Layout:
```
path_to_EUROC_dir/
    - MH_01/
        - mav0/
            - 2011_09_30_drive_0034_extract
            - cam0
            - imu0
            - state_groundtruth_estimate0
        - ...
```

#### Preprocessing:
Change parameters "DATASET_DIR" to the appropriate directory the shell scripts

`preprocess_kitti_seqs.sh` (need MATLAB with geographic lib installed to process)

`preprocess_euroc_seqs.sh`

#### Training:
Get the model weight from [here](https://drive.google.com/file/d/1I2018f6ZXKNrPig58Dc28bm8gejkVnvu/view?usp=sharing).

`python3 main.py  --gpu_id 0`

#### Evaluation:
`python3 results_directory/main.py  --gpu_id 0 --run_eval_only`
