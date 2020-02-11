# Deep EKF VIO
This is the continuation of using an EKF as part of an end-to-end learnable network (https://github.com/lichunshang/end_to_end_odometry) 
to fuse vision and IMU. In this work
an robocentric EKF formulation is used as part of a deep CNN LSTM network to learn visual inertial odoemtry in an end to 
end manner. Good results are obtained in the KITTI dataset, however it performs poorly in the EUROC MAV dataset due to
its CNN-LSTM not generalizing well to 3D motion on small amount of data. You can find more details regarding this work in 
https://github.com/lichunshang/deep_ekf_vio/blob/master/docs/report.pdf including all the derivations and diagrams.
This repository started out as a fork to https://github.com/ChiWeiHsiao/DeepVO-pytorch, however it has been changed
extensively for our project.

Note code is still in the process of being cleaned up...

## Dependencies:
See `docker/cuda10.1/Dockerfile` for list of dependencies

## Example Usage:
Change parameters "/home/cs4li/Dev/deep_ekf_vio" to the appropriate directory in param.py
Help:
`python main.py -h`

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
Change parameters "/home/cs4li/Dev/deep_ekf_vio" to the appropriate directory the shell scripts

`preprocess_kitti_seqs.sh` (need MATLAB with geographic lib installed)

`preprocess_euroc_seqs.sh`

#### Training:
Get the pretrain flownet weights from [here](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5).

`python3 main.py --description <experiment_desciprtion> --gpu_id 0`

#### Evaluation:
`python3 results_directory/main.py  --gpu_id 0 --run_eval_only`


## System Architecture:
![Alt text](docs/e2evio_system.png)
