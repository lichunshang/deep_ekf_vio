folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230604-22-42-58'
model=$folder/saved_model.valid

python exec.py gen_trajectory $model

python exec.py plot_trajectory $folder/saved_model.valid.traj

python exec.py calc_error  $folder/saved_model.valid.traj

python exec.py plot_error $folder/saved_model.valid.traj

python exec.py kitti_eval $folder/saved_model.valid.traj