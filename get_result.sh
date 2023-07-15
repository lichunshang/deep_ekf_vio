folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230715-01-47-04'
model=$folder/saved_model.eval

python exec.py gen_trajectory $model

python exec.py plot_trajectory $folder/saved_model.eval.traj

python exec.py calc_error  $folder/saved_model.eval.traj

python exec.py plot_error $folder/saved_model.eval.traj

python exec.py np_traj_to_kitti  $folder/saved_model.eval.traj

python exec.py kitti_eval $folder/saved_model.eval.traj