folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230801-19-51-34'
model=$folder/saved_model.eval

python exec.py gen_trajectory $model

python exec.py plot_trajectory $folder/saved_model.eval.traj

python exec.py calc_error  $folder/saved_model.eval.traj

python exec.py plot_error $folder/saved_model.eval.traj

python exec.py np_traj_to_kitti  $folder/saved_model.eval.traj

python exec.py kitti_eval $folder/saved_model.eval.traj

# python exec.py euroc_eval $folder/saved_model.eval.traj MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 V2_01 V2_02