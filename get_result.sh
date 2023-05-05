folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230505-18-28-45'
model=$folder/saved_model.valid

python exec.py gen_trajectory $model

python exec.py plot_trajectory $folder/saved_model.valid.traj