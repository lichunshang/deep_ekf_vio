folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/train_20230614-06-57-55'

python main.py --gpu_id=0\
                --resume_model_from $folder/saved_model.eval\
                --resume_optimizer_from $folder/saved_optimizer.checkpoint
                # --run_eval_only