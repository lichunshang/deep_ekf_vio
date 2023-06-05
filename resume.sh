folder='/mnt/data/teamAI/duy/deep_ekf_vio/results/noEKF_failed_01'

python main.py --gpu_id=0\
                --resume_model_from $folder/saved_model.valid\
                --resume_optimizer_from $folder/saved_optimizer.checkpoint\
                # --run_eval_only