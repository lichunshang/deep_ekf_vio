from trainer import _OnlineDatasetEvaluator, E2EVIO
from log import logger
import torch
from params import par

e2e_vio_model = E2EVIO()
resume_path = "/home/cs4li/Dev/deep_ekf_vio/results/imu_covar_tests/train_20190402-21-07-50/saved_model.valid"
save_path = "/home/cs4li/Dev/deep_ekf_vio/results/ekf_total_test2"
e2e_vio_model.load_state_dict(logger.clean_state_dict_key(torch.load(resume_path)))
e2e_vio_model = e2e_vio_model.cuda()
e2e_vio_model.eval()
e2e_vio_ta = _OnlineDatasetEvaluator(e2e_vio_model, ['K04', 'K06', 'K07', 'K10'], 50)
# e2e_vio_ta = _OnlineDatasetEvaluator(e2e_vio_model, ['K10'], 50)

par.enable_ekf = True
logger.initialize(save_path, use_tensorboard=True)
e2e_vio_ta.evaluate_abs()

par.enable_ekf = False
e2e_vio_ta.evaluate_rel()
