import sys
import os
import shutil
import glob
import subprocess
import torch
import collections
from tensorboardX import SummaryWriter
from params import par


class Logger(object):
    __instance = None

    @staticmethod
    def get_instance():
        if not Logger.__instance:
            Logger.__instance = Logger()
        return Logger.__instance

    def __init__(self):
        self.working_dir = None
        self.tensorboard = None
        self.record_file_handle = None
        self.log_training_state_latest_epoch = collections.OrderedDict()

    def initialize(self, working_dir, use_tensorboard):
        self.working_dir = working_dir

        # create directory to save the run
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.record_file_handle = open(os.path.join(self.working_dir, "record.txt"), "a")
        if use_tensorboard:
            self.tensorboard = SummaryWriter(os.path.join(self.working_dir))
        else:
            self.tensorboard = None

    def log_parameters(self):
        # Write all hyperparameters
        parameters_str = ""
        parameters_str += '\n' + '=' * 50 + '\n'
        parameters_str += '\n'.join("%s: %s" % item for item in vars(par).items())
        parameters_str += '\n' + '=' * 50
        self.print(parameters_str)

    def log_source_files(self):
        # log files
        files_to_log = []
        for filename in glob.iglob(os.path.join(par.project_dir, "**"), recursive=True):
            if "/results/" not in filename and filename.endswith(".py"):
                files_to_log.append(filename)

        Logger.log_file_content(self.working_dir, files_to_log)

        # log git commit number and status
        git_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=par.project_dir).decode(
                'ascii').strip()
        git_status_output = subprocess.check_output(['git', 'status'], cwd=par.project_dir).decode('ascii')
        git_file = open(os.path.join(self.working_dir, git_commit_hash), 'w')
        git_file.write(git_status_output)
        git_file.close()

        # log git diff
        diff_ret = subprocess.check_output(['git', '--no-pager', 'diff'], cwd=par.project_dir).decode('ascii')
        diff_file = open(os.path.join(self.working_dir, 'diff'), "w")
        diff_file.write(diff_ret)
        diff_file.close()

    def print(self, *args):
        string = " ".join([str(arg) for arg in args])
        sys.stdout.write(string)
        sys.stdout.write("\n")
        sys.stdout.flush()

        self.record_file_handle.write(string)
        self.record_file_handle.write("\n")
        self.record_file_handle.flush()

    def log_training_state(self, tag, epoch, model_state_dict, optimizer_state_dict=None):
        logger.print('Save model at ep %d, type: %s' % (epoch, tag))
        torch.save(model_state_dict, os.path.join(self.working_dir, "saved_model.%s" % tag))
        if optimizer_state_dict:
            torch.save(optimizer_state_dict, os.path.join(self.working_dir, "saved_optimizer.%s" % tag))

        self.log_training_state_latest_epoch[tag] = epoch

    def get_tensorboard(self):
        assert (self.tensorboard is not None)
        return self.tensorboard

    @staticmethod
    def make_dir_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def ensure_file_dir_exists(path):
        Logger.make_dir_if_not_exist(os.path.dirname(path))
        return path

    @staticmethod
    def log_file_content(log_path, file_paths):
        common_prefix = os.path.commonprefix(file_paths)
        for f in file_paths:
            rel_path = os.path.relpath(f, common_prefix)
            shutil.copyfile(f, Logger.ensure_file_dir_exists(os.path.join(log_path, rel_path)))


logger = Logger.get_instance()
