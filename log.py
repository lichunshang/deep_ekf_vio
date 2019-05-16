import sys
import os
import shutil
import glob
import subprocess
import torch
import collections
import time
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

        self.print("Logging results to %s" % self.record_file_handle.name)

    def log_parameters(self):
        # Write all hyperparameters
        self.print("---------- PARAMETERS -----------")
        d = vars(par)
        s = '\n'.join("%s: %s" % (key, d[key]) for key in sorted(list(d.keys())))
        self.print(s)
        self.print("---------------------------------")
        if self.tensorboard:
            self.tensorboard.add_text("params", s)

    def log_source_files(self):
        # log files
        files_to_log = []
        curr_dir = os.path.abspath(os.path.dirname(__file__))
        for filename in glob.iglob(os.path.join(curr_dir, "**"), recursive=True):
            if "/results/" not in filename and filename.endswith(".py"):
                files_to_log.append(filename)

        Logger.log_file_content(self.working_dir, files_to_log)

        # log git commit number and status
        git_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=curr_dir).decode(
                'ascii').strip()
        git_status_output = subprocess.check_output(['git', 'status'], cwd=curr_dir).decode('ascii')
        git_file = open(os.path.join(self.working_dir, git_commit_hash), 'w')
        git_file.write(git_status_output)
        git_file.close()

        # log git diff
        diff_ret = subprocess.check_output(['git', '--no-pager', 'diff'], cwd=curr_dir).decode('ascii')
        diff_file = open(os.path.join(self.working_dir, 'diff'), "w")
        diff_file.write(diff_ret)
        diff_file.close()

    def print(self, *args, end="\n"):
        string = " ".join([str(arg) for arg in args])
        sys.stdout.write(string)
        sys.stdout.write(end)
        sys.stdout.flush()
        if self.record_file_handle:
            self.record_file_handle.write(string)
            self.record_file_handle.write(end)
            self.record_file_handle.flush()

    def log_training_state(self, tag, epoch, model_state_dict, optimizer_state_dict=None):
        start_time = time.time()
        torch.save(model_state_dict, os.path.join(self.working_dir, "saved_model.%s" % tag))
        if optimizer_state_dict:
            torch.save(optimizer_state_dict, os.path.join(self.working_dir, "saved_optimizer.%s" % tag))
        logger.print('Save model at ep %d, type: %s, time: %.2fs' % (epoch, tag, time.time() - start_time))
        self.log_training_state_latest_epoch[tag] = epoch

    @staticmethod
    def clean_state_dict_key(state_dict):
        keys = list(state_dict.keys())
        cleaned_keys = []
        new_state_dict = {}
        for key in keys:
            # if saved with dual GPU, remove the keys that start with "module."
            if key[:7] == "module.":
                new_state_dict[key[7:]] = state_dict[key]
                cleaned_keys.append(key)
            else:
                new_state_dict[key] = state_dict[key]
        logger.print("Cleaned keys: [%s]" % ", ".join(cleaned_keys))
        return new_state_dict

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
