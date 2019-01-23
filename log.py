import sys
import os
from tensorboardX import SummaryWriter
from params import par
import shutil


class Logger(object):
    __instance = None

    @staticmethod
    def get_instance():
        if not Logger.__instance:
            Logger.__instance = Logger()
        return Logger.__instance

    def __init__(self):

        # create directory to save the run
        if not os.path.exists(par.results_dir):
            os.makedirs(par.results_dir)

        self.record_file_handle = open(par.record_path, "a")

        self.tensorboard = SummaryWriter(os.path.join(par.results_dir))

    def log_parameters(self):
        # Write all hyperparameters
        parameters_str = ""
        parameters_str += '\n' + '=' * 50 + '\n'
        parameters_str += '\n'.join("%s: %s" % item for item in vars(par).items())
        parameters_str += '\n' + '=' * 50
        self.print(parameters_str)

    def print(self, string=""):
        sys.stdout.write(string)
        sys.stdout.write("\n")
        sys.stdout.flush()

        self.record_file_handle.write(string)
        self.record_file_handle.write("\n")
        self.record_file_handle.flush()

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
            shutil.copyfile(f, Logger.ensure_file_dir_exists(os.path.join(log_path, "code_log", rel_path)))


logger = Logger.get_instance()
