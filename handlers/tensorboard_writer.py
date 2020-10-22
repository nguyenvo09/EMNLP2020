from tensorboardX import SummaryWriter

class TensorboardWrapper():
    my_tensorboard_writer = None

    def __init__(self):
        pass

    @classmethod
    def init_log_files(cls, log_file):
        if log_file != None:
            cls.my_tensorboard_writer = SummaryWriter(log_file)

    @classmethod
    def mywriter(cls):
        assert cls.my_tensorboard_writer != None, "The LogFile is not initialized yet!"
        return cls.my_tensorboard_writer