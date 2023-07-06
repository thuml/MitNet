"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
import sys
import time
from typing import Optional, List


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TextLogger(object):
    """Writes stream output to external text file.

    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


class CompleteLogger:
    """
    A useful logger that

    - writes outputs to files and displays them on the console at the same time.
    - manages the directory of checkpoints and debugging images.

    Args:
        root (str): the root directory of logger
        phase (str): the phase of training.

    """

    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.visualize_directory = os.path.join(self.root, "visualize")
        self.checkpoint_directory = os.path.join(self.root, "checkpoints")
        self.epoch = 0

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.visualize_directory, exist_ok=True)
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        # redirect std out
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        log_filename = os.path.join(self.root, "{}-{}.txt".format(phase, now))
        if os.path.exists(log_filename):
            os.remove(log_filename)
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger
        if phase != 'train':
            self.set_epoch(phase)

    def set_epoch(self, epoch):
        """Set the epoch number. Please use it during training."""
        os.makedirs(os.path.join(self.visualize_directory, str(epoch)), exist_ok=True)
        self.epoch = epoch

    def _get_phase_or_epoch(self):
        if self.phase == 'train':
            return str(self.epoch)
        else:
            return self.phase

    def get_image_path(self, filename: str):
        """
        Get the full image path for a specific filename
        """
        return os.path.join(self.visualize_directory, self._get_phase_or_epoch(), filename)

    def get_checkpoint_path(self, name=None):
        """
        Get the full checkpoint path.

        Args:
            name (optional): the filename (without file extension) to save checkpoint.
                If None, when the phase is ``train``, checkpoint will be saved to ``{epoch}.pth``.
                Otherwise, will be saved to ``{phase}.pth``.

        """
        if name is None:
            name = self._get_phase_or_epoch()
        name = str(name)
        return os.path.join(self.checkpoint_directory, name + ".pth")

    def close(self):
        self.logger.close()
