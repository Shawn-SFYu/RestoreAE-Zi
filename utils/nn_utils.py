import os
import math
import time
from collections import defaultdict, deque
import datetime
import argparse
import yaml
import numpy as np

from pathlib import Path

import torch
import torch.distributed as dist

from tensorboardX import SummaryWriter

from models.mobilenet_ae import MobileNetAE
from models.efficient_ae import EfficientNetAE
from models.convnext_ae import ConvNextAE
from models.vit_ae import ViT_AE


def read_yaml_config(yaml_input):
    with open(yaml_input, "r") as config_file:
        config_dict = yaml.safe_load(config_file)
    config = argparse.Namespace(**config_dict)
    return config

def overwrite_config(config, args):
    for key in vars(args):
        value = getattr(args, key)
        if (value is not None) or (not hasattr(config, key)):
            setattr(config, key, value)
    return config

def write_yaml_config(config):
    pass

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Autoencoder training and evaluation script for restoration",
        add_help=False,
    )

    parser.add_argument("-c", "--config", required=True, help="path to yaml config")
    parser.add_argument("--batch_size", default=None, type=int, help="Per GPU batch size")
    parser.add_argument("--epochs", default=None, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--latent_size", default=None, type=int, help="Latent space size for CAE"
    )

    parser.add_argument(
        "--update_freq", default=None, type=int, help="gradient accumulation steps"
    )

    # Optimization parameters
    parser.add_argument(
        "--opt",
        default=None,
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=None,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=None, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="Final value of the weight decay.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=4e-3,
        metavar="LR",
        help="learning rate (default: 4e-3), with total batch size 4096",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default=None, type=str, help="dataset path"
    )
    parser.add_argument(
        "--eval_data_path", default=None, type=str, help="dataset path for evaluation"
    )
    parser.add_argument(
        "--nb_classes",
        default=3751,
        type=int,
        help="number of the classification types",
    )
    # need to check
    parser.add_argument(
        "--data_set",
        default="image_folder",
        choices=["CIFAR", "IMNET", "image_folder"],
        type=str,
        help="ImageNet dataset path",
    )
    # need to check
    parser.add_argument(
        "--output_dir", default="./checkpoint", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--log_dir", default="./log", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=None, type=int)

    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", type=bool, default=True)
    parser.add_argument("--save_ckpt", type=bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_num", default=3, type=int)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--eval_only", type=bool, default=False, help="Perform evaluation only"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Perform evaluation only"
    )
    parser.add_argument(
        "--disable_eval",
        type=bool,
        default=True,
        help="Disabling evaluation during training",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        type=bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=4, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=bool, default=False)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(f"print self {str(self)}")
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step
            )

    def flush(self):
        self.writer.flush()


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(
    args, epoch, model, optimizer
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }

        torch.save(to_save, checkpoint_path)

    to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
    old_ckpt = output_dir / ("checkpoint-%s.pth" % to_del)
    if os.path.exists(old_ckpt):
        os.remove(old_ckpt)


def auto_load_model(
    args, model, optimizer
):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob

        all_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*.pth"))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split("-")[-1].split(".")[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, "checkpoint-%d.pth" % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if not isinstance(
                checkpoint["epoch"], str
            ):  # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint["epoch"] + 1
            else:
                assert args.eval, "Does not support resuming with checkpoint-best"
            print("With optim & sched!")


def create_model(model, latent_dimension):
    if model == 'effinet':
        return EfficientNetAE(latent_dimension)
    elif model == 'mobilenet':
        return MobileNetAE(latent_dimension)
    elif model == 'convnext':
        return ConvNextAE(latent_dimension)
    elif model == 'vit':
        return ViT_AE(in_channels=2, out_channel=1)
    else:
        raise ValueError('unrecognized model name')
