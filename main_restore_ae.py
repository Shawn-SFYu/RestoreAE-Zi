import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import yaml
import os

from pathlib import Path
from utils.optim_factory import create_optimizer

from engine import train_one_epoch, evaluate
from restore_dataset import build_dataset
from utils import nn_utils
from utils.nn_utils import create_model, get_args_parser, read_yaml_config, overwrite_config


def main(args):
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset, args.nb_classes = build_dataset(args=args)
    val_len = int(args.val_ratio * len(dataset))
    train_len = len(dataset) - val_len
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_len, val_len])

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = nn_utils.TensorboardLogger(log_dir=args.log_dir)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        # sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    print(f"Dataset {dataset_train}")

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            # sampler=sampler_val,
            batch_size=int(args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    model = create_model(args.model, args.latent_size)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    assigner = None

    optimizer = create_optimizer(
        args,
        model,
        skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None,
        filter_bias_and_bn=True,
    )


    print("Use Cosine LR scheduler")
    lr_schedule_values = nn_utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = nn_utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    criterion = torch.nn.MSELoss()
    print("criterion = %s" % str(criterion))

    nn_utils.auto_load_model(
        args=args,
        model=model,
        optimizer=optimizer
    )

    if args.eval_only:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Loss of the network on {len(dataset_val)} test images: {test_stats['loss']:.5f}%"
        )
        return

    min_loss = np.Inf

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        print(f"length dataloader {len(data_loader_train)}")
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                nn_utils.save_model(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                )
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device)
            print(
                f"Loss of the model on the {len(dataset_val)} test images: {test_stats['loss']:.4f}%"
            )
            if min_loss > test_stats["loss"]:
                min_loss = test_stats["loss"]
                if args.output_dir and args.save_ckpt:
                    nn_utils.save_model(
                        args=args,
                        model=model,
                        optimizer=optimizer,
                        epoch="best",
                    )
            print(f"Min loss: {min_loss:.2f}%")

            if log_writer is not None:
                log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(yaml.dump(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Autoencoder training and evaluation script for restoration", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    config = read_yaml_config(args.config)
    args = overwrite_config(config, args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
