# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import builtins
import math
import os
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
# import wandb

from termcolor import colored
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger


torch.set_printoptions(precision=4, sci_mode=False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
color_fmt = (
    colored("[%(asctime)s %(name)s]", "green")
    + colored("(%(filename)s %(lineno)d)", "yellow")
    + ": %(levelname)s %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(console_handler)


best_acc = 0


def save_checkpoint(
    state, is_best, checkpoint, filename="checkpoint.pth.tar", epoch_p=1
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def compute_adjustment_by_py(py, tro, args):
    adjustments = torch.log(py**tro + 1e-12)
    adjustments = adjustments.to(args.device)
    return adjustments


def main():
    parser = argparse.ArgumentParser(description="PyTorch FixMatch Training")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=[
            "cifar10",
            "cifar100",
            "stl10",
            "svhn",
            "smallimagenet",
            "smallimagenet_1k",
        ],
        help="dataset name",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/data/",
        type=str,
    )
    parser.add_argument(
        "--num-labeled", type=int, default=4000, help="number of labeled data"
    )
    parser.add_argument(
        "--arch",
        default="wideresnet",
        type=str,
        choices=["wideresnet", "resnext", "resnet"],
    )
    parser.add_argument(
        "--total-steps", default=250000, type=int, help="number of total steps to run"
    )
    parser.add_argument(
        "--eval-step", default=500, type=int, help="number of eval steps to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="train batchsize")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.03,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--warmup", default=0, type=float, help="warmup epochs (unlabeled data based)"
    )
    parser.add_argument("--wdecay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--nesterov", action="store_true", default=True, help="use nesterov momentum"
    )
    parser.add_argument(
        "--use-ema", action="store_true", default=True, help="use EMA model"
    )
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument(
        "--mu", default=1, type=int, help="coefficient of unlabeled batch size"
    )
    parser.add_argument("--T", default=1, type=float, help="pseudo label temperature")
    parser.add_argument(
        "--threshold", default=0.95, type=float, help="pseudo label threshold"
    )
    parser.add_argument(
        "--out", default="result", help="directory to output the result"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    parser.add_argument(
        "--no-progress", action="store_true", help="don't use progress bar"
    )

    parser.add_argument(
        "--num-max", default=500, type=int, help="the max number of the labelled data"
    )
    parser.add_argument(
        "--num-max-u",
        default=4000,
        type=int,
        help="the max number of the unlabeled data",
    )
    parser.add_argument(
        "--imb-ratio-label",
        default=1,
        type=int,
        help="the imbalanced ratio of the labelled data",
    )
    parser.add_argument(
        "--imb-ratio-unlabel",
        default=1,
        type=int,
        help="the imbalanced ratio of the unlabeled data",
    )
    parser.add_argument(
        "--flag-reverse-LT",
        default=0,
        type=int,
        help="whether to reverse the distribution of the unlabeled data",
    )
    parser.add_argument("--ema-mu", default=0.99, type=float, help="mu when ema")

    parser.add_argument(
        "--tau", default=1, type=float, help="tau for head1 consistency"
    )

    parser.add_argument(
        "--ema-u",
        default=0.9,
        type=float,
        help="ema ratio for estimating distribution of the unlabeled data",
    )
    parser.add_argument(
        "--est-epoch",
        default=5,
        type=int,
        help="the start step to estimate the distribution",
    )
    parser.add_argument(
        "--img-size", default=32, type=int, help="image size for small imagenet"
    )

    parser.add_argument(
        "--version",
        type=str,
    )
    parser.add_argument(
        "--run",
        type=int,
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
    )

    args = parser.parse_args()
    global best_acc

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.local_rank = args.gpu

    # if args.rank == 0:
    #    wandb.login(key="")
    #    wandb.init(
    #        # set the wandb project where this run will be logged
    #        project="",
    #        name="",
    #        mode=args.wandb_mode,
    #        # track hyperparameters and run metadata
    #        config=args,
    #    )

    if args.rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

        def info_pass(*args, **kwargs):
            pass

        logger.info = info_pass

    args.distributed = True
    args.multiprocessing_distributed = True

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.world_size = torch.distributed.get_world_size()
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # create file handlers
    os.makedirs(args.out, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.out, "log.txt"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    def create_model(args):
        if args.arch == "wideresnet":
            import models.wideresnet as models

            model = models.build_wideresnet(
                depth=args.model_depth,
                widen_factor=args.model_width,
                dropout=0,
                num_classes=args.num_classes,
            )
        elif args.arch == "resnext":
            import models.resnext as models

            model = models.build_resnext(
                cardinality=args.model_cardinality,
                depth=args.model_depth,
                width=args.model_width,
                num_classes=args.num_classes,
            )
        elif args.arch == "resnet":
            import models.resnet as models

            model = models.ResNet50(
                num_classes=args.num_classes, rotation=True, classifier_bias=True
            )

        logger.info(
            "Total params: {:.2f}M".format(
                sum(p.numel() for p in model.parameters()) / 1e6
            )
        )
        return model

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}",
    )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank == 0:
        os.makedirs(args.out, exist_ok=True)

    if args.dataset == "cifar10":
        args.num_classes = 10
        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == "cifar100":
        args.num_classes = 100
        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == "stl10":
        args.num_classes = 10
        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == "svhn":
        args.num_classes = 10
        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == "smallimagenet":
        args.num_classes = 127
        args.arch = "resnet"
    elif args.dataset == "smallimagenet_1k":
        args.num_classes = 1000
        args.arch = "resnet"

    (
        labeled_dataset,
        unlabeled_dataset,
        test_dataset,
        cls_num_list_l,
        cls_num_list_u,
        cls_num_list_t,
    ) = DATASET_GETTERS[args.dataset](args, args.data_dir)

    args.cls_num_list_l = torch.Tensor(cls_num_list_l).float().cuda()
    args.cls_num_list_u = torch.Tensor(cls_num_list_u).float().cuda()
    args.cls_num_list_t = torch.Tensor(cls_num_list_t).float().cuda()
    args.py_l = args.cls_num_list_l / args.cls_num_list_l.sum()
    args.py_u = args.cls_num_list_u / args.cls_num_list_u.sum()
    args.py_t = args.cls_num_list_t / args.cls_num_list_t.sum()

    if args.local_rank in [-1, 0]:
        logger.info(f"cls_num_list_l: {args.cls_num_list_l}")
        logger.info(f"cls_num_list_u: {args.cls_num_list_u}")
        logger.info(f"cls_num_list_t: {args.cls_num_list_t}")

        logger.info(f"py_l: {args.py_l}")
        logger.info(f"py_u: {args.py_u}")
        logger.info(f"py_t: {args.py_t}")

    labeled_sampler = DistributedSampler(labeled_dataset)
    unlabeled_sampler = DistributedSampler(unlabeled_dataset)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        sampler=labeled_sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        sampler=unlabeled_sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.py_con = args.py_l
    args.py_uni = torch.ones(args.num_classes) / args.num_classes
    args.py_uni = args.py_uni.to(args.device)
    args.py_rev = torch.flip(args.py_con, dims=[0])

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = "FixMatch-" + args.dataset
    args.logger = Logger(os.path.join(args.out, "acc_log.txt"), title=title)

    args.logger.set_names(
        [
            "Top1 acc",
            "Top5 acc",
            "Best Top1 acc",
        ]
    )

    model = create_model(args)

    torch.distributed.barrier()

    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wdecay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.SGD(
        grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov
    )

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup * args.eval_step, args.total_steps
    )

    if args.use_ema:
        from models.ema import ModelEMA

        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    torch.cuda.set_device(args.gpu)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(
        args,
        labeled_trainloader,
        unlabeled_trainloader,
        test_loader,
        model,
        optimizer,
        ema_model,
        scheduler,
    )
    args.logger.close()


def train(
    args,
    labeled_trainloader,
    unlabeled_trainloader,
    test_loader,
    model,
    optimizer,
    ema_model,
    scheduler,
):
    global best_acc
    test_accs = []
    avg_time = []
    end = time.time()

    labeled_epoch = 0
    unlabeled_epoch = 0
    labeled_trainloader.sampler.set_epoch(labeled_epoch)
    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    l_len = len(labeled_trainloader)
    u_len = len(unlabeled_trainloader)

    args.scale_ratio = u_len / l_len
    logger.info(f"scale ratio: {args.scale_ratio}")

    logger.info(f"len of labeled_trainloader: {len(labeled_trainloader)}")
    logger.info(f"len of unlabeled_trainloader: {len(unlabeled_trainloader)}")

    KL_div = nn.KLDivLoss(reduction="sum")

    model.train()
    args.adapt_dis = args.py_con
    args.estimated_dis = args.py_uni
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"current epoch: {epoch+1}")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        args.adjustment_u = compute_adjustment_by_py(args.estimated_dis, args.tau, args)
        args.adjustment = compute_adjustment_by_py(args.adapt_dis, args.tau, args)

        dis_unlabeled = torch.zeros(args.num_classes).to(args.device)
        count_labeled_dataset = torch.zeros(args.num_classes).to(args.device)

        for batch_idx in tqdm(range(args.eval_step), total=args.eval_step):
            try:
                (inputs_l_w, inputs_l_s), l_real = labeled_iter.next()
            except StopIteration:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)

                labeled_iter = iter(labeled_trainloader)
                (inputs_l_w, inputs_l_s), l_real = labeled_iter.next()
            except Exception as e:
                raise e

            try:
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = unlabeled_iter.next()
            except StopIteration:
                unlabeled_epoch += 1
                unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = unlabeled_iter.next()
            except Exception as e:
                raise e

            if args.dataset == "stl10":
                u_real = u_real + 1

            u_real = u_real.cuda()

            data_time.update(time.time() - end)
            batch_size = inputs_l_w.shape[0]

            inputs = torch.cat(
                (inputs_l_w, inputs_l_s, inputs_u_w, inputs_u_s, inputs_u_s1)
            ).cuda()

            targets_l = l_real.to(args.device)

            logits = model(inputs)

            logits_l_w, logits_l_s = (
                logits[:batch_size],
                logits[batch_size : batch_size * 2],
            )
            logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size * 2 :].chunk(3)

            pseudo_label = torch.softmax(
                logits_u_w.detach() + args.adjustment_u, dim=-1
            )

            max_probs, _ = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold)

            count_labeled_dataset += torch.bincount(
                targets_l, minlength=args.num_classes
            )

            dis_unlabeled += torch.sum(pseudo_label[mask], dim=0)

            mask = mask.float()
            mask_twice = torch.cat([mask, mask], dim=0).cuda()

            logits_u_s_twice = torch.cat([logits_u_s, logits_u_s1], dim=0).cuda()
            targets_u_twice = torch.cat([pseudo_label, pseudo_label], dim=0).cuda()

            logits_l = torch.cat([logits_l_w, logits_l_s], dim=0).cuda()
            targets_l = torch.cat([targets_l, targets_l], dim=0).cuda()

            Lx = (
                F.cross_entropy(logits_l + args.adjustment, targets_l, reduction="mean")
                / args.scale_ratio
            )

            Lu = (
                F.cross_entropy(
                    logits_u_s_twice + args.adjustment,
                    targets_u_twice,
                    reduction="none",
                )
                * mask_twice
            ).mean()

            loss = Lx + Lu

            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

        eps = 1e-12

        estimated_dis = dis_unlabeled / (dis_unlabeled.sum() + 1)

        args.estimated_dis = args.estimated_dis * args.ema_u + (estimated_dis) * (
            1 - args.ema_u
        )

        KL_estim = 0.5 * KL_div((estimated_dis + eps).log(), args.py_u) + 0.5 * KL_div(
            args.py_u.log(), estimated_dis
        )
        logger.info(f"KL_estim: {KL_estim}")

        count_forward = count_labeled_dataset + dis_unlabeled

        args.adapt_dis = args.adapt_dis * args.ema_u + (
            count_forward / count_forward.sum()
        ) * (1 - args.ema_u)

        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank == 0:
            test_loss, test_acc, test_top5_acc = test(
                args, test_loader, test_model, epoch
            )

            # wandb.log(
            #    {
            #        "KL_estim": KL_estim,
            #        "train_loss": losses.avg,
            #        "train_loss_x": losses_x.avg,
            #        "train_loss_u": losses_u.avg,
            #        "test_acc": test_acc,
            #        "test_loss": test_loss,
            #    },
            #    commit=True,
            #    step=epoch,
            # )

            is_best = test_acc > best_acc

            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = (
                    ema_model.ema.module
                    if hasattr(ema_model.ema, "module")
                    else ema_model.ema
                )

            if (epoch + 1) % 10 == 0 or (is_best and epoch // args.epochs > 0.9):
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                        "ema_state_dict": ema_to_save.state_dict()
                        if args.use_ema
                        else None,
                        "acc": test_acc,
                        "best_acc": best_acc,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    is_best,
                    args.out,
                    epoch_p=epoch + 1,
                )

            test_accs.append(test_acc)
            logger.info("Best top-1 acc: {:.2f}".format(best_acc))
            logger.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))

            args.logger.append(
                [
                    test_acc,
                    test_top5_acc,
                    best_acc,
                ]
            )


def shot_acc(preds, labels, many_shot_thr=800, low_shot_thr=10, acc_per_cls=False):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("Type ({}) of preds not supported".format(type(preds)))

    test_class_count = []
    class_correct = []
    for label in np.unique(labels):
        test_class_count.append(len(labels[labels == label]))
        class_correct.append((preds[labels == label] == labels[labels == label]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(test_class_count)):
        if test_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif test_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            total_logits = torch.cat((total_logits, outputs), dim=0)
            total_labels = torch.cat((total_labels, targets), dim=0)

            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

        if args.num_classes != 1000:
            _, preds = F.softmax(total_logits, dim=1).max(dim=1)
            many_acc_top1, median_acc_top1, low_acc_top1, class_accs = shot_acc(
                preds, total_labels, acc_per_cls=True
            )
            logger.info(f"many_acc_top1: {many_acc_top1}")
            logger.info(f"median_acc_top1: {median_acc_top1}")
            logger.info(f"low_acc_top1: {low_acc_top1}")
            logger.info(f"class_accs: {class_accs}")

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
