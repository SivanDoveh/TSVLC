import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.open_clip import ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
from sentence_transformers import SentenceTransformer, util


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def data_wrapper(args,batch):
    if 'CC3M' in args.train_data and (args.vl_pos or args.vl_negs):
        images, texts, info_dict = batch
    elif 'CC3M' in args.train_data:
        images, texts = batch
        info_dict = {}
    elif 'laion' in args.train_data and (args.vl_pos or args.vl_negs):
        images, texts, neg, pos = batch
        info_dict = {"negatives": neg, "positives": pos}
    else:
        images, texts, neg, pos= batch
        info_dict = {}

    negs = info_dict.get("negatives", None) if args.vl_negs else None
    poss = info_dict.get("positives", None) if args.vl_pos else None

    return images, texts, negs, poss,None,None

def prepare_data_for_neg_loss(negs,args,device,texts):
    negs = negs.to(device=device, non_blocking=True)
    negs = negs.view(-1, negs.shape[-1])
    # clean non-negs that are zero. they r there because not every text has a negative
    pos_that_have_negs = [i for i, l in enumerate(list(negs[::args.num_negs])) if l.nonzero().any()]
    negs = [l for l in list(negs) if l.nonzero().any()]
    if len(negs) == 0:
        pos_that_have_negs=None
    else:
        texts = torch.cat((texts, torch.stack(negs)), dim=0)

    return texts,pos_that_have_negs

def prepare_data_for_pos_loss(poss, args, device, texts):
    poss = poss.to(device=device, non_blocking=True)
    poss = poss.view(-1, poss.shape[-1])
    texts = torch.cat((poss, texts), dim=0)
    return texts

def loop_save_data(data, epoch, args):
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    for i, batch in enumerate(dataloader):
        continue

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        args=args)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_neg_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    total_time=0

    for i, batch in enumerate(dataloader):
        start_t = time.time()

        if epoch < args.warmup_ep_no_bn_update:
            optimizer.zero_grad()
            optimizer.step()
            continue

        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images, texts, negs, poss,text_orig, text_pos = data_wrapper(args,batch)

        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        pos_that_have_negs=None
        if poss is not None:
            texts = prepare_data_for_pos_loss(poss, args, device, texts)

        if negs is not None:
            texts,pos_that_have_negs = prepare_data_for_neg_loss(negs,args,device,texts)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss, loss_neg = loss(image_features, text_features, logit_scale, pos_that_have_negs)
        if epoch < args.warmup_ep:
            total_loss = total_loss*0.

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))


        total_time = total_time + time.time()-start_t

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1


        if is_master(args) and args.ZS_steps_eval and (epoch >= args.warmup_ep and epoch >= args.warmup_ep_no_bn_update) and i % args.ZS_freq == 0:
            logging.info(f"eval batch {batch_count}")
            evaluate(model, data, batch_count, args, tb_writer)


        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)

            loss_neg_m.update(loss_neg.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss Neg: {loss_neg_m.val:#.5g} ({loss_neg_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "loss_neg": loss_neg_m.val,
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)


    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if args.vl_pos or args.vl_negs:
                    images, texts, _ = batch
                else:
                    images, texts = batch


                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
