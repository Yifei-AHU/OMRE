import collections
import logging
import random
import time
import torch
from datasets.build import build_filter_loader
from model import objectives
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F

def do_train(start_epoch, args, model, train_loader, evaluator0, optimizer,
             scheduler, checkpointer, trainset):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("OMRE.train")
    if get_rank() == 0:
        logger.info("Validation before training - Epoch: {}".format(-1))
        top1 = evaluator0.eval(model.eval())
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "boma_loss": AverageMeter(),
        "reg_loss": AverageMeter(),
        "hnm_loss": AverageMeter(),
        "crsr_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1_0 = 0.0
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()


        for n_iter, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if k != "img_name" else v) for k, v in batch.items()}
           
            ret = model(batch)
            ret = {key: values.mean() for key, values in ret.items()}
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            
            meters['loss'].update(total_loss.item(), batch_size)
            meters['boma_loss'].update(ret.get('boma_loss', 0), batch_size)
            meters['reg_loss'].update(ret.get('reg_loss', 0), batch_size)
            meters['hnm_loss'].update(ret.get('hnm_loss', 0), batch_size)
            meters['crsr_loss'].update(ret.get('crsr_loss', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / 60
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[min] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            logger.info(f"best R1: {best_top1_0}")
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1_0 = evaluator0.eval(model.module.eval())
                else:
                    top1_0 = evaluator0.eval(model.eval())
                torch.cuda.empty_cache()
                if best_top1_0 <= top1_0:
                    best_top1_0 = top1_0
                    arguments["epoch"] = epoch
                    checkpointer.save("best0", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1_0} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
