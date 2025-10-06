from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
import time
import os.path as op
from model.build_finetune4 import build_finetune_model

from datasets import build_dataloader
from datasets.build import build_test_loader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/20240901_092350_finetune/configs.yaml') # logs/RSTPReid/20240903_094627_finetune/configs.yaml
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda:0"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    # train_loader, _, _, num_classes = build_dataloader(args)
    model = build_finetune_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best0.pth'))
    model.to(device)
    # do_inference(model, train_loader)
    do_inference(model, test_img_loader, test_txt_loader) # 原来的