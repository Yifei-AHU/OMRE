import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image
from model.build_finetune import build_finetune_model
from datasets.cuhkpedes import CUHKPEDES
from datasets.rstpreid import RSTPReid


config_file  = 'logs/RSTPReid/20241106_175312_finetune/configs.yaml'
args = load_train_configs(config_file)
args.batch_size = 1024
args.training = False
device = "cuda"
test_img_loader, test_txt_loader,num_classes = build_dataloader(args)
model = build_finetune_model(args,num_classes)
checkpointer = Checkpointer(model)
checkpointer.load(f=op.join(args.output_dir, 'best0.pth'))
model.to(device)

evaluator = Evaluator(test_img_loader, test_txt_loader)

qfeats, gfeats, prob_similarity, qids, gids = evaluator._compute_embedding(model.eval())
qfeats = F.normalize(qfeats, p=2, dim=1) # text features
gfeats = F.normalize(gfeats, p=2, dim=1) # image features

similarity = qfeats @ gfeats.t()

simi = torch.tensor(0.8).to(device) * similarity + torch.tensor(0.2).to(device) * prob_similarity.to(device)

# acclerate sort with topk
_, indices = torch.topk(simi, k=10, dim=1, largest=True, sorted=True)  # q * topk

dataset = RSTPReid(root='/data/dengyifei/Data/RSTPReid/')
test_dataset = dataset.test

img_paths = test_dataset['img_paths']
captions = test_dataset['captions']
gt_img_paths = test_dataset['img_paths']

def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    image_paths = [img_paths[j] for j in indices[idx]]
    image_ids = gids[indices[idx]]
    gt_image_path = gt_img_paths[idx]
    return query_id, image_ids, query_caption, image_paths, gt_image_path

def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path, fname=None):
    print(query_id)
    print(image_ids)
    print(query_caption)
    col = len(image_paths)
    # plot ground truth image
    print(gt_img_path)

    for i in range(col):
        print(image_paths[i])
    return query_id,query_caption,gt_img_path,image_paths
    
# idx is the index of qids(A list of query ids, range from 0 - len(qids))
test_file=[]
for i in range(len(qids)):
    item={}
    query_id, image_ids, query_caption, image_paths, gt_img_path = get_one_query_caption_and_result_by_id(i, indices.cpu(), qids, gids, captions, img_paths, gt_img_paths)
    item["id"]=query_id
    item["caption"]=query_id
    item["true_img_path"]=gt_img_path
    item["predict_img_path"]=image_paths
    print("id:{},caption:{},img:{}".format(query_id,query_caption,gt_img_path))
# 读取并解析JSON文件
import json 
with open('test.json', 'w') as file:
    json.dump(test_file,file)