from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from model import objectives

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity.data.cpu(), dim=1, descending=True)
        indices = indices.to(similarity.device)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, test_loader):
        self.test_loader = test_loader # gallery
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        q_sample_feats, g_sample_feats = [], []

        all_caption_tokens = []

        for i, (pid, image_id, img_name, img, caption_tokens) in enumerate(self.test_loader):
            all_caption_tokens.append(caption_tokens)
            caption = caption_tokens.to(device)
            img = img.to(device)
            with torch.no_grad():
                img_feat, sample_img_feats = model.new_encode_image(img)
                text_feat, sample_text_feats = model.new_encode_text(caption)

            name_count = {}
            new_img_name = []
            for name in img_name:
                if name in name_count:
                    name_count[name] += 1
                else:
                    # 如果名称第一次出现，计数初始化为1
                    name_count[name] = 1
                # 修改名称，添加后缀
                new_name = f"{name}_{name_count[name]}"
                new_img_name.append(new_name)

            new_img_name = tuple(new_img_name)
            text_feat = text_feat.float()
            img_feat = img_feat.float()
            model.cross_former(text_feat.unsqueeze(1),img_feat,img_feat,caption,img, new_img_name)
            

        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat, sample_text_feats = model.new_encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat.data.cpu())
            # qfeats.append(text_feat.data)
            q_sample_feats.append(sample_text_feats.data.cpu())
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        q_sample_feats = torch.cat(q_sample_feats, dim=0) # [b1,num,512]

        # image
        ori_img = []
        img_names = []
        for pid, img, img_name in self.img_loader:
            img = img.to(device)
            ori_img.append(img) # 存储每个图象
            img_names.append(img_name)
            with torch.no_grad():
                img_feat, sample_img_feats = model.new_encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat.data.cpu())
            # gfeats.append(img_feat.data)
            g_sample_feats.append(sample_img_feats.data.cpu())
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        g_sample_feats = torch.cat(g_sample_feats, 0) # [b2,num,512]

        # 在这里进行注意力图的绘制 单个query 多个图像 做CA
        import cv2
        qfeats = qfeats[0].unsqueeze(0) # 预设一个Query
        caption_ids = caption[0].unsqueeze(0)
        for i in range(gfeats.size(0)):
            gfeats_i = gfeats[i].unsqueeze(0) # 获取第i个图象的特征
            ori_img_i = ori_img[i] # 获取第i个原始图像
            img_name = img_names[i]
            model.cross_former(qfeats.to(caption_ids.device), gfeats_i.to(caption_ids.device), gfeats_i.to(caption_ids.device), caption_ids, ori_img_i, img_name)

        _, t2i_prob_simi = objectives.cal_simi_v5(gfeats, qfeats, g_sample_feats, q_sample_feats)

        # return qfeats.cuda(), gfeats.cuda(), q_sample_feats.cuda(), g_sample_feats.cuda(), qids, gids
        return qfeats.cuda(), gfeats.cuda(), t2i_prob_simi, qids, gids
        # return qfeats.cuda(), gfeats.cuda(), qids, gids
    
    def get_one_query_caption_and_result_by_id(self, idx, indices, qids, gids, captions, img_paths, gt_img_paths):
        query_caption = captions[idx]
        query_id = qids[idx]
        image_paths = [img_paths[j] for j in indices[idx]]
        image_ids = gids[indices[idx]]
        # gt_image_path = gt_img_paths[idx]
        return query_id, image_ids, query_caption, image_paths
    
    
    
    def eval(self, model, i2t_metric=False):

        # qfeats, gfeats, q_sample_feats, g_sample_feats, qids, gids = self._compute_embedding(model)

        # import time
        # start_time = time.time() 

        qfeats, gfeats, prob_similarity, qids, gids = self._compute_embedding(model)
        # qfeats, gfeats, qids, gids = self._compute_embedding(model) # IRRA

        # end_time = time.time() 
        # inference_time = (end_time - start_time) / qfeats.size(0)


        # 存放测试特征，用于进行t-sne
        # tensor_dict = {
        #     'qfeats': qfeats,
        #     'gfeats': gfeats,
        #     'qids': qids,
        #     'gids': gids
        # }
        # torch.save(tensor_dict, 'rstp_for_tsne_demo.pth')
        # 到这里为止

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        similarity = qfeats @ gfeats.t()

        a = torch.tensor([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        best_R1 = 0.0
        for i in range(len(a)):
            simi = a[i].to('cuda') * similarity + (1 - a[i]).to('cuda') * prob_similarity.to('cuda')
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=simi, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            print('weight = ', a[i])
            print("R1:", t2i_cmc[0], "R5:", t2i_cmc[4], "R10:", t2i_cmc[9], "mAP:", t2i_mAP, "mINP:", t2i_mINP)
            if t2i_cmc[0] > best_R1:
                best_R1 = t2i_cmc[0]
                best_R5 = t2i_cmc[4]
                best_R10 = t2i_cmc[9]
                best_mAP = t2i_mAP
                best_mINP = t2i_mINP

        # 用于可视化检索的图像代码
        # from datasets.rstpreid import RSTPReid
        # import json 

        # simi = torch.tensor(0.9).to('cuda') * similarity + torch.tensor(0.1).to('cuda') * prob_similarity.to('cuda')

        # t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=simi, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        # t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        # print("R1:", t2i_cmc[0], "R5:", t2i_cmc[4], "R10:", t2i_cmc[9], "mAP:", t2i_mAP, "mINP:", t2i_mINP)
        
        # _, indices = torch.topk(simi, k=10, dim=1, largest=True, sorted=True)  # q * topk

        # dataset = RSTPReid(root='/data/dengyifei/Data/')
        # test_dataset = dataset.test

        # img_paths = test_dataset['img_paths']
        # captions = test_dataset['captions']
        # gt_img_paths = test_dataset['img_paths']

        # test_file=[]
        # for i in range(len(qids)):
        #     item={}
        #     query_id, image_ids, query_caption, image_paths = self.get_one_query_caption_and_result_by_id(i, indices.cpu(), qids, gids, captions, img_paths, gt_img_paths)
        #     item["id"]=query_id.item()
        #     item["caption"]=query_id.item()
        #     item["predict_img_path"]=image_paths
        #     test_file.append(item)
        #     print("id:{},caption:{}".format(query_id,query_caption))
        
        # with open('test.json', 'w') as file:
        #     json.dump(test_file,file)

        #### 到这为止
        
        table1 = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table1.add_row(['probt2i', best_R1, best_R5, best_R10, best_mAP, best_mINP])

        table1.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table1))
        
        return best_R1
