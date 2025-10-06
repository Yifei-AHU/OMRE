from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from model import objectives
import time

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
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        q_sample_feats, g_sample_feats = [], []
        
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat, sample_text_feats = model.new_encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat.data.cpu())
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
            g_sample_feats.append(sample_img_feats.data.cpu())
        
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        g_sample_feats = torch.cat(g_sample_feats, 0) # [b2,num,512]    

        _, t2i_prob_simi = objectives.cal_prob_simi(gfeats, qfeats, g_sample_feats, q_sample_feats)
        
        return qfeats.cuda(), gfeats.cuda(), t2i_prob_simi, qids, gids
    
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, prob_similarity, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

        simi = 0.9 * similarity.cpu() + 0.1 * prob_similarity.cpu()
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=simi, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        
        table1 = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])

        table1.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        table1.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table1.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table1))
        
        return t2i_cmc[0]
