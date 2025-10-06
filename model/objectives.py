import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)

def cal_prob_simi(img_mu, text_mu, image_fetures, text_fetures):  
    img_batch_size = image_fetures.size(0)
    text_batch_size = text_fetures.size(0)
    sample_num = image_fetures.shape[1] # image_fetures-- [b,num,512]
    
    img_mu_norm = img_mu / img_mu.norm(dim=-1, keepdim=True) # [b,512]
    text_mu_norm = text_mu / text_mu.norm(dim=-1, keepdim=True) # [b,512]

    image_fetures_norm = image_fetures / image_fetures.norm(dim=-1, keepdim=True)
    text_fetures_norm = text_fetures / text_fetures.norm(dim=-1, keepdim=True)

    image_fetures_norm = image_fetures_norm.view(-1, image_fetures_norm.shape[-1]) # [b*n, 512]
    text_fetures_norm = text_fetures_norm.view(-1, text_fetures_norm.shape[-1]) # [b*n, 512]

    i2t_m2s_simi = img_mu_norm @ text_fetures_norm.t() # [b1,b2*n]
    i2t_s2m_simi = image_fetures_norm @ text_mu_norm.t() # [b1*n,b2]
    i2t_m2m_simi = img_mu_norm @ text_mu.t() # [b1, b2]

    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b2,b1*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b2*n,b1]
    t2i_m2m_simi = text_mu @ img_mu_norm.t() # [b2, b1]

    i2t_simi = i2t_m2s_simi.reshape(img_batch_size,text_batch_size,sample_num).sum(dim=-1) + i2t_s2m_simi.reshape(img_batch_size,sample_num,text_batch_size).sum(1)
    t2i_simi = t2i_m2s_simi.reshape(text_batch_size,img_batch_size,sample_num).sum(dim=-1) + t2i_s2m_simi.reshape(text_batch_size,sample_num,img_batch_size).sum(1)
    return i2t_simi, t2i_simi

def compute_boma_loss(img_mu, text_mu, image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):  
    batch_size = image_fetures.size(0)
    sample_num = image_fetures.shape[1] # image_fetures-- [b,num,512]

    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
    
    img_mu_norm = img_mu / img_mu.norm(dim=-1, keepdim=True) # [b,512]
    text_mu_norm = text_mu / text_mu.norm(dim=-1, keepdim=True) # [b,512]

    image_fetures_norm = image_fetures / image_fetures.norm(dim=-1, keepdim=True)
    text_fetures_norm = text_fetures / text_fetures.norm(dim=-1, keepdim=True)

    image_fetures_norm = image_fetures_norm.view(-1, image_fetures_norm.shape[-1]) # [b*n, 512]
    text_fetures_norm = text_fetures_norm.view(-1, text_fetures_norm.shape[-1]) # [b*n, 512]

    # Text Mean - Img Sample
    i2t_m2s_simi = img_mu_norm @ text_fetures_norm.t() # [b,b*n]
    i2t_s2m_simi = image_fetures_norm @ text_mu_norm.t() # [b*n,b]
    i2t_m2m_simi = img_mu_norm @ text_mu.t() # [b, b]

    # Img Mean - Text Sample
    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b,b*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b*n,b]
    t2i_m2m_simi = text_mu @ img_mu_norm.t()
    
    # Fuse Simi
    i2t_simi = (i2t_m2s_simi.reshape(batch_size,batch_size,sample_num).sum(dim=-1) / sample_num + i2t_s2m_simi.reshape(batch_size,sample_num,batch_size).sum(1) / sample_num) / 2
    t2i_simi = (t2i_m2s_simi.reshape(batch_size,batch_size,sample_num).sum(dim=-1) / sample_num + t2i_s2m_simi.reshape(batch_size,sample_num,batch_size).sum(1) / sample_num) / 2

    t2i_simi = logit_scale * t2i_simi
    i2t_simi = logit_scale * i2t_simi
    
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(i2t_simi, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(i2t_simi, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(t2i_simi, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_simi, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def rowwise_softmax(x):
    result = torch.zeros_like(x)
    for i in range(x.size(0)):
        non_zero_elements = x[i][x[i] != 0]
        softmax_values = F.softmax(non_zero_elements, dim=0)
        result[i][x[i] != 0] = softmax_values
    return result

def triplet_ranking_loss(A, B, margin=0.2):
    loss = (margin + A - B).clamp(min=0.0, max=margin)
    num_triplets = torch.nonzero(loss).shape[0]
    if num_triplets == 0:
        return loss.mean()
    else:
        return loss.sum() / num_triplets

def compute_hnm_loss(img_mu, text_mu, image_fetures, text_fetures, pid, margin=0.2): # RSTP-0.3 ICFG-0.4 CUHK-1.0
    batch_size = image_fetures.size(0)
    sample_num = image_fetures.shape[1] # image_fetures-- [b,num,512]
    
    img_mu_norm = img_mu / img_mu.norm(dim=-1, keepdim=True) # [b,512]
    text_mu_norm = text_mu / text_mu.norm(dim=-1, keepdim=True) # [b,512]

    image_fetures_norm = image_fetures / image_fetures.norm(dim=-1, keepdim=True)
    text_fetures_norm = text_fetures / text_fetures.norm(dim=-1, keepdim=True)

    image_fetures_norm = image_fetures_norm.view(-1, image_fetures_norm.shape[-1]) # [b*n, 512]
    text_fetures_norm = text_fetures_norm.view(-1, text_fetures_norm.shape[-1]) # [b*n, 512]

    # 计算文本均值->图像采样点的相似度 文本采样点到图像均值的相似度
    i2t_m2s_simi = img_mu_norm @ text_fetures_norm.t() # [b,b*n]
    i2t_s2m_simi = image_fetures_norm @ text_mu_norm.t() # [b*n,b]

    # 计算图像均值->文本采样点的相似度 图像采样点到文本均值的相似度
    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b,b*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b*n,b]

    i2t_simi = (i2t_m2s_simi.reshape(batch_size,batch_size,sample_num).sum(dim=-1) / sample_num + i2t_s2m_simi.reshape(batch_size,sample_num,batch_size).sum(1) / sample_num) / 2
    t2i_simi = (t2i_m2s_simi.reshape(batch_size,batch_size,sample_num).sum(dim=-1) / sample_num + t2i_s2m_simi.reshape(batch_size,sample_num,batch_size).sum(1) / sample_num) / 2

    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    
    mask = (torch.eye(batch_size) > .5).cuda()
    neg_mask = ~mask

    i2t_pos = i2t_simi[mask].view(i2t_simi.shape[0], -1, 1).permute(1, 0, 2) # [1,b,1]
    i2t_neg = i2t_simi[neg_mask].view(1, i2t_simi.shape[0], -1)
    i2t_loss = triplet_ranking_loss(i2t_neg, i2t_pos, margin) # [1,b,b-1]

    t2i_pos = t2i_simi[mask.t()].view(t2i_simi.shape[0], -1, 1).permute(1, 0, 2)
    t2i_neg = t2i_simi[neg_mask.t()].view(1, t2i_simi.shape[0], -1)
    t2i_loss = triplet_ranking_loss(t2i_neg, t2i_pos, margin)

    return i2t_loss + t2i_loss

def kl_divergence(mu, sigma): 
    # return -0.5 * (1 + torch.log(sigma**2) - mu.pow(2) - sigma**2).sum() # sigma是标准差
    return -0.5 * (1 + sigma - mu.pow(2) - sigma.exp()).sum() # 原来的错误版本

def compute_reg_loss(image_logsigma, text_logsigma, img_margin_value=300, text_margin_value=300, margin_weight=1): # RSTP 1  ICFG 10  v2
    margin_loss1 = margin_entropy_loss(img_margin_value, image_logsigma)
    margin_loss2 = margin_entropy_loss(text_margin_value, text_logsigma)
    margin_loss = (margin_loss1 + margin_loss2) 
    margin_loss = margin_weight * margin_loss
    return margin_loss

def margin_entropy_loss(margin, logsigma): # 见PUM和ReID那两篇文章
    feat_dim = logsigma.shape[-1]
    entropy = float(feat_dim / 2 * (np.log(2 * np.pi) + 1)) + torch.sum(logsigma, -1) / 2 # []
    zero = torch.zeros_like(entropy)
    loss = torch.max(margin - entropy, zero)
    loss = torch.mean(loss) 
    return loss 

def compute_specific_location_simi_loss(ori_text_embeds, fusion_embeds, indices):
    ori_text_embeds = ori_text_embeds / ori_text_embeds.norm(dim=-1, keepdim=True)
    fusion_embeds = fusion_embeds / fusion_embeds.norm(dim=-1, keepdim=True)
    
    mask = (indices == 1)
    A_at_indices = ori_text_embeds[mask]
    B_at_indices = fusion_embeds[mask]
    cosine_sim = F.cosine_similarity(A_at_indices, B_at_indices, dim=-1)
    total_similarity = cosine_sim.mean()
    return 1 - total_similarity
