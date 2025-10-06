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

    # 计算文本均值->图像采样点的相似度 文本采样点到图像均值的相似度
    i2t_m2s_simi = img_mu_norm @ text_fetures_norm.t() # [b1,b2*n]
    i2t_s2m_simi = image_fetures_norm @ text_mu_norm.t() # [b1*n,b2]
    i2t_m2m_simi = img_mu_norm @ text_mu.t() # [b1, b2]

    # 计算图像均值->文本采样点的相似度 图像采样点到文本均值的相似度
    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b2,b1*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b2*n,b1]
    t2i_m2m_simi = text_mu @ img_mu_norm.t() # [b2, b1]
    
    # 融合相似度
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

def sdm_v6(img_mu, text_mu, image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):  
    batch_size = image_fetures.size(0)
    sample_num = image_fetures.shape[1] # image_fetures-- [b,num,512]

    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]

    new_pid = pid.repeat_interleave(sample_num).reshape((batch_size*sample_num, 1))
    
    labels = (pid == new_pid.T).float()
    trans_labels = labels.t()
    
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

    # 计算文本均值->图像采样点的相似度 文本采样点到图像均值的相似度
    i2t_m2s_simi = img_mu_norm @ text_fetures_norm.t() # [b,b*n]
    i2t_s2m_simi = image_fetures_norm @ text_mu_norm.t() # [b*n,b]

    # 计算图像均值->文本采样点的相似度 图像采样点到文本均值的相似度
    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b,b*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b*n,b]

    labels_distribute = labels / labels.sum(dim=1, keepdim=True) # [b,b*k]
    trans_labels_distribute = trans_labels / trans_labels.sum(dim=-1,keepdim=True)
    
    # 图像采样点与文本均值点
    t2i_m2s_simi_logits = logit_scale * t2i_m2s_simi
    i2t_s2m_simi_logits = logit_scale * i2t_s2m_simi

    # 图像均值点与文本采样点
    i2t_m2s_simi_logits = i2t_m2s_simi * logit_scale
    t2i_s2m_simi_logits = t2i_s2m_simi * logit_scale

    # 图像采样点与文本均值点 互相的损失
    i2t_pred = F.softmax(i2t_s2m_simi_logits, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(i2t_s2m_simi_logits, dim=1) - torch.log(trans_labels_distribute + epsilon))
    t2i_pred = F.softmax(t2i_m2s_simi_logits, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_m2s_simi_logits, dim=1) - torch.log(labels_distribute + epsilon))

    loss_1 = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    # 图像均值点与文本采样点
    i2t_pred = F.softmax(i2t_m2s_simi_logits, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(i2t_m2s_simi_logits, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(t2i_s2m_simi_logits, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_s2m_simi_logits, dim=1) - torch.log(trans_labels_distribute + epsilon))

    loss_2 = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    loss = (loss_1 + loss_2) / 2
    return loss

def multi_instance_infonce(img_mu, text_mu, image_fetures, text_fetures, pid, logit_scale=0.07, image_id=None, factor=0.3, epsilon=1e-8):  
    batch_size = image_fetures.size(0)
    sample_num = image_fetures.shape[1] # image_fetures-- [b,num,512]

    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]

    new_pid = pid.repeat_interleave(sample_num).reshape((batch_size*sample_num, 1))
    
    labels = (pid == new_pid.T).float() # [b,b*n] 正样本掩码
    trans_labels = labels.t()
    
    img_mu_norm = img_mu / img_mu.norm(dim=-1, keepdim=True) # [b,512]
    text_mu_norm = text_mu / text_mu.norm(dim=-1, keepdim=True) # [b,512]

    image_fetures_norm = image_fetures / image_fetures.norm(dim=-1, keepdim=True)
    text_fetures_norm = text_fetures / text_fetures.norm(dim=-1, keepdim=True)

    image_fetures_norm = image_fetures_norm.view(-1, image_fetures_norm.shape[-1]) # [b*n, 512]
    text_fetures_norm = text_fetures_norm.view(-1, text_fetures_norm.shape[-1]) # [b*n, 512]

    logit_scale = 1 / logit_scale

    # 计算文本均值->图像采样点的相似度 文本采样点到图像均值的相似度
    i2t_m2s_simi = logit_scale * (img_mu_norm @ text_fetures_norm.t()) # [b,b*n]
    i2t_s2m_simi = logit_scale * (image_fetures_norm @ text_mu_norm.t()) # [b*n,b]

    # 计算图像均值->文本采样点的相似度 图像采样点到文本均值的相似度
    t2i_m2s_simi = logit_scale * (text_mu_norm @ image_fetures_norm.t()) # [b,b*n]
    t2i_s2m_simi = logit_scale * (text_fetures_norm @ img_mu_norm.t())  # [b*n,b]

    # 计算所有正样本相似度
    pos_i2t_m2s_sim = i2t_m2s_simi * labels
    pos_sum_i2t_m2s = torch.sum(torch.exp(pos_i2t_m2s_sim), dim=1)  # (N,)
    all_i2t_m2s_sim = torch.exp(i2t_m2s_simi).sum(dim=1) 
    loss_i2t_m2s = -torch.log(pos_sum_i2t_m2s / all_i2t_m2s_sim + epsilon) 

    pos_t2i_s2m_simi = t2i_s2m_simi * trans_labels
    pos_sum_t2i_s2m = torch.sum(torch.exp(pos_t2i_s2m_simi), dim=1)  # (N,)
    all_t2i_s2m_sim = torch.exp(t2i_s2m_simi).sum(dim=1) 
    loss_t2i_s2m = -torch.log(pos_sum_t2i_s2m / all_t2i_s2m_sim + epsilon) 

    pos_t2i_m2s_simi = t2i_m2s_simi * labels
    pos_sum_t2i_m2s = torch.sum(torch.exp(pos_t2i_m2s_simi), dim=1)  # (N,)
    all_t2i_m2s_sim = torch.exp(t2i_m2s_simi).sum(dim=1) 
    loss_t2i_m2s = -torch.log(pos_sum_t2i_m2s / all_t2i_m2s_sim + epsilon) 

    pos_i2t_s2m_simi = i2t_s2m_simi * trans_labels
    pos_sum_i2t_s2m = torch.sum(torch.exp(pos_i2t_s2m_simi), dim=1)
    all_i2t_s2m_simi = torch.exp(i2t_s2m_simi).sum(dim=1) 
    loss_i2t_s2m= -torch.log(pos_sum_i2t_s2m / all_i2t_s2m_simi + epsilon) 

    loss_1 = (loss_i2t_m2s.mean() + loss_t2i_s2m.mean()) / 2
    loss_2 = (loss_t2i_m2s.mean() + loss_i2t_s2m.mean()) / 2

    return loss_1 + loss_2
 

def rowwise_softmax(x):
    result = torch.zeros_like(x)
    for i in range(x.size(0)):
        non_zero_elements = x[i][x[i] != 0]
        softmax_values = F.softmax(non_zero_elements, dim=0)
        result[i][x[i] != 0] = softmax_values
    return result

def soft_sdm_v4(img_mu, text_mu, image_fetures, text_fetures, pid, logit_scale, a, image_id=None, factor=0.3, epsilon=1e-8):  
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

    # 计算文本均值->图像采样点的相似度 文本采样点到图像均值的相似度
    i2t_m2s_simi = img_mu_norm @ text_fetures_norm.t() # [b,b*n]
    i2t_s2m_simi = image_fetures_norm @ text_mu_norm.t() # [b*n,b]
    i2t_m2m_simi = img_mu_norm @ text_mu.t() # [b, b]

    # 计算图像均值->文本采样点的相似度 图像采样点到文本均值的相似度
    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b,b*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b*n,b]
    t2i_m2m_simi = text_mu @ img_mu_norm.t()
    
    # 融合相似度
    i2t_simi = torch.log(torch.exp(a * i2t_m2s_simi).reshape(batch_size,batch_size,sample_num).sum(dim=-1) + torch.exp(a * i2t_s2m_simi).reshape(batch_size,sample_num,batch_size).sum(1)) / (a*(sample_num))
    t2i_simi = torch.log(torch.exp(a * t2i_m2s_simi).reshape(batch_size,batch_size,sample_num).sum(dim=-1) + torch.exp(a * t2i_s2m_simi).reshape(batch_size,sample_num,batch_size).sum(1)) / (a*(sample_num))

    t2i_simi = logit_scale * t2i_simi
    i2t_simi = logit_scale * i2t_simi

    # 根据均值相似度进行软标签设定
    soft_label = i2t_m2m_simi * labels 
    soft_label = rowwise_softmax(soft_label)
    
    # labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(i2t_simi, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(i2t_simi, dim=1) - torch.log(soft_label + epsilon))
    t2i_pred = F.softmax(t2i_simi, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_simi, dim=1) - torch.log(soft_label + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def sdm_v3(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
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

    image_fetures_norm = image_fetures / image_fetures.norm(dim=-1, keepdim=True)
    text_fetures_norm = text_fetures / text_fetures.norm(dim=-1, keepdim=True)

    image_fetures_norm = image_fetures_norm.view(-1, image_fetures_norm.shape[-1]) # [b*n, 512]
    text_fetures_norm = text_fetures_norm.view(-1, text_fetures_norm.shape[-1]) # [b*n, 512]

    i2t_simi = image_fetures_norm @ text_fetures_norm.t() # [b*n, b*n]
    t2i_simi = i2t_simi.t()

    new_i2t_simi = i2t_simi.view(batch_size,sample_num,-1).permute(0,2,1).reshape(batch_size,text_fetures.shape[0],sample_num,sample_num).permute(0,1,3,2) # [b,b,n,n]
    new_i2t_simi = torch.exp(new_i2t_simi) # p_ij
    i2t_simi = torch.sum(new_i2t_simi[:,:,0,:], dim=-1) + torch.sum(new_i2t_simi[:,:,1:,0], dim=-1)

    new_t2i_simi = t2i_simi.view(batch_size,sample_num,-1).permute(0,2,1).reshape(batch_size,text_fetures.shape[0],sample_num,sample_num).permute(0,1,3,2) # [b,b,n,n]
    new_t2i_simi = torch.exp(new_t2i_simi) # p_ij
    t2i_simi = torch.sum(new_t2i_simi[:,:,0,:], dim=-1) + torch.sum(new_t2i_simi[:,:,1:,0], dim=-1)

    labels_distribute = labels / labels.sum(dim=1)

    final_i2t_simi = i2t_simi / torch.sum(i2t_simi, dim=-1, keepdim=True)  # 最终的p_ij
    final_t2i_simi = t2i_simi / torch.sum(t2i_simi, dim=-1, keepdim=True)

    i2t_loss = final_i2t_simi * (torch.log(final_i2t_simi) - torch.log(labels_distribute + epsilon))
    t2i_loss = final_t2i_simi * (torch.log(final_t2i_simi) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def loss_diversity(sample_feats1):
    batch_size = sample_feats1.shape[0] 

    sample_num = sample_feats1.shape[1] # sample_feats-- [b,num,512]

    sample_feats1_norm = sample_feats1 / sample_feats1.norm(dim=-1, keepdim=True) # 归一化--之后计算相似度

    similarity = sample_feats1_norm.bmm(sample_feats1_norm.transpose(1,2))
    I = torch.autograd.Variable((torch.eye(similarity.size(-1)) > 0.5).repeat(similarity.size(0), 1, 1)).to(similarity.device)

    similarity.masked_fill_(I, 0.0)

    loss = torch.stack([torch.norm(g, p=2) for g in similarity]) / ((sample_num-1)**2)
    return loss.mean() 

def loss_diversity2(sample_feats1):
    batch_size = sample_feats1.shape[0] 
    num_samples = sample_feats1.shape[1] 
    num_tokens = sample_feats1.shape[2] 
    feature_dim = sample_feats1.shape[3]  # 512

    # 1. 先对每个token的特征进行归一化
    sample_feats1_norm = sample_feats1 / sample_feats1.norm(dim=-1, keepdim=True)  # [batch_size, num_samples, num_tokens, 512]

    # 2. 将所有token的特征合并，以便计算整体的相似度
    sample_feats1_norm = sample_feats1_norm.view(batch_size, num_samples, -1)  # [batch_size, num_samples, num_tokens * 512]

    # 3. 计算相似度矩阵
    similarity = sample_feats1_norm.bmm(sample_feats1_norm.transpose(1,2))  # [batch_size, num_samples, num_samples]

    # 4. 屏蔽对角线元素（即自相似性）
    I = torch.autograd.Variable((torch.eye(similarity.size(-1)) > 0.5).repeat(similarity.size(0), 1, 1)).to(similarity.device)
    similarity.masked_fill_(I, 0.0)

    # 5. 计算多样性损失
    loss = torch.stack([torch.norm(g, p=2) for g in similarity]) / ((num_samples-1)**2)
    return loss.mean()

def infonce_v4(img_mu, text_mu, image_fetures, text_fetures, pid, logit_scale, a, image_id=None, factor=0.3, epsilon=1e-8):  
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
    i2t_m2m_simi = img_mu_norm @ text_mu.t() # [b, b]

    # 计算图像均值->文本采样点的相似度 图像采样点到文本均值的相似度
    t2i_m2s_simi = text_mu_norm @ image_fetures_norm.t() # [b,b*n]
    t2i_s2m_simi = text_fetures_norm @ img_mu_norm.t()  # [b*n,b]
    t2i_m2m_simi = text_mu @ img_mu_norm.t()
    
    # 融合相似度
    i2t_simi = torch.log(torch.exp(a * i2t_m2s_simi).reshape(batch_size,batch_size,sample_num).sum(dim=-1) + torch.exp(a * i2t_s2m_simi).reshape(batch_size,sample_num,batch_size).sum(1)) / (a*(sample_num))
    t2i_simi = torch.log(torch.exp(a * t2i_m2s_simi).reshape(batch_size,batch_size,sample_num).sum(dim=-1) + torch.exp(a * t2i_s2m_simi).reshape(batch_size,sample_num,batch_size).sum(1)) / (a*(sample_num))

    t2i_simi = logit_scale * t2i_simi
    i2t_simi = logit_scale * i2t_simi

    labels = torch.arange(batch_size, device=img_mu.device)
    loss = F.cross_entropy(i2t_simi, labels) + F.cross_entropy(t2i_simi, labels)
    return loss

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

# def compute_margin(image_logsigma, text_logsigma, img_margin_value=300, text_margin_value=300, margin_weight=1): # RSTP 1  ICFG 10
#     if image_logsigma is not None:
#         margin_loss1 = margin_entropy_loss(img_margin_value, image_logsigma)
#     else:
#         margin_loss1 = 0
#     if text_logsigma is not None:
#         margin_loss2 = margin_entropy_loss(text_margin_value, text_logsigma)
#     else:
#         margin_loss2 = 0
#     margin_loss = (margin_loss1 + margin_loss2) / 2
#     margin_loss = margin_weight * margin_loss
#     return margin_loss


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

    # 添加归一化
    ori_text_embeds = ori_text_embeds / ori_text_embeds.norm(dim=-1, keepdim=True)
    fusion_embeds = fusion_embeds / fusion_embeds.norm(dim=-1, keepdim=True)
    
    mask = (indices == 1)
    A_at_indices = ori_text_embeds[mask]
    B_at_indices = fusion_embeds[mask]
    cosine_sim = F.cosine_similarity(A_at_indices, B_at_indices, dim=-1)
    total_similarity = cosine_sim.mean()
    return 1 - total_similarity