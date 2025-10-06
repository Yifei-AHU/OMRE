from model import objectives
from .clip_model3 import ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer, Transformer2, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.cuda.amp import autocast

class OMRE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.gaussian = args.gaussian

        if self.gaussian:
            self.sample_num = args.sample_num # 
            self.mu_num = args.mu_num # 1
            self.margin = args.margin_loss
            self.margin_value = args.margin_value # 300
            self.margin_weight = args.margin_weight

        if 'crsr' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer2(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def new_encode_image(self, image):
        fix_img_embedding, image_mu, image_sigma = self.base_model.encode_image(image)
        image_class_mu = image_mu[:,0,:].float()
        image_class_sigma = image_sigma[:,0,:].float()
        img_sample_feats, _, _ = self.img_gaussian_modeling(image_mu, image_sigma)
        return image_class_mu, img_sample_feats[:,:,0,:].float()#, fix_img_embedding

    def new_encode_text(self, text):
        _, text_mu, text_sigma = self.base_model.encode_text(text)
        text_class_mu = text_mu[torch.arange(text_mu.shape[0]), text.argmax(dim=-1)].float()
        text_class_sigma = text_sigma[torch.arange(text_sigma.shape[0]), text.argmax(dim=-1)].float()
        text_sample_feats, _, _ = self.text_gaussian_modeling(text_mu, text_sigma)
        return text_class_mu, torch.stack([text_sample_feats[i][torch.arange(text_sample_feats[i].shape[0]), text[i].argmax(dim=-1)] for i in range(text_sample_feats.shape[0])], dim=0).float()
        
    def img_gaussian_modeling(self, image_mu, image_sigma):
        z_img = []
        for _ in range(self.sample_num):
            eps = torch.randn(image_sigma.shape[0], image_sigma.shape[1], image_sigma.shape[2], device=image_mu.device)  # 完全的token
            z1 = image_mu + torch.exp(image_sigma) * eps
            z_img.append(z1)
        image_embed = torch.stack(z_img, dim=1) # [batch, num, 512] 

        return image_embed, image_mu, image_sigma

    def text_gaussian_modeling(self, text_mu, text_sigma):
        z_text = []
        for _ in range(self.sample_num):
            eps = torch.randn(text_sigma.shape[0], text_sigma.shape[1], text_sigma.shape[2], device=text_mu.device)
            z1 = text_mu + torch.exp(text_sigma) * eps
            z_text.append(z1)
        text_embed = torch.stack(z_text, dim=1) # [batch, num, 193, 512] 

        return text_embed, text_mu, text_sigma

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        ori_caption_ids = batch['ori_caption_ids']

        with autocast():
            image_mu, img_sigma, text_mu, text_sigma, ori_text_feat, fix_img_embedding, fix_text_embedding = self.base_model(images, caption_ids, ori_caption_ids)
            ori_text_feat = ori_text_feat.detach()

        logit_scale = self.logit_scale

        if self.gaussian:
            # 全部的token
            image_embeds, img_mu, img_sigma = self.img_gaussian_modeling(image_mu, img_sigma)
            mlm_text_embeds, mlm_text_mu, mlm_text_sigma = self.text_gaussian_modeling(text_mu, text_sigma) # [b,num,77,512]

            image_mu_class_token = img_mu[:,0,:].float()
            text_mu_class_token = mlm_text_mu[torch.arange(mlm_text_mu.shape[0]), caption_ids.argmax(dim=-1)].float()

        # 获取 采样点与均值的 class token 
        sample_image_class_embeds = image_embeds[:,:,0,:].contiguous().float() # [b,num,512]
        sample_mlm_text_class_embeds = torch.stack([mlm_text_embeds[i][torch.arange(mlm_text_embeds[i].shape[0]), caption_ids[i].argmax(dim=-1)] for i in range(mlm_text_embeds.shape[0])], dim=0).float()

        if 'boma' in self.current_task:
            boma_loss = objectives.compute_boma_loss(image_mu_class_token, text_mu_class_token, sample_image_class_embeds, sample_mlm_text_class_embeds, batch['pids'], logit_scale)
            ret.update({'boma_loss':boma_loss})
        
        if 'reg' in self.current_task:
            reg_loss = objectives.compute_reg_loss(img_sigma, mlm_text_sigma, img_margin_value=400, text_margin_value=400, margin_weight=10)
            ret.update({'reg_loss': reg_loss}) 
        
        if 'hnm' in self.current_task: 
            hnm_loss = objectives.compute_hnm_loss(image_mu_class_token, text_mu_class_token, sample_image_class_embeds, sample_mlm_text_class_embeds, batch['pids'])
            ret.update({'hnm_loss':hnm_loss})
        
        if 'crsr' in self.current_task: 
            crsr_loss = []
            mlm_labels = batch['mlm_labels']
            mlm_labels = mlm_labels.clamp(min=0, max=1)
            for i in range(self.sample_num):
                image_feats_i = image_embeds[:,i,:,:]  
                with autocast():
                    recover_i = self.cross_former(mlm_text_mu, image_feats_i, image_feats_i)
                    crsr_loss_i = objectives.compute_specific_location_simi_loss(ori_text_feat, recover_i, mlm_labels)
                    crsr_loss.append(crsr_loss_i)
            crsr_loss = sum(crsr_loss) / self.sample_num
            ret.update({'crsr_loss': crsr_loss})
        
        return ret

def build_finetune_model(args, num_classes=11003):
    model = OMRE(args, num_classes)
    convert_weights(model)
    return model