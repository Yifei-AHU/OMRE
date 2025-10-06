from model import objectives
from .clip_model3 import ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.cuda.amp import autocast

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.point_temperature = torch.ones([]) * (1 / args.point_temperature)

        self.alpha = args.triplet_alpha
        self.triplet_margin = args.margin

        self.gaussian = args.gaussian
        self.num_head = 8
        self.a = 64

        if self.gaussian:
            self.sample_num = args.sample_num # 
            self.mu_num = args.mu_num # 1
            self.margin = args.margin_loss
            self.margin_value = args.margin_value # 300
            self.margin_weight = args.margin_weight
            self.text_margin_value = args.text_margin_value

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
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
        image_mu, image_sigma = self.base_model.encode_image(image)
        image_mu = image_mu[:,0,:].float()
        image_sigma = image_sigma[:,0,:].float()
        img_sample_feats, img_mu, _ = self.img_gaussian_modeling(image_mu, image_sigma)
        return img_mu, img_sample_feats
        # return image_mu, img_sample_feats[:,0,:]

    def new_encode_text(self, text):
        text_mu, text_sigma = self.base_model.encode_text(text)
        text_mu = text_mu[torch.arange(text_mu.shape[0]), text.argmax(dim=-1)].float()
        text_sigma = text_sigma[torch.arange(text_sigma.shape[0]), text.argmax(dim=-1)].float()
        text_sample_feats, text_mu, _ = self.text_gaussian_modeling(text_mu, text_sigma)
        return text_mu, text_sample_feats

    def img_gaussian_modeling(self, image_mu, image_sigma):
        # z_img = [image_mu] * self.mu_num
        z_img = []
        for _ in range(self.sample_num):
            eps = torch.randn(image_sigma.shape[0], image_sigma.shape[1], device=image_mu.device)
            z1 = image_mu + torch.exp(image_sigma) * eps
            # z1 = image_mu + image_sigma * eps
            z_img.append(z1)
        image_embed = torch.stack(z_img, dim=1) # [batch, num, 512] 每一个样本对应的num个采样点

        return image_embed, image_mu, image_sigma

    def text_gaussian_modeling(self, text_mu, text_sigma):
        # z_text = [text_mu] * self.mu_num
        z_text = []
        for _ in range(self.sample_num):
            eps = torch.randn(text_sigma.shape[0], text_sigma.shape[1], device=text_mu.device)
            z1 = text_mu + torch.exp(text_sigma) * eps
            # z1 = text_mu + text_sigma * eps  # 标准差
            z_text.append(z1)
        text_embed = torch.stack(z_text, dim=1) # [batch, num, 193, 512] 每一个样本对应的num个采样点

        return text_embed, text_mu, text_sigma

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        with autocast():
            image_mu, img_sigma, text_mu, text_sigma = self.base_model(images, caption_ids)

        image_mu_class_token = image_mu[:,0,:].float()
        img_sigma_class_token = img_sigma[:,0,:].float()
        text_mu_class_token = text_mu[torch.arange(text_mu.shape[0]), caption_ids.argmax(dim=-1)].float()
        text_sigma_class_token = text_sigma[torch.arange(text_sigma.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale

        if self.gaussian:
            image_embeds, img_mu, img_sigma = self.img_gaussian_modeling(image_mu_class_token, img_sigma_class_token)
            mlm_text_embeds, mlm_text_mu, mlm_text_sigma = self.text_gaussian_modeling(text_mu_class_token, text_sigma_class_token)

        sample_image_embeds = image_embeds
        sample_mlm_text_embeds = mlm_text_embeds
 
        # image_embeds = image_embeds[:,0,:,:]
        # mlm_text_embeds = mlm_text_embeds[:,0,:,:]
        # sample_image_class_embeds = image_embeds[:,0,:].float()
        # sample_mlm_text_class_embeds = mlm_text_embeds[torch.arange(mlm_text_mu.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        if 'sdm' in self.current_task:
            sdm_loss = objectives.sdm_v5(img_mu, mlm_text_mu, sample_image_embeds, sample_mlm_text_embeds, batch['pids'], logit_scale)
            triplet_loss = objectives.triplet_loss(img_mu, mlm_text_mu, sample_image_embeds, sample_mlm_text_embeds, self.a, batch['pids'])
            diversity_loss = objectives.loss_diversity(sample_image_embeds) + objectives.loss_diversity(sample_mlm_text_embeds)
            mmd_loss = objectives.mmd_rbf_loss(sample_image_embeds, sample_mlm_text_embeds)
            ret.update({'margin_loss':objectives.compute_margin(img_sigma, mlm_text_sigma, img_margin_value=100, text_margin_value=100)})
            ret.update({'diversity_loss': diversity_loss})
            ret.update({'sdm_loss':sdm_loss})
            ret.update({'mmd_loss': mmd_loss})
            ret.update({'triplet_loss': triplet_loss})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'att_mlm' in self.current_task:
            for att_type in ['shoes','hairstyle','genders','top','trousers','belongings']:
                mlm_ids = batch[att_type+'_mlm_ids']

                mlm_feats = self.base_model.encode_text(mlm_ids)

                x = self.cross_former(mlm_feats, image_feats, image_feats)

                x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

                scores = x.float().reshape(-1, self.args.vocab_size)
                mlm_labels = batch[att_type+'_mlm_labels'].reshape(-1)
                ret.update({att_type+'_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

                pred = scores.max(1)[1]
                mlm_label_idx = torch.nonzero(mlm_labels)
                acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                ret.update({att_type+'_acc': acc})

        return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
