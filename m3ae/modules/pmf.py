import torch
import argparse
import torch.nn as nn

class MLP_adapter(nn.Module):
    # Non-Linear Transformation in the paper, acting as the translator between modalities.
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class PMF(nn.Module):
    def __init__(self, config, language_encoder, vision_encoder):
        super().__init__()

        self.config = config
        # self.bert_encoder = BertModel.from_pretrained('downloaded/bert-{}-uncased'.format(args.bert_sz))
        # 替换bert模型的原始输出层（分类头）
        # self.bert_encoder.heads = nn.Linear(self.bert_encoder.config.hidden_size, args.n_classes)
        # self.vit_encoder = ViTModel.from_pretrained('downloaded/vit-{}-patch16-224'.format(args.vit_sz))
        # 将ViT的隐藏状态映射到类别空间
        # self.vit_encoder.heads = nn.Linear(self.vit_encoder.config.hidden_size, args.n_classes)

        # # 验证传入的参数args是否有效
        # self.config = self.check_args(config)

        # v2t: vision-to-text. t2v: text-to-vision
        self.bert_encoder = language_encoder
        self.bert_encoder.heads = nn.Linear(config['hidden_size'], config['n_classes'])
        self.vit_encoder = vision_encoder
        self.vit_encoder.heads = nn.Linear(config['hidden_size'], config['n_classes'])

        # 创建一个参数列表用于存储视觉到文本融合过程中的查询提示（QP）向量，将这些张量包装为模型的参数，在训练过程中被优化
        self.v2t_qp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, config['n_qp'], config['hidden_size']).normal_(std=0.02)) for _ in
             range(config['num_top_layer'])])
        self.t2v_qp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, config['n_qp'], config['hidden_size']).normal_(std=0.02)) for _ in
             range(config['num_top_layer'])])

        # 创建一个参数列表用于存储视觉到文本融合过程中的查询上下文提示（QCP）向量
        self.v2t_qcp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, config['n_qcp'], config['hidden_size']).normal_(std=0.02)) for _ in
             range(config['num_top_layer'])])
        self.t2v_qcp = nn.ParameterList(
            [nn.Parameter(torch.empty(1, config['n_qcp'], config['hidden_size']).normal_(std=0.02)) for _ in
             range(config['num_top_layer'])])

        # 创建一个参数列表用于存储视觉到文本融合过程中的融合上下文提示（FCP）向量
        if config['n_fcp'] > 0:
            self.vision_fcp = nn.ParameterList(
                [nn.Parameter(torch.empty(1, config['n_fcp'], config['hidden_size']).normal_(std=0.02)) for _
                 in range(config['num_top_layer'])])
            self.text_fcp = nn.ParameterList(
                [nn.Parameter(torch.empty(1, config['n_fcp'], config['hidden_size']).normal_(std=0.02)) for _
                 in range(config['num_top_layer'])])

        # v2t_trans用于存储将视觉特征转换为文本特征空间的MLP_adapter
        self.v2t_trans = nn.ModuleList(
            [MLP_adapter(config['hidden_size'], config['mlp_hidden_sz'], config['hidden_size'])
             for _ in range(config['num_top_layer'])])
        # t2v_trans用于存储将文本特征转换为视觉特征空间的MLP_adapter
        self.t2v_trans = nn.ModuleList(
            [MLP_adapter(config['hidden_size'], config['mlp_hidden_sz'], config['hidden_size'])
             for _ in range(config['num_top_layer'])])

        # self.grad_control()

    def forward(self, image, txt_input_ids, txt_attn_mask, layer_idx, vit_encoder, bert_encoder):
        n = image.shape[0]
        device = image.device
        assert image.shape[0] == txt_input_ids.shape[0]

        # # pre_processing before two encoders
        # img_tokens = self.vit_encoder.embeddings(image)
        # txt_tokens = self.bert_encoder.embeddings(txt_input_ids, txt_token_type_ids)
        #
        # ## generate extra txt attn mask
        # txt_attn_mask = self.get_extended_txt_attn_mask(txt_attn_mask)
        max_prompt_length = self.config['n_qp'] + self.config['n_qcp'] + self.config['n_fcp']
        batch_extra_attn_mask = torch.ones(n, max_prompt_length).to(device)
        batch_extra_attn_mask = self.get_extended_txt_attn_mask(batch_extra_attn_mask)
        #
        # # main forward
        # ## unimodal base feature extraction
        # for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers - self.args.n_fusion_layers):
        #     txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](txt_tokens, txt_attn_mask)[0] # (n, seq_len, hidden_size)
        # for vit_layer_id in range(self.vit_encoder.config.num_hidden_layers - self.args.n_fusion_layers):
        #     img_tokens = self.vit_encoder.encoder.layer[vit_layer_id](img_tokens)[0]

        ## multimodal fusion layers
        for fusion_layer_id in range(layer_idx-1):
            ### get prompts
            # 获取v2t和t2v的QP、QCP、FCP，并扩展到batch维度n
            batch_v2t_qp = self.v2t_qp[fusion_layer_id].expand(n, -1, -1).to(device)
            batch_t2v_qp = self.t2v_qp[fusion_layer_id].expand(n, -1, -1).to(device)

            batch_v2t_qcp = self.v2t_qcp[fusion_layer_id].expand(n, -1, -1).to(device)
            batch_t2v_qcp = self.t2v_qcp[fusion_layer_id].expand(n, -1, -1).to(device)

            if self.config['n_fcp'] > 0:
                batch_vision_fcp = self.vision_fcp[fusion_layer_id].expand(n, -1, -1).to(device)
                batch_text_fcp = self.text_fcp[fusion_layer_id].expand(n, -1, -1).to(device)

            ### Query Stage
            # prepare text attn_mask
            ## slice attn_mask for corresponding text prompts
            # 从扩展的注意力掩码中切片，以获取与查询提示和上下文提示相对应的掩码
            layer_t2v_qcp_attn_mask = batch_extra_attn_mask[:, :, :, :self.config['n_qcp']]
            layer_t2v_qp_attn_mask = batch_extra_attn_mask[:, :, :, :self.config['n_qp']]
            layer_text_fcp_attn_mask = batch_extra_attn_mask[:, :, :, :self.config['n_fcp']]
            layer_v2t_qp_attn_mask = batch_extra_attn_mask[:, :, :, :self.config['n_qp']]

            ## reform text attn_mask
            # 将不同的掩码连接起来，形成查询阶段和融合阶段所需的新的注意力掩码
            query_txt_attn_mask = torch.cat([txt_attn_mask, layer_t2v_qcp_attn_mask, layer_t2v_qp_attn_mask], dim=3)
            fusion_txt_attn_mask = torch.cat([txt_attn_mask, layer_text_fcp_attn_mask, layer_v2t_qp_attn_mask], dim=3)

            # for t2v: get text fusion intermediate hidden-state for ViT
            query_txt_tokens = torch.cat([txt_input_ids, batch_t2v_qcp, batch_t2v_qp], dim=1)
            t2v_fusion_intermediate = bert_encoder.encoder.layer[layer_idx + fusion_layer_id + 1](
                query_txt_tokens, query_txt_attn_mask)
            t2v_fusion_intermediate = t2v_fusion_intermediate[0][:, -self.config['n_qp']:, :]
            t2v_fusion_intermediate = self.t2v_trans[fusion_layer_id](t2v_fusion_intermediate).to(device)

            # for v2t: get vision fusion intermediate hidden-state for BERT
            query_img_tokens = torch.cat([image, batch_v2t_qcp, batch_v2t_qp], dim=1)
            # v2t_fusion_intermediate = vit_encoder.encoder.layer[layer_idx + fusion_layer_id + 1](query_img_tokens)
            # TODO:vit_encoder没有指定层数，直接调用了forward_trans(合理性待验证
            # v2t_fusion_intermediate = vit_encoder.forward_trans_layer(query_img_tokens, layer_idx + fusion_layer_id + 1)
            v2t_fusion_intermediate = vit_encoder.forward_trans(query_img_tokens)
            v2t_fusion_intermediate = v2t_fusion_intermediate[:, -self.config['n_qp']:, :]
            v2t_fusion_intermediate = self.v2t_trans[fusion_layer_id](v2t_fusion_intermediate).to(device)

            # Fusion Stage
            img_tokens = torch.cat([image, batch_vision_fcp, t2v_fusion_intermediate], dim=1)
            txt_tokens = torch.cat([txt_input_ids, batch_text_fcp, v2t_fusion_intermediate], dim=1)
            # TODO:vit_encoder没有指定层数，直接调用了forward_trans(合理性待验证
            # img_tokens = vit_encoder.forward_trans_layer(img_tokens, layer_idx + fusion_layer_id + 1)
            img_tokens = vit_encoder.forward_trans(query_img_tokens)
            txt_tokens = \
            bert_encoder.encoder.layer[layer_idx + fusion_layer_id + 1](txt_tokens, fusion_txt_attn_mask)[0]

            txt_tokens = txt_tokens[:, :-self.config['n_qp'] - self.config['n_fcp'], :]
            img_tokens = img_tokens[:, :-self.config['n_qp'] - self.config['n_fcp'], :]

        # after main forwards
        txt_tokens = txt_tokens[:, 0]
        # TODO：vit_encoder没有layernorm，暂时先省略（合理性待验证
        # img_tokens = vit_encoder.layernorm(img_tokens)
        img_tokens = img_tokens[:, 0]

        # txt_pred = self.bert_encoder.heads(txt_tokens)
        # img_pred = self.vit_encoder.heads(img_tokens)

        return txt_tokens, img_tokens

    # 将文本注意力掩码扩展到与模型中的多头注意力机制兼容的形状，并对其进行适当的转换（1，8）->（1，1，1，8）
    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def check_args(self, args):
        assert args.n_qp > 0
        assert args.n_fusion_layers <= min(self.bert_encoder.config.num_hidden_layers,
                                           self.vit_encoder.config.num_hidden_layers)
        if args.mlp_hidden_sz == -1:
            args.mlp_hidden_sz = max(int(self.vit_encoder.config.hidden_size / 2),
                                     int(self.bert_encoder.config.hidden_size / 2))
        return args

    # 这个方法确保只有特定引入的可训练模块的参数计算梯度，其他所有参数梯度被禁用
    def grad_control(self):
        # Does not require grad for parameters other than the introduced trainable modules
        trainable_modules = [self.v2t_qp, self.t2v_qp,
                             self.v2t_qcp, self.t2v_qcp,
                             self.v2t_trans.modules(), self.t2v_trans.modules(),
                             self.bert_encoder.heads.modules(),
                             self.vit_encoder.heads.modules()]
        if hasattr(self, 'fcp'):
            trainable_modules.append(self.vision_fcp)
            trainable_modules.append(self.text_fcp)

        for module in self.modules():
            module.requires_grad_(False)

        for module in trainable_modules:
            for item in module:
                item.requires_grad_(True)
