# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
from torch.nn import CrossEntropyLoss, MSELoss
from losses import NTXentLoss,AlignLoss
import torch.nn.functional as F
import random
import json
import os
from itertools import chain

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            return self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx,
                                output_attentions=True, output_hidden_states=True)
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1), output_attentions=True,
                                output_hidden_states=True)

    def ori_loss(self, code_inputs, code_outputs, nl_outputs):
        #get code and nl vectors
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        
        #calculate scores and loss
        scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device = scores.device))
        
        return loss

    @staticmethod
    def js_divergence(p, q, epsilon=1e-8):
        """
        Compute Jensen-Shannon Divergence between two distributions P and Q.
        p: Tensor representing the first distribution.
        q: Tensor representing the second distribution.
        epsilon: Small constant to avoid log(0).
        """
        # Ensure probabilities are normalized
        p = p / (p.sum() + epsilon)
        q = q / (q.sum() + epsilon)

        # M is the midpoint distribution
        m = 0.5 * (p + q)

        # Compute KL divergences
        kl_pm = F.kl_div(torch.log(p + epsilon), m, reduction="sum")
        kl_qm = F.kl_div(torch.log(q + epsilon), m, reduction="sum")

        # JS Divergence
        js = 0.5 * (kl_pm + kl_qm)
        return js


    def batch_alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens):
        
        # align_outputs_1 = nl_outputs[0][local_index]
        # align_outputs_2 = code_outputs[0][local_index]

        # use the second last layer hidden states to align
        align_outputs_1 = nl_outputs[2][-2][local_index]
        align_outputs_2 = code_outputs[2][-2][local_index]

        lcs_pairs = sample_align
        loss_align_code = self.build_contrastive_pairs_effecient(align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens)
        # print("loss_align_code",loss_align_code)
        
        # attention loss part
        code_attentions = code_outputs[3]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[3]  # List of tensors, one for each layer

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[local_index, :, 0, :].mean(dim=0) # shape of seq_len
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[local_index, :, 0, :].mean(dim=0) # shape of seq_len


        alignment_code_indices_set = set()
        alignment_nl_indices_set = set()

        for n, m in lcs_pairs:
            if isinstance(m, list):
                alignment_code_indices_set.update(chain.from_iterable(range(m[i], m[i + 1] + 1) for i in range(0, len(m), 2)))
            else:
                alignment_code_indices_set.add(m)
                
            if isinstance(n, list):
                alignment_nl_indices_set.update(chain.from_iterable(range(n[i], n[i + 1] + 1) for i in range(0, len(n), 2)))
            else:
                alignment_nl_indices_set.add(n)

        # 转换为列表并 +1
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices_set]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices_set]

        # 计算 attention loss
        epsilon = 1e-8
        # 直接使用 mask 对 positive 和 negative 的 attention 进行区分
        code_mask = torch.zeros_like(code_cls_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(nl_cls_attention, dtype=torch.bool)
        
        # 对于正样本位置，设置 mask
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        # Compute attention loss
        # 将 CLS 的注意力值转换为概率分布
        code_cls_attention = torch.softmax(code_cls_attention, dim=-1)
        nl_cls_attention = torch.softmax(nl_cls_attention, dim=-1)
        # 将掩码转换为概率分布
        code_mask = code_mask.float()  # 转为浮点数以便后续操作
        nl_mask = nl_mask.float()
        code_mask = code_mask / (code_mask.sum() + 1e-8)  # 避免除以 0
        nl_mask = nl_mask / (nl_mask.sum() + 1e-8)

        attention_loss_code = self.js_divergence(code_cls_attention, code_mask)
        attention_loss_nl = self.js_divergence(nl_cls_attention, nl_mask)

        # Combine the two parts of attention loss
        attention_loss = attention_loss_code + attention_loss_nl

        # Normalize by the number of alignment indices
        attention_loss = attention_loss / (len(alignment_code_indices) + len(alignment_nl_indices))

        # 计算正样本和负样本的 attention loss
        # positive_code_attention = code_cls_attention[code_mask]
        # negative_code_attention = code_cls_attention[~code_mask]
        # positive_nl_attention = nl_cls_attention[nl_mask]
        # negative_nl_attention = nl_cls_attention[~nl_mask]

        # # 计算 attention loss
        # attention_loss = (torch.sum(-torch.log(positive_code_attention + epsilon)) +
        #                   torch.sum(-torch.log(1.0 - negative_code_attention + epsilon)) +
        #                   torch.sum(-torch.log(positive_nl_attention + epsilon)) +
        #                   torch.sum(-torch.log(1.0 - negative_nl_attention + epsilon)))
        #
        # attention_loss = attention_loss / (len(alignment_code_indices) + len(alignment_nl_indices))

        return loss_align_code, attention_loss


    def build_contrastive_pairs_effecient(self, align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens,
                                          num_negative=19):
        loss_align_code = 0
        num_pair = 0
        temperature = 0.2
        temperature_pos = 0.1
        lambda_neg = 1.0

        # 提前创建所有负样本的索引，避免每次循环内重复生成
        all_indices = torch.arange(total_code_tokens, device=align_outputs_2.device)

        for pair in lcs_pairs:
            # 获取 comment 和 code 的 indices
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))

            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))

            # 构建负样本相似度（使用所有非 code_indices 的负样本）
            negative_indices = set(all_indices) - set(code_indices)

            for c_idx in comment_indices:
                num_pair += 1
                comment_embedding = align_outputs_1[c_idx + 1]

                # 计算正样本相似度
                pos_similarities = torch.stack([
                    F.cosine_similarity(comment_embedding, align_outputs_2[code_idx + 1], dim=0).unsqueeze(0)
                    for code_idx in code_indices
                ])

                # 计算负样本相似度
                raw_neg_similarities = torch.stack([
                    F.cosine_similarity(comment_embedding, align_outputs_2[neg_idx + 1], dim=0).unsqueeze(0)
                    for neg_idx in negative_indices
                ])

                # 负样本相似度裁剪到非负范围，然后映射到 [-1, 1]
                neg_similarities = 2 * torch.clamp(raw_neg_similarities, min=0) - 1


                # 对比损失（正负样本部分）
                pos_exp_sim = torch.exp(pos_similarities / temperature_pos)
                neg_exp_sim = torch.exp(neg_similarities / temperature)

                # 正负样本数量
                num_pos = len(code_indices)
                num_neg = len(negative_indices)

                # 正负样本相似度归一化
                pos_similarity_sum = pos_exp_sim.sum() / num_pos
                neg_similarity_sum = neg_exp_sim.sum() / num_neg

                # 计算对比损失
                nt_xent_loss = -torch.log(pos_similarity_sum / (pos_similarity_sum + neg_similarity_sum))

                # 平衡负样本惩罚项
                neg_loss = torch.mean(torch.clamp(-raw_neg_similarities, min=0))
                neg_loss_weighted = neg_loss * (num_neg / (num_pos + num_neg))

                # 总体损失
                total_loss = nt_xent_loss + lambda_neg * neg_loss_weighted
                loss_align_code += total_loss

        return loss_align_code / num_pair

