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

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, output_attentions=True)
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1), output_attentions=True)
            # , output_hidden_states=True
        
    def ori_loss(self, code_inputs, code_outputs, nl_outputs):
        #get code and nl vectors
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        
        #calculate scores and loss
        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
        
        return loss
        
    def alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, labelled_mapped_tokens, common_mapped_tokens, args, return_vec=False, return_scores=False):
        bs = code_inputs.shape[0]
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        # 获取注意力权重
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        for layer_idx in range(len(code_attentions)-1, -1, -1):
            # attention 的形状: (batch_size, num_heads, seq_length, seq_length)
            code_attention = code_attentions[layer_idx].mean(dim=1)  # (batch_size, seq_length, seq_length)
            nl_attention = nl_attentions[layer_idx].mean(dim=1)
            
            # 计算每个 token 对 CLS token 的贡献
            if layer_idx == len(code_attentions)-1:
                code_cls_attention = code_attention[:, 0, :]  # (batch_size, seq_length)
                nl_cls_attention = nl_attention[:, 0, :]
            else:
                code_cls_attention = torch.matmul(code_cls_attention.unsqueeze(1), code_attention).squeeze(1)  # (batch_size, seq_length)
                nl_cls_attention = torch.matmul(nl_cls_attention.unsqueeze(1), nl_attention).squeeze(1)

        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
        
        loss_align_code = 0

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]
        # print(type(align_outputs_2))
        # aaa

        lcs_pairs = sample_align
        selected_align_outputs_1 = []
        selected_align_outputs_2 = []
        n = 0
        for pair in lcs_pairs:
            # pair[0] 是 comment 的区间，pair[1] 是 code 的区间
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            comment_embeddings = [align_outputs_1[idx+1] for idx in comment_indices]
            # print(len(comment_embeddings))
            comment_mean = torch.mean(torch.stack(comment_embeddings), dim=0)
            # selected_align_outputs_1.append(comment_mean)

            # 计算 code 区间的平均值
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))    
            code_embeddings = [align_outputs_2[idx+1] for idx in code_indices]
            # print(len(code_embeddings))
            code_mean = torch.mean(torch.stack(code_embeddings), dim=0)
            # selected_align_outputs_1.append(code_mean)
            # 计算 code 区间的平均值
            # code_embeddings = []
            # for idx in labelled_mapped_tokens[n]:
            #     if idx < len(align_outputs_2):
            #         code_embeddings.append(align_outputs_2[idx])
            #     else:
            #         break
            # print(len(code_embeddings))
            if code_embeddings:  # 检查 code_embeddings 是否为空
                code_mean = torch.mean(torch.stack(code_embeddings), dim=0)
                selected_align_outputs_1.append(comment_mean)
                selected_align_outputs_2.append(code_mean)
            n += 1
        # 将列表转换为张量
        selected_align_outputs_1 = torch.stack(selected_align_outputs_1)
        selected_align_outputs_2 = torch.stack(selected_align_outputs_2)
        # print(selected_align_outputs_1.shape[0], selected_align_outputs_2.shape[0])

        align_loss = NTXentLoss(args, selected_align_outputs_1.shape[0],temperature=3.0)

        loss_align_code += align_loss(selected_align_outputs_1, selected_align_outputs_2)
        loss_align_code = (loss_align_code/len(lcs_pairs))*2

        # code_cls_attention[local_index], nl_cls_attention[local_index]
        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 步骤 1: 展开嵌套列表
        # flat_labelled_mapped_tokens = [item for sublist in labelled_mapped_tokens for item in sublist]
        # 去重处理
        # alignment_code_indices = list(set(flat_labelled_mapped_tokens))
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        # 给 alignment_code_indices 中的每一个值 +1
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]
        # print(alignment_code_indices, alignment_nl_indices)

        # 创建 noisy_code_attention 并复制 code_cls_attention 的值
        noisy_code_attention = code_cls_attention[local_index].clone()
        noisy_nl_attention = nl_cls_attention[local_index].clone()
        # print(noisy_code_attention.shape, noisy_nl_attention.shape)

        # # 将 alignment_code_indices 对应位置的值取相反数
        # noisy_code_attention[0, alignment_code_indices] = -code_cls_attention[0, alignment_code_indices]
        # noisy_nl_attention[0, alignment_nl_indices] = -nl_cls_attention[0, alignment_nl_indices]

        # # 计算 noisy token 的 attention loss
        # noisy_attention_loss = self.calculate_noisy_attention_loss(noisy_code_attention, noisy_nl_attention)

        epsilon = 1e-8
        noisy_code_attention = torch.where(noisy_code_attention == 0, torch.full_like(noisy_code_attention, epsilon), noisy_code_attention)
        noisy_nl_attention = torch.where(noisy_nl_attention == 0, torch.full_like(noisy_nl_attention, epsilon), noisy_nl_attention)
        # 初始化四个张量，全为 epsilon
        positive_code_attention = torch.full_like(noisy_code_attention, epsilon)
        negative_code_attention = torch.full_like(noisy_code_attention, epsilon)
        positive_nl_attention = torch.full_like(noisy_nl_attention, epsilon)
        negative_nl_attention = torch.full_like(noisy_nl_attention, epsilon)

        # 创建一个 mask，用于选择正样本位置
        code_mask = torch.zeros_like(noisy_code_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(noisy_nl_attention, dtype=torch.bool)

        # 对于正样本位置，将值设置为原始的 attention_scores
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        # 更新正样本的得分
        positive_code_attention[code_mask] = noisy_code_attention[code_mask]
        positive_nl_attention[nl_mask] = noisy_nl_attention[nl_mask]

        # 负样本是所有非对齐的位置
        code_neg_mask = ~code_mask
        nl_neg_mask = ~nl_mask

        # 更新负样本的得分
        negative_code_attention[code_neg_mask] = noisy_code_attention[code_neg_mask]
        negative_nl_attention[nl_neg_mask] = noisy_nl_attention[nl_neg_mask]

        # print(positive_code_attention, negative_code_attention)
        # focal loss
        # noisy_attention_loss = (self.focal_attention_loss(positive_code_attention, negative_code_attention) + self.focal_attention_loss(positive_nl_attention, negative_nl_attention))/2

        # regularization loss
        # noisy_attention_loss = (self.regularization_attention_loss(positive_code_attention, negative_code_attention) + self.regularization_attention_loss(positive_nl_attention, negative_nl_attention))/2

        # dynamic_weighting loss
        noisy_attention_loss = (self.dynamic_weighting_attention_loss(positive_code_attention, negative_code_attention) + self.dynamic_weighting_attention_loss(positive_nl_attention, negative_nl_attention))/2
        # print(loss_align_code, noisy_attention_loss)

        if len(common_mapped_tokens) > 0:
            # common_token_loss = self.alignment_loss(align_outputs_2, common_mapped_tokens, args.code_length)
            # 创建一个目标张量，初始值为零
            target = torch.zeros_like(align_outputs_2)
            seq_length = args.code_length
            # 将 common_mapped_tokens 中不在的索引对应的 target 位置设置为 align_outputs_2 中的值
            for idx in range(seq_length):
                if idx not in common_mapped_tokens:
                    target[idx] = align_outputs_2[idx]
            
            # 计算损失
            common_token_loss = F.mse_loss(align_outputs_2, target)
            if noisy_attention_loss > 0:
                total_loss = loss + loss_align_code + noisy_attention_loss + common_token_loss
        else:
            if noisy_attention_loss > 0:
                common_token_loss = 0
                total_loss = loss + loss_align_code + noisy_attention_loss
            else:
                print(noisy_attention_loss)
                print(positive_nl_attention)
                print(negative_nl_attention)
        # total_loss = loss + loss_align_code + noisy_attention_loss

        return loss, total_loss, code_vec, nl_vec, loss_align_code, noisy_attention_loss, common_token_loss

    def attention_alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, labelled_mapped_tokens, common_mapped_tokens, args, return_vec=False, return_scores=False):
        bs = code_inputs.shape[0]
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        # 获取注意力权重
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        # all_code_hidden_states = code_outputs.hidden_states

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[:, :, 0, :].mean(dim=1)
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[:, :, 0, :].mean(dim=1)

        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
        
        loss_align_code = 0

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]

        lcs_pairs = sample_align
        selected_align_outputs_1 = []
        selected_align_outputs_2 = []
        n = 0
        for pair in lcs_pairs:
            # pair[0] 是 comment 的区间，pair[1] 是 code 的区间
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            comment_embeddings = [align_outputs_1[idx+1] for idx in comment_indices]
            comment_mean = torch.mean(torch.stack(comment_embeddings), dim=0)

            # 计算 code 区间的平均值
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))    
            code_embeddings = [align_outputs_2[idx+1] for idx in code_indices]
            code_mean = torch.mean(torch.stack(code_embeddings), dim=0)

            if code_embeddings:  # 检查 code_embeddings 是否为空
                code_mean = torch.mean(torch.stack(code_embeddings), dim=0)
                selected_align_outputs_1.append(comment_mean)
                selected_align_outputs_2.append(code_mean)
            n += 1
        # 将列表转换为张量
        selected_align_outputs_1 = torch.stack(selected_align_outputs_1)
        selected_align_outputs_2 = torch.stack(selected_align_outputs_2)

        align_loss = NTXentLoss(args, selected_align_outputs_1.shape[0],temperature=3.0)

        loss_align_code += align_loss(selected_align_outputs_1, selected_align_outputs_2)
        loss_align_code = (loss_align_code/len(lcs_pairs))*2

        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 步骤 1: 展开嵌套列表
        # flat_labelled_mapped_tokens = [item for sublist in labelled_mapped_tokens for item in sublist]
        # 去重处理
        # alignment_code_indices = list(set(flat_labelled_mapped_tokens))
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        # 给 alignment_code_indices 中的每一个值 +1
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]
        # print(alignment_code_indices, alignment_nl_indices)

        # Step 1: 计算 attention loss
        certain_code_cls_attention = code_cls_attention[local_index]
        certain_nl_cls_attention = nl_cls_attention[local_index]
        code_attention_loss = self.calculate_attention_loss(certain_code_cls_attention, alignment_code_indices)
        nl_attention_loss = self.calculate_attention_loss(certain_nl_cls_attention, alignment_nl_indices)

        # Step 2: 合并代码和自然语言的 attention loss
        noisy_attention_loss = (code_attention_loss + nl_attention_loss) / 2

        if len(common_mapped_tokens) > 0:
            # common_token_loss = self.alignment_loss(align_outputs_2, common_mapped_tokens, args.code_length)
            # 创建一个目标张量，初始值为零
            target = torch.zeros_like(align_outputs_2)
            seq_length = args.code_length
            # 将 common_mapped_tokens 中不在的索引对应的 target 位置设置为 align_outputs_2 中的值
            for idx in range(seq_length):
                if idx not in common_mapped_tokens:
                    target[idx] = align_outputs_2[idx]
            
            # 计算损失
            common_token_loss = F.mse_loss(align_outputs_2, target)
            if noisy_attention_loss > 0:
                total_loss = loss + loss_align_code + noisy_attention_loss + common_token_loss
        else:
            if noisy_attention_loss > 0:
                common_token_loss = 0
                total_loss = loss + loss_align_code + noisy_attention_loss
            else:
                print(noisy_attention_loss)
                # print(positive_nl_attention)
                # print(negative_nl_attention)
        # total_loss = loss + loss_align_code + noisy_attention_loss

        return loss, total_loss, code_vec, nl_vec, loss_align_code, noisy_attention_loss, common_token_loss

    def calculate_attention_loss(self, attention, alignment_indices):
        """
        计算 attention loss，目标是：
        - 对齐的 token 的 attention 越大越好（接近 1）
        - 非对齐的 token 的 attention 越小越好（接近 0）
        """        
        # 创建目标 attention 分布
        target_attention = torch.zeros_like(attention)
        
        # 对对齐的 token，将目标 attention 设置为 1
        for idx in alignment_indices:  # 遍历对齐的 token 索引
            target_attention[idx] = 1.0  # 期望这些 token 的 attention 是 1
        
        # 使用 MSE loss 来使对齐的 token attention 趋向 1，非对齐的 token attention 趋向 0
        attention_loss = F.mse_loss(attention, target_attention)

        return attention_loss
    
    def alignment_loss(align_outputs_2, common_mapped_tokens, seq_length, alpha=1.0):
        """
        计算对齐损失，使得对于 common_mapped_tokens 中的索引, align_outputs_2 中的隐藏状态尽可能接近零。
        
        Args:
        - align_outputs_2 (torch.Tensor): 模型输出的最后隐藏状态，形状为 (seq_length, hidden_state_dim)。
        - common_mapped_tokens (list of int): 需要对齐的索引。
        - alpha (float): 损失的权重因子。
        
        Returns:
        - loss (torch.Tensor): 计算得到的损失值。
        """ 
        # 创建一个目标张量，初始值为零
        target = torch.zeros_like(align_outputs_2)
        
        # 将 common_mapped_tokens 中不在的索引对应的 target 位置设置为 align_outputs_2 中的值
        for idx in range(seq_length):
            if idx not in common_mapped_tokens:
                target[idx] = align_outputs_2[idx]
        
        # 计算损失
        loss = F.mse_loss(align_outputs_2, target)
        
        return alpha * loss

    def calculate_noisy_attention_loss(self, noisy_code_attention, noisy_nl_attention):
        # 目标是最小化 noisy token 的 attention
        return (torch.sum(noisy_code_attention) + torch.sum(noisy_nl_attention))/2
    
    def focal_attention_loss(self, positive_attention, negative_attention, gamma=2.0):
        # Compute the focal loss for positive and negative attention scores
        positive_loss = (1 - positive_attention) ** gamma * positive_attention.log()
        negative_loss = negative_attention ** gamma * (1 - negative_attention).log()
        return -(positive_loss - negative_loss).mean()
    
    def dynamic_weighting_attention_loss(self, positive_attention, negative_attention):
        pos_weight = (1 - positive_attention).mean()
        neg_weight = negative_attention.mean()
        loss = pos_weight * positive_attention.log() + neg_weight * (1 - negative_attention).log()
        return -loss.mean()
    
    def regularization_attention_loss(self, positive_attention, negative_attention, alpha=0.1):
        # L1正则化
        positive_reg_loss = alpha * positive_attention.norm(1)
        negative_reg_loss = (1 - alpha) * negative_attention.norm(1)
        return positive_reg_loss + negative_reg_loss
    

    def new_alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens):
        bs = code_inputs.shape[0]
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]

        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        retrieval_loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]

        lcs_pairs = sample_align
        loss_align_code = self.build_contrastive_pairs(align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens)
        # print("loss_align_code",loss_align_code)
        
        # attention loss part
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[local_index, :, 0, :].mean(dim=0)
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[local_index, :, 0, :].mean(dim=0)

        # 获取对齐的 indices
        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 去重处理并 +1
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]

        # 计算 attention loss
        epsilon = 1e-8
        # 直接使用 mask 对 positive 和 negative 的 attention 进行区分
        code_mask = torch.zeros_like(code_cls_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(nl_cls_attention, dtype=torch.bool)
        
        # 对于正样本位置，设置 mask
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        # 计算正样本和负样本的 attention loss
        positive_code_attention = code_cls_attention[code_mask]
        negative_code_attention = code_cls_attention[~code_mask]
        positive_nl_attention = nl_cls_attention[nl_mask]
        negative_nl_attention = nl_cls_attention[~nl_mask]

        # 计算 attention loss
        attention_loss = (torch.sum(-torch.log(positive_code_attention + epsilon)) +
                          torch.sum(-torch.log(1.0 - negative_code_attention + epsilon)) +
                          torch.sum(-torch.log(positive_nl_attention + epsilon)) +
                          torch.sum(-torch.log(1.0 - negative_nl_attention + epsilon)))
        attention_loss = attention_loss / (len(alignment_code_indices) + len(alignment_nl_indices))
        # print("attention_loss",attention_loss)
        # print("nl_cls_attention",nl_cls_attention)

        return retrieval_loss, code_vec, nl_vec, loss_align_code, attention_loss

      
    def build_contrastive_pairs(self, align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens, num_negative=19):
        loss_align_code = 0
        num_pair = 0

        # 遍历每一个对齐的 lcs_pairs
        for pair in lcs_pairs:
            # 获取 comment 和 code 的 indices
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))

            # 遍历所有 comment indices，构建正负样本对
            # if num_pair == 0:
            for c_idx in comment_indices:
                comment_embedding = align_outputs_1[c_idx + 1]
                # 计算正样本相似度
                for code_idx in code_indices:
                    # if num_pair == 0:
                    row_similarities = []
                    row_labels = []
                    num_pair += 1
                    code_embedding = align_outputs_2[code_idx + 1]
                    pos_similarity = F.cosine_similarity(comment_embedding, code_embedding, dim=0)
                    row_similarities.append(pos_similarity.unsqueeze(0))
                    row_labels.append(torch.tensor(1, device=comment_embedding.device))  # 正样本标签为 1
                    # 构建负样本相似度（使用所有非 code_indices 的负样本）
                    negative_indices = set(range(total_code_tokens)) - set(code_indices)
                    for neg_idx in negative_indices:
                        negative_embedding = align_outputs_2[neg_idx + 1]
                        neg_similarity = F.cosine_similarity(comment_embedding, negative_embedding, dim=0)
                        row_similarities.append(neg_similarity.unsqueeze(0))
                        row_labels.append(torch.tensor(0, device=comment_embedding.device))  # 负样本标签为 0
                    temperature = 0.2
                    similarities = torch.stack(row_similarities)
                    # if num_pair == 1:
                    #     print(similarities)
                    similarities = similarities / temperature
                    exp_similarities = torch.exp(similarities)
                    # print(exp_similarities)
                    pos_similarity = exp_similarities[0]
                    neg_sum = exp_similarities[1:].sum()
                    # print(pos_similarity)
                    nt_xent_loss = -torch.log(pos_similarity / (pos_similarity + neg_sum))
                    nt_xent_loss = nt_xent_loss.mean()  # 确保是标量
                    loss_align_code += nt_xent_loss

        return loss_align_code / num_pair

    def contrastive_loss(self, similarities, labels, temperature=0.07):
        # 对比学习损失的实现
        similarities = similarities / temperature
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for i in range(similarities.size(0)):
            loss = criterion(similarities[i].unsqueeze(0), labels[i].unsqueeze(0))
            total_loss += loss
        loss = total_loss / similarities.size(0)
        return loss
 
