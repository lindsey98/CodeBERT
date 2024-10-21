import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import numpy as np


class NTXentLoss(_Loss):
    def __init__(self, args, batch_size, temperature=1.0, use_cosine_similarity=True, hidden_size=768, device="cuda"):
        super(NTXentLoss, self).__init__(args)
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion_simclr = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):

        self.batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[
            self.mask_samples_from_same_repr[:(self.batch_size * 2), :(self.batch_size * 2)]]

        negatives = negatives.view(2 * self.batch_size, -1)[:, :20]

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion_simclr(logits, labels)

        return loss / (2 * self.batch_size)
    # def forward(self, zis, zjs, yis, yjs):
    #     # refine negative
    #     self.batch_size = zis.shape[0]
    #     representations = torch.cat([zjs, zis], dim=0)

    #     similarity_matrix = self.similarity_function(representations, representations)

    #     # filter out the scores from the positive samples
    #     l_pos = torch.diag(similarity_matrix, self.batch_size)
    #     r_pos = torch.diag(similarity_matrix, -self.batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
    #     # use 2 * self.batch_size, -1 negative pairs
    #     neg_representations = torch.cat([yjs, yis], dim=0)

    #     neg_similarity_matrix = self.similarity_function(neg_representations, neg_representations)

    #     negatives = neg_similarity_matrix[
    #         self.mask_samples_from_same_repr[:(self.batch_size * 2), :(self.batch_size * 2)]]

    #     negatives = negatives.view(2 * self.batch_size, -1)

    #     logits = torch.cat((positives, negatives), dim=1)
    #     logits /= self.temperature

    #     labels = torch.zeros(2 * self.batch_size).to(self.device).long()
    #     loss = self.criterion_simclr(logits, labels)

    #     return loss / (2 * self.batch_size)
    
    # def forward(self, zis, zjs, yis, yjs):
    #     # refine negative
    #     self.batch_size = zis.shape[0]
    #     representations = torch.cat([zjs, zis], dim=0)

    #     similarity_matrix = self.similarity_function(representations, representations)

    #     # filter out the scores from the positive samples
    #     l_pos = torch.diag(similarity_matrix, self.batch_size)
    #     r_pos = torch.diag(similarity_matrix, -self.batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

    #     # use 5(or other configuration value) pairs
    #     # 初始化l_neg存储相似度结果
    #     l_neg = torch.empty(yis.shape[0], dtype=torch.float32, device=self.device)
    #     for i in range(yis.shape[0]):
    #         similarity = self.similarity_function(yis[i:i+1], yjs[i:i+1])
    #         l_neg[i] = similarity
    #     negatives = l_neg.view(positives.shape[0], 5)

    #     logits = torch.cat((positives, negatives), dim=1)
    #     # print(zis.shape, positives.shape, l_neg.shape, negatives.shape, logits.shape)
    #     # print(logits, logits * (1 / self.temperature))
    #     logits /= self.temperature

    #     labels = torch.zeros(2 * self.batch_size).to(self.device).long()
    #     loss = self.criterion_simclr(logits, labels)
    #     # print(loss / (2 * self.batch_size))

    #     return loss / (2 * self.batch_size)
    
    def rearrange_l_neg(self, l_neg, num_rows):
        num_cols = 5  # 每行的列数

        # 初始化一个大小为 [num_rows, num_cols] 的张量
        rearranged_l_neg = torch.empty((num_rows, num_cols), dtype=l_neg.dtype, device=l_neg.device)

        # 前半部分重排列
        for i in range(num_rows // 2):
            rearranged_l_neg[i] = l_neg[i * 2*num_cols : i * 2*num_cols + num_cols]

        # 后半部分重排列
        for i in range(num_rows // 2, num_rows):
            rearranged_l_neg[i] = l_neg[(i - num_rows // 2) * 2*num_cols + num_cols : (i - num_rows // 2) * 2*num_cols + 2*num_cols]

        return rearranged_l_neg


class AlignLoss(_Loss):
    def __init__(self, args, batch_size, temperature=1.0, device="cuda"):
        super(AlignLoss, self).__init__(args)
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self._cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, zis, zjs):
        self.batch_size = zis.shape[0]
        a = self._cosine_similarity(zis, zjs)

        return torch.mean(1 - a)