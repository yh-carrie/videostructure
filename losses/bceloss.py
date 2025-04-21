import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None


    def contrastive_loss(self, logits, labels, temperature=0.3):
        # 多标签分类的计算
        # 计算欧氏距离
        dists = logits[-1]  # 得到的是两两之间的欧氏距离的矩阵
        con_loss = 0.0
        con_loss_num = 0

        # 求出相互之间标签的点乘并计算相似度矩阵 C_juzhen
        C_juzhen = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().sum(dim=2)
        # C_juzhen = C_juzhen-C_juzhen.min()

        # 遍历每个元素计算 L_con
        for i in range(dists.size(0)):
            # 计算分母：对所有 k 的求和，排除自己
            exp_terms = torch.exp(-dists[i] / temperature)

            # 创建掩码，排除对角线元素
            mask = torch.ones_like(C_juzhen[i], dtype=torch.bool)
            mask[i] = False  # 将对角线元素设为 False
            C_row_sums = torch.sum(C_juzhen[i][mask])  # 使用掩码排除对角线元素

            # 计算分母，仅包括掩码为 True 的元素
            denominator = torch.sum(exp_terms[mask])

            # 计算分子
            if C_row_sums.item() == 0 or denominator.item() == 0:
                continue
            con_loss_num += 1
            for j in range(dists.size(1)):
                if i != j:  # 排除对角线元素
                    B_coe = C_juzhen[i][j] / C_row_sums
                    numerator =  torch.exp(-dists[i][j] / temperature)
                    L_con_ij = (-B_coe) * torch.log(numerator / denominator)
                    con_loss += L_con_ij
        if con_loss_num == 0:
            return torch.tensor(0.0).cuda()
        else:
            return con_loss / con_loss_num

    def bce_loss(self, logits, targets):
        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return loss_m

    def ce_loss(self, logits, targets):
        loss_m = F.cross_entropy(logits, targets, reduction='none')
        return loss_m

    def forward(self, logits, targets):

        # 前面五行是为了对比学习加上去的
        # feat_sim = logits[1] # 得到了欧式距离的矩阵
        # targ0 = targets.unsqueeze(0)
        # targ1 = targets.unsqueeze(1)
        # targ_sim = torch.cosine_similarity(targ0, targ1, 2)
        # loss_sim = abs(feat_sim - targ_sim).mean()
        # 因为对比学习的特征是从分类器之前获取的，所以增加了分类器之后，并不影响整体的损失的计算
        # contrastive_loss = self.contrastive_loss(logits, targets)
        # contrastive_loss = 0.0
        # print('对比学习损失是：', contrastive_loss)

        #  单个分类器，多标签分类的计算
        logits = logits[0]
        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)
        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)  # (24,26)
            loss_m = (loss_m * sample_weight.cuda())
        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss_bce = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()
        # loss = loss_m.mean() if self.size_sum else loss_m.sum() # 大胆
        # print('BCE损失是：', loss_bce)

        # 为了多个分类器加上去的
        # ce_loss_list = []
        # target_split = [[19, 20, 21], [23, 24, 25]]
        # target_loss = ['CE', 'CE']
        # for i in range(len(target_split)):
        #     # 针对每一个分类器都需要计算一次损失
        #     # target构造方式1：改为和logits.shape适配的大小
        #     targets_cur = targets[:, target_split[i]]
        #     logits_cur = logits[:, target_split[i]]
        #     # if logits[i].size(1) > targets_cur.size(1):
        #     #     num_to_add = logits[i].size(1) - targets_cur.size(1)
        #     #     # 为每一行计算 fill_value
        #     #     fill_values = torch.where(targets_cur.sum(dim=1) == 0, 1, 0)  # 如果某行和为0，则填1，否则填0
        #     #
        #     #     # 将 fill_values 扩展为 (targets_cur.size(0), num_to_add) 的形状
        #     #     add_values = fill_values.unsqueeze(1).expand(-1, num_to_add)
        #     #
        #     #     # 将新值拼接到 targets_cur 的末尾
        #     #     targets_cur = torch.cat((targets_cur, add_values), dim=1)
        #
        #     # # 不同分类器对应的是多标签分类，用BCE_LOSS
        #     # if target_loss[i]=='BCE':
        #     #     loss_m = self.bce_loss(logits[i], targets_cur)
        #     #     targets_mask = torch.where(targets_cur.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        #     #     if self.sample_weight is not None:
        #     #         sample_weight = ratio2weight(targets_mask, self.sample_weight[target_split[i]])  # (24,26)
        #     #         loss_m = (loss_m * sample_weight.cuda())
        #     #     loss_bce = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()
        #
        #     if target_loss[i] == 'CE':
        #         # 这个计算方式，是绝对不会允许预测结果中出现两个1的
        #         indices_tensor = (targets_cur == 1).nonzero(as_tuple=True)[1]
        #         loss_cur = self.ce_loss(logits_cur, indices_tensor)
        #         sample_weight = torch.from_numpy(self.sample_weight).cuda()
        #         weights = sample_weight[target_split[i]][indices_tensor]  # 获取并索引权重
        #         loss_cur = loss_cur * torch.exp(1-weights)
        #         ce_loss_list.append(torch.mean(loss_cur))
        #         # 取平均值
        # loss_ce = sum(ce_loss_list)
        # print('ce损失是：', loss_ce)
        # 新增的损失计算是LOSS_CE和contrastive_loss，可以通过控制权重将他们的占比降为0
        return [loss_bce], [loss_m]
        # return [loss_bce, contrastive_loss], [loss_m]

        # return [loss_bce, loss_ce, contrastive_loss], [loss_bce]



    # # 示例使用
    # batch_size = 24
    # feature_dim = 32768
    # features = torch.randn(batch_size, feature_dim)  # 假设的特征张量
    # labels = torch.randint(0, 10, (batch_size,))  # 假设的标签张量，10个类
    #
    # loss = contrastive_loss(features, labels)
    # print(f'Contrastive Loss: {loss.item()}')