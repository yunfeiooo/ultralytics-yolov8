# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        Args:
            pred_dist, [b, num_anchors, 4*reg_max], pred distribution. 4 指的是 ltrb 四个变量. reg_max 指的是 DFL 中的离散数量
            pred_bboxes, (b, num_anchors, 4), xyxy 形式，预测的 bbox（根据 pred_dist 得到的, 0-1）
            anchor_points, [num_anchors, 2], 每个 grid 的中心点位置
            target_bboxes, [b, num_anchors, 4], 每个 anchor 对应的 xyxy target, 0-1
            target_scores, [b, num_anchors, nc], 每个 anchor 对应的分类信息, pos gt class 对应的 norm_align_metric 数值
            target_scores_sum, norm 项, target_scores.sum()
            fg_mask, [b, num_anchors], 每个 anchor 是否被分配上 target（正样本）

        """
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max) # 将标签转成 dist 形式
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight # self.reg_max + 1 == Detect.reg_max
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # 可以认为学习一个插值
        # Return sum of left and right DFL losses
        tl = target.long()  # target left, 3.4 -> 3, torch.floor
        tr = tl + 1  # target right, 3.4 -> 3 -> 4, torch.ceil
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)
