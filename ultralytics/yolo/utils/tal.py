# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checks import check_version
from .metrics import bbox_iou

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape (num_total_anchors, 2)
        gt_bboxes (Tensor): shape(b, max_num_obj, 4)
    Return:
        (Tensor): shape(b, max_num_obj, num_total_anchors)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    # rb - xy_centers[None] -> 只预测正数
    # [b*max_num_obj, num_total_anchors, 4] -> [b, max_num_obj, num_total_anchors, 4]
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps) # [b, max_num_obj, num_total_anchors],


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes, 只保留 IoU 最大的 gt bbox
        # 找到匹配多个 gt 的 anchor, 做后续处理
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w) # 每个 anchor 对应最大 iou 的 gt bbox 序号
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes), 转成 one-hot 形式
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w), mask 形式

        # 替换 mask 信息，移除多余的 gt bbox
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)，更新 mask
        fg_mask = mask_pos.sum(-2) # 当前 anchor 是否含有正样本的 mask (现在每个 anchor 只匹配一个 gt bbox，sum 之后还是0/1)
    # find each grid serve which gt(index)， 每个 anchor 匹配的 gt bbox 序号
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1), 当前 gt bbox 是否为正（打包 batch 的时候有填充空标签)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        #
        # mask_pos, (b, max_num_obj, num_anchors), [x, i, j] 表示第 i 个 gt bbox 和第 j 个 anchor 是否匹配
        # align_metric, (b, max_num_obj, num_anchors), [x, i, j] 表示第 i 个 gt bbox  和第 j 个 anchor 之间的距离度量
        # overlaps, (b, max_num_obj, num_anchors), [x, i, j] 表示第 i 个 gt bbox  和第 j 个 anchor 之间的 IoU
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        # 如果一个 anchor 被分配给多个 gt bbox，只分配给 IOU 最大的那个 gt
        # target_gt_idx: (b, num_anchors), 每个 anchor 匹配的 gt bbox 序号
        # fg_mask: (b, num_anchors), 当前 anchor 是否含有正样本的 mask
        # mask_pos: (b, max_num_obj, num_anchors), [x, i, j] 表示第 i 个 gt bbox 和第 j 个 anchor 是否匹配
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        # target_labels: [b, num_anchors], 每个 anchor 对应的类别 index target
        # target_bboxes: [b, num_anchors, 4], 每个 anchor 对应的 xyxy target
        # target_scores: [b, num_anchors, nc], 不是 score 是一个 mask
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize
        # https://github.com/fcjian/TOOD/blob/c5388000afacf4a0a60b89e380fabae8935ea751/mmdet/models/dense_heads/tood_head.py#L684
        align_metric *= mask_pos
        overlaps *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj, 1; 每个 gt bbox 的最大匹配系数
        pos_overlaps = overlaps.amax(axis=-1, keepdim=True)  # b, max_num_obj, 1; 每个 gt bbox 的最大匹配 IOU
        # ---> 得到 score 的本体, [b, max_num_obj, num_anchors] -> [b, num_anchors] -> [b, num_anchors, 1]
        norm_align_metric = (pos_overlaps * (align_metric / (pos_align_metrics + self.eps))).amax(-2).unsqueeze(-1)
        # 仅保留 pos gt class 对应的 norm_align_metric
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # 首先计算 IOU(pred bbox, gt bbox) 和 align metric(pred score, IOU.), 基于这个指标进行正负样本分配
        # 不同于 anchor-based 的典型方法，这里是【动态分配】正负样本的！
        # get anchor_align metric, (b, max_num_obj, num_anchors)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)

        # mask_in_gts: (b, max_num_obj, num_anchors)
        # mask_in_gts[i, j, k], 第 i 张图片中, 第 j 个 gt bbox 和第 k 个 anchor 是否匹配
        # mask_in_gts[:, :, k].sum() >= 1，表明第 k 个 anchor 是正 anchor
        #   - mask_in_gts[:, :, k].sum() > 1, 表明这个 anchor 匹配多个 gt bbox, 只保留最大 IOU 的 gt bbox
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes) #TODO???
        # mask_in_gts[:, j, :].sum() >= 1，表明第 j 个 gt bbox 匹配多个 anchor，需要执行 select_topk_candidates 选择 topk 个 anchor
        # select_topk_candidates 根据 align metrics 筛选 anchors（task-aligned anchors）
        # get topk_metric mask, (b, max_num_obj, num_anchors)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, (b, max_num_obj, num_anchors)
        # 1. topk 控制数量
        # 2. anchor(xy) 位于 bbox 内部
        # 3. 是有 gt bbox (不是填充的空标签)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, num_total_anchors
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)        # [b, max_num_obj, num_total_anchors]
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)  # [b, max_num_obj, num_total_anchors]
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """根据 align_metric（bbox 和 anchor 之间的匹配度量，基于 IoU(pred, gt) 和 pred score 计算得到) 为每个 gt 找到 top-K
        个匹配的 anchors
        Args:
            metrics: (b, max_num_obj, num_anchors).
            topk_mask: (b, max_num_obj, topk) or None
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk), 每个 gt 找 topK 个 anchors, 每个 gt bbox 对应的 anchors 信息
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk), 忽略 gt 背景
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, num_anchors)
            fg_mask: (b, h*w)
        """

        # assigned target labels, (b, 1)
        #
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None] * self.n_max_boxes # offset
        target_gt_idx = target_gt_idx + batch_ind  # (b, h*w), 加上 batch index 信息, 方便 flatten
        #
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)，每个 anchor 对应的类别

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0) # one-hot 形式

        return target_labels, target_bboxes, target_scores


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    distance(pred_dist), [N, num_anchors, 4]
    anchor_points, [num_anchors, 2]
    """
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
