"""轴对齐 3D bbox 的 IoU / overlap 计算，供扩散训练的 iou loss 使用。

迁移自 BuildingBlock/scene_synthesis/networks/loss.py。
来源： mmdetection3d 的 axis_aligned iou3d 实现。
"""
import torch


def axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """计算两组轴对齐 3D bbox 的重叠。

    Args:
        bboxes1 (Tensor): (B, m, 6)，格式 <x1, y1, z1, x2, y2, z2>。
        bboxes2 (Tensor): (B, n, 6)。
        mode (str): "iou" / "giou" / "overlap_ratio"。
        is_aligned (bool): True 时 m == n，逐对计算。
    Returns:
        Tensor: is_aligned=False 时形如 (B, m, n)。
    """
    assert mode in ["iou", "giou", "overlap_ratio"], f"Unsupported mode {mode}"
    assert bboxes1.size(-1) == 6 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 6 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (
        (bboxes1[..., 3] - bboxes1[..., 0])
        * (bboxes1[..., 4] - bboxes1[..., 1])
        * (bboxes1[..., 5] - bboxes1[..., 2])
    )
    area2 = (
        (bboxes2[..., 3] - bboxes2[..., 0])
        * (bboxes2[..., 4] - bboxes2[..., 1])
        * (bboxes2[..., 5] - bboxes2[..., 2])
    )

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]
        union = area1 + area2 - overlap
        min_area = torch.min(area1, area2)
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])
        rb = torch.min(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]
        union = area1[..., None] + area2[..., None, :] - overlap
        min_area = torch.min(area1[..., None], area2[..., None, :])
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode == "iou":
        return ious
    if mode == "overlap_ratio":
        return overlap / torch.max(min_area, min_area.new_tensor([eps]))
    # giou
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    return ious - (enclose_area - union) / enclose_area
