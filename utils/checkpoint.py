"""checkpoint 保存与恢复。"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import torch


_MODEL_RE = re.compile(r"^model_(\d{5,})$")


def save_checkpoint(model, optimizer, epoch: int, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(directory, f"model_{epoch:05d}"))
    torch.save(optimizer.state_dict(), os.path.join(directory, f"opt_{epoch:05d}"))


def _find_latest(directory: str) -> Optional[int]:
    if not os.path.isdir(directory):
        return None
    ids = []
    for fn in os.listdir(directory):
        m = _MODEL_RE.match(fn)
        if m:
            ids.append(int(m.group(1)))
    return max(ids) if ids else None


def load_latest_checkpoint(
    model,
    optimizer,
    directory: str,
    device: str = "cpu",
    strict: bool = True,
) -> Tuple[int, bool]:
    """如果存在历史 checkpoint 则加载，返回 (下一个 epoch, 是否加载成功)。"""
    max_id = _find_latest(directory)
    if max_id is None:
        return 0, False

    model_path = os.path.join(directory, f"model_{max_id:05d}")
    opt_path = os.path.join(directory, f"opt_{max_id:05d}")

    print(f"[ckpt] loading model from {model_path}")
    _load_state_dict(model, model_path, device=device, strict=strict)

    if os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        except Exception as e:
            print(f"[ckpt] load optimizer failed, skip: {e}")

    return max_id + 1, True


def load_weights(model, weight_path: str, device: str = "cpu", strict: bool = True) -> None:
    """只加载权重，不关心 optimizer。"""
    print(f"[ckpt] loading weights from {weight_path}")
    _load_state_dict(model, weight_path, device=device, strict=strict)


def _load_state_dict(model, weight_path: str, device: str = "cpu", strict: bool = True) -> None:
    state = torch.load(weight_path, map_location=device)
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[ckpt] missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[ckpt] unexpected keys: {incompatible.unexpected_keys}")
