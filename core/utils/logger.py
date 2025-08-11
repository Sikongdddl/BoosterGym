# utils/tb_logger.py
import os
import time
from typing import Dict, Any, Optional, Sequence, Union

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

ArrayLike = Union[np.ndarray, torch.Tensor]

class TBLogger:
    """
    Thin wrapper for TensorBoard SummaryWriter.

    Features:
      - step 管理（global_step 可自增）
      - 标量/直方图/图片/文本/视频
      - 批量写入 scalars
      - 简单的节流 flush
    """
    def __init__(
        self,
        logdir: str,
        run_name: Optional[str] = None,
        flush_secs: int = 10,
        max_queue: int = 100,
        auto_timestamp: bool = True,
    ):
        if auto_timestamp:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            run_name = run_name or stamp
        self.log_path = os.path.join(logdir, run_name or "")
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path, flush_secs=flush_secs, max_queue=max_queue)
        self._step = 0

    # -------- step 管理 --------
    @property
    def step(self) -> int:
        return self._step

    def set_step(self, step: int):
        self._step = int(step)

    def inc_step(self, n: int = 1):
        self._step += int(n)

    # -------- 核心 API --------
    def add_scalar(self, tag: str, value: float, step: Optional[int] = None):
        self.writer.add_scalar(tag, float(value), global_step=self._g(step))

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        self.writer.add_scalars(main_tag, {k: float(v) for k, v in tag_scalar_dict.items()}, global_step=self._g(step))

    def add_histogram(self, tag: str, values: ArrayLike, bins: str = "auto", step: Optional[int] = None):
        values = self._to_np(values)
        self.writer.add_histogram(tag, values, global_step=self._g(step), bins=bins)

    def add_image(self, tag: str, img: ArrayLike, step: Optional[int] = None, dataformats: str = "CHW"):
        img = self._to_np(img)
        self.writer.add_image(tag, img, global_step=self._g(step), dataformats=dataformats)

    def add_text(self, tag: str, text: str, step: Optional[int] = None):
        self.writer.add_text(tag, text, global_step=self._g(step))

    def add_video(self, tag: str, frames: ArrayLike, fps: int = 20, step: Optional[int] = None):
        """
        frames: shape [N, T, C, H, W] or [T, C, H, W]
        会自动补齐 batch 维
        """
        arr = self._to_np(frames)
        if arr.ndim == 4:  # [T, C, H, W]
            arr = arr[None, ...]  # -> [1, T, C, H, W]
        assert arr.ndim == 5, "video must be [N, T, C, H, W] or [T, C, H, W]"
        self.writer.add_video(tag, arr, global_step=self._g(step), fps=fps)

    def add_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        # 注意：TensorBoard 会单独显示 hparams
        self.writer.add_hparams(hparam_dict, {k: float(v) for k, v in metric_dict.items()})

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    # -------- 内部工具 --------
    def _g(self, step: Optional[int]) -> int:
        return self._step if step is None else int(step)

    @staticmethod
    def _to_np(x: ArrayLike) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)
