import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(r),
            np.asarray(s2, dtype=np.float32),
            bool(d),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)

    def debug_print(self, n=5):
        """随机打印 n 条样本，帮助你直观查看 buffer 结构。"""
        if len(self.buf) == 0:
            print("\n[ReplayBuffer] empty.")
            return

        n = min(n, len(self.buf))
        sample = random.sample(self.buf, n)

        print(f"\n=== ReplayBuffer Debug ({n} samples out of {len(self.buf)}) ===")
        for i, (s, a, r, s2, d) in enumerate(sample):
            print(f"[{i}]")
            print(f"s.shape:  {np.shape(s)}")
            print(f"s[:5]:    {np.asarray(s).flatten()[:5]}")   # 只看前几个数，防止太长
            print(f"a:        {a}")
            print(f"r:        {r:.4f}")
            print(f"s2.shape: {np.shape(s2)}")
            print(f"s2[:5]:   {np.asarray(s2).flatten()[:5]}")
            print(f"done:     {d}")
            print("-" * 40)
