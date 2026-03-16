import random
from collections import deque
import numpy as np
import os
import time

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)
        self.capacity = capacity
        self.total_pushes = 0  # 全局计数器，记录总共 push 了多少 transition（不受 capacity 限制）

    def push(self, s, a, r, s2, d, note="", difficulty=None, success=False):
        transition = (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(r),
            np.asarray(s2, dtype=np.float32),
            bool(d),
            str(note),
            difficulty,
            bool(success),
        )
        self.buf.append(transition)

        tid = self.total_pushes
        self.total_pushes += 1
        return tid  # 全局 ID

    def sample(self, batch_size, r_min=None, r_max=None, sigma=0.2, epsilon=0.1, success_bonus=1.2, hard_focus=0.0):
        use_weighted = (r_min is not None) and (r_max is not None)
        hard_focus = float(hard_focus)

        if use_weighted:
            weights = []
            for _, _, _, _, _, _, difficulty, success in self.buf:
                if difficulty is None:
                    weight = 1.0
                else:
                    if r_min <= difficulty <= r_max:
                        weight = 1.0
                    else:
                        dist = min(abs(difficulty - r_min), abs(difficulty - r_max))
                        weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                    if hard_focus > 0.0:
                        span = max(1e-6, float(r_max) - float(r_min))
                        hardness = np.clip((float(difficulty) - float(r_min)) / span, 0.0, 1.0)
                        weight *= np.exp(hard_focus * hardness)
                if success:
                    weight *= float(success_bonus)
                weights.append(weight)

            weights = np.array(weights, dtype=np.float64)
            wsum = float(weights.sum())
            if wsum <= 0.0 or not np.isfinite(wsum):
                weights = np.ones(len(self.buf), dtype=np.float64) / len(self.buf)
            else:
                weights /= wsum

            indices = []
            for _ in range(batch_size):
                if np.random.rand() < epsilon:
                    indices.append(np.random.randint(len(self.buf)))
                else:
                    indices.append(np.random.choice(len(self.buf), p=weights))
        else:
            indices = random.sample(range(len(self.buf)), batch_size)

        batch = [self.buf[i] for i in indices]
        s, a, r, s2, d, notes, difficulties, successes = map(np.array, zip(*batch))
        return s, a, r, s2, d, notes, difficulties, successes

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
        for i, (s, a, r, s2, d, note, difficulty, success) in enumerate(sample):
            print(f"[{i}]")
            print(f"s.shape:  {np.shape(s)}")
            print(f"s[:5]:    {np.asarray(s).flatten()[:5]}")   # 只看前几个数，防止太长
            print(f"a:        {a}")
            print(f"r:        {r:.4f}")
            print(f"s2.shape: {np.shape(s2)}")
            print(f"s2[:5]:   {np.asarray(s2).flatten()[:5]}")
            print(f"done:     {d}")
            print(f"note:     {note}")
            print(f"difficulty:{difficulty}")
            print(f"success:  {success}")
            print("-" * 40)
    
    def save_to_disk(self, save_dir="logs/replay", filename=None):
        """
        把当前 buffer 中的所有数据完整保存到磁盘（npz 格式）
        适合你后续做 reward / HER / curriculum 的离线 debug。

        保存字段：
            s, a, r, s2, d, notes, difficulties, successes
        """
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            filename = f"replay_{ts}_N{len(self.buf)}.npz"

        path = os.path.join(save_dir, filename)

        if len(self.buf) == 0:
            print("[ReplayBuffer] buffer empty, nothing saved.")
            return None

        # 解包 buffer
        s_list, a_list, r_list, s2_list, d_list, notes_list, difficulties_list, successes_list = zip(*self.buf)

        s_arr  = np.stack(s_list, axis=0)
        a_arr  = np.stack(a_list, axis=0)
        r_arr  = np.asarray(r_list, dtype=np.float32)
        s2_arr = np.stack(s2_list, axis=0)
        d_arr  = np.asarray(d_list, dtype=np.bool_)
        notes_arr = np.asarray(notes_list, dtype=object)
        difficulties_arr = np.asarray(difficulties_list, dtype=np.float32)
        successes_arr = np.asarray(successes_list, dtype=np.bool_)

        np.savez_compressed(
            path,
            s=s_arr,
            a=a_arr,
            r=r_arr,
            s2=s2_arr,
            d=d_arr,
            notes=notes_arr,
            difficulties=difficulties_arr,
            successes=successes_arr,
            size=len(self.buf),
        )

        print(f"[ReplayBuffer] saved to: {path}")
        print(f"  s:  {s_arr.shape}")
        print(f"  a:  {a_arr.shape}")
        print(f"  r:  {r_arr.shape}")
        print(f"  s2: {s2_arr.shape}")
        print(f"  d:  {d_arr.shape}")
        print(f"  notes: {notes_arr.shape}")
        print(f"  difficulties: {difficulties_arr.shape}")
        print(f"  successes: {successes_arr.shape}")

        return path
    
    def _global_to_local_idx(self, tid):
        """
        把全局 transition id 映射到当前 deque 下标。
        如果已经因为 maxlen 被丢弃，则返回 None。
        """
        n = self.total_pushes
        L = len(self.buf)

        if L == 0:
            return None

        # 还没发生过溢出：0..(n-1) 与 buf[0..L-1] 对应
        if n <= self.capacity:
            return tid if 0 <= tid < L else None

        # 发生溢出了：当前 buf 中保存的是最近 L 条 transition
        earliest_tid = n - L  # buf[0] 对应的全局 id
        if tid < earliest_tid or tid >= n:
            return None

        return tid - earliest_tid

    def add_reward(self, tid, bonus):
        """
        给指定全局 id 的 transition 奖励 +bonus。
        如果这条 transition 已经被丢弃，则返回 False。
        """
        idx = self._global_to_local_idx(tid)
        if idx is None:
            print(f"[ReplayBuffer] transition {tid} already dropped, skip reward backprop.")
            return False

        s, a, r, s2, d, note, difficulty, success = self.buf[idx]
        self.buf[idx] = (s, a, r + bonus, s2, d, note, difficulty, success)
        return True

    def get_transition(self, tid):
        """
        按全局 id 取出当前还存在的 transition。
        不存在则返回 None。
        """
        idx = self._global_to_local_idx(tid)
        if idx is None:
            return None
        return self.buf[idx]
