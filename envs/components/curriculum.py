# envs/components/curriculum.py
from collections import deque
from typing import Deque, Dict, Tuple, Optional

DEFAULT_CFG = {
    # 初始窗口
    "r_min": 1.0,
    "r_max": 1.5,

    # 上/下调 r_max 的最大步长（实际会按比例缩放）
    "inc": 0.3,
    "dec": 0.2,

    # 边界
    "r_max_cap": 15.0,
    "r_max_floor": 1.2,

    # —— 升降难阈值（全局 / 当前级别）——
    "high_thresh": 0.6,       # 全局窗口升难阈值
    "low_thresh": 0.3,        # 降难辅助阈值
    "high_thresh_curr": 0.6,  # 当前级别升难阈值（同级验证）

    # —— 滑窗大小 ——（全局窗口与你此前逻辑一致）
    "window_global": 50,      # 全局滑窗大小
    "window_curr": 50,        # 当前级别滑窗大小（每次变更后重置）

    # —— 升难前的最小驻留与成功数 ——（确保拿到“中等难度”成功经验）
    "min_episodes_at_level": 30,
    "min_successes_at_level": 18,

    # —— 冷却：每次调整后至少等待多少个 episode 才允许再次变更 —— 
    "cooldown_episodes": 8,

    # —— 平滑：成功率的 EMA（0 表示关闭 EMA）——
    "ema_beta": 0.9,
}


class CurriculumPolicy:
    """稳定的课程学习策略：
    - 防连跳：同级验证 + 最小驻留 + 冷却 + 窗口重置
    - 温和调整：按阈值超/欠幅比例化增减 r_max
    - 可选 EMA 平滑
    """

    def __init__(self, cfg_dict: Optional[Dict] = None):
        # 合并配置
        cfg = dict(DEFAULT_CFG)
        if isinstance(cfg_dict, dict):
            cfg.update(cfg_dict)
        self.cfg: Dict = cfg

        # 当前窗口
        self.r_min: float = float(cfg.get("r_min", DEFAULT_CFG["r_min"]))
        self.r_max: float = float(cfg.get("r_max", DEFAULT_CFG["r_max"]))

        # 统计（两条滑窗：全局 / 当前级别）
        self._succ_hist_global: Deque[int] = deque(maxlen=int(cfg.get("window_global", 50)))
        self._succ_hist_curr: Deque[int] = deque(maxlen=int(cfg.get("window_curr", 50)))

        # 等级内计数
        self._episodes_at_level: int = 0
        self._successes_at_level: int = 0

        # 冷却
        self._last_change_ep: int = -10**9

        # EMA 平滑
        self._beta: float = float(cfg.get("ema_beta", 0.0))
        self._ema_global: Optional[float] = None
        self._ema_curr: Optional[float] = None

    # ---- 工厂方法 ----
    @classmethod
    def from_dict(cls, cfg_dict: Optional[Dict]):
        return cls(cfg_dict)

    # ---- 兼容旧接口（不推荐，但保留）----
    def update_by_success_rate(self, success_rate: float) -> Tuple[float, float]:
        """仅按单个成功率作比例化调整（不含冷却/同级验证/驻留）。"""
        try:
            rate = float(success_rate)
        except Exception:
            rate = 0.0

        high = float(self.cfg.get("high_thresh", DEFAULT_CFG["high_thresh"]))
        low = float(self.cfg.get("low_thresh", DEFAULT_CFG["low_thresh"]))
        inc_max = float(self.cfg.get("inc", DEFAULT_CFG["inc"]))
        dec_max = float(self.cfg.get("dec", DEFAULT_CFG["dec"]))
        cap = float(self.cfg.get("r_max_cap", DEFAULT_CFG["r_max_cap"]))
        floor = float(self.cfg.get("r_max_floor", DEFAULT_CFG["r_max_floor"]))

        if rate > high:
            frac = min(1.0, max(0.0, (rate - high) / max(1e-6, 1 - high)))
            delta = inc_max * frac
            self.r_max = min(self.r_max + delta, cap)
        elif rate < low:
            frac = min(1.0, max(0.0, (low - rate) / max(1e-6, low)))
            delta = dec_max * frac
            self.r_max = max(self.r_max - delta, floor)
        # 介于阈值之间不变

        return self.r_min, self.r_max

    # ---- 新推荐接口：按 episode 结束驱动（含防连跳）----
    def update_on_episode_end(self, success: bool, episode_idx: int) -> Tuple[float, float, bool, Dict]:
        """在每个 episode 结束时调用，进行“同级验证 + 最小驻留 + 冷却 + 比例化步长”的课程调整。
        返回: (r_min, r_max, changed, info)
        """
        s = 1 if bool(success) else 0

        # 统计更新
        self._succ_hist_global.append(s)
        self._succ_hist_curr.append(s)
        self._episodes_at_level += 1
        self._successes_at_level += s

        # 成功率（可选 EMA）
        rate_g_raw = sum(self._succ_hist_global) / max(1, len(self._succ_hist_global))
        rate_c_raw = sum(self._succ_hist_curr) / max(1, len(self._succ_hist_curr))

        if self._beta > 0.0:
            self._ema_global = rate_g_raw if self._ema_global is None else (
                self._beta * self._ema_global + (1 - self._beta) * rate_g_raw
            )
            self._ema_curr = rate_c_raw if self._ema_curr is None else (
                self._beta * self._ema_curr + (1 - self._beta) * rate_c_raw
            )
            rate_g = self._ema_global
            rate_c = self._ema_curr
        else:
            rate_g, rate_c = rate_g_raw, rate_c_raw

        # 读取配置
        high_g = float(self.cfg.get("high_thresh", DEFAULT_CFG["high_thresh"]))
        high_c = float(self.cfg.get("high_thresh_curr", DEFAULT_CFG["high_thresh"]))
        low_g = float(self.cfg.get("low_thresh", DEFAULT_CFG["low_thresh"]))

        inc_max = float(self.cfg.get("inc", DEFAULT_CFG["inc"]))
        dec_max = float(self.cfg.get("dec", DEFAULT_CFG["dec"]))
        cap = float(self.cfg.get("r_max_cap", DEFAULT_CFG["r_max_cap"]))
        floor = float(self.cfg.get("r_max_floor", DEFAULT_CFG["r_max_floor"]))

        min_epi = int(self.cfg.get("min_episodes_at_level", 30))
        min_succ = int(self.cfg.get("min_successes_at_level", 18))
        cooldown = int(self.cfg.get("cooldown_episodes", 8))

        # 可否变更（冷却）
        can_change = (episode_idx - self._last_change_ep) >= cooldown
        changed = False
        reason: Optional[str] = None

        # ===== 升难：同级验证 + 最小驻留 + 冷却 =====
        if (not changed) and can_change and (rate_g > high_g) and (rate_c > high_c) \
                and (self._episodes_at_level >= min_epi) and (self._successes_at_level >= min_succ):
            # 比例化上调：超过阈值越多，增量越接近 inc_max（但不超过）
            frac_g = min(1.0, max(0.0, (rate_g - high_g) / max(1e-6, 1 - high_g)))
            frac_c = min(1.0, max(0.0, (rate_c - high_c) / max(1e-6, 1 - high_c)))
            frac = min(frac_g, frac_c)
            delta = inc_max * frac
            if delta > 0:
                self.r_max = min(self.r_max + delta, cap)
                changed = True
                reason = f"up:{delta:.3f} (rg={rate_g:.3f}, rc={rate_c:.3f})"
                self._after_change(episode_idx)

        # ===== 降难：防“卡死” =====
        if (not changed) and can_change:
            should_drop = False
            # 规则1：当前级别窗口已满仍很差
            if (len(self._succ_hist_curr) == self._succ_hist_curr.maxlen) and (rate_c < low_g):
                should_drop = True
            # 规则2：驻留很久几乎无成功
            if (self._episodes_at_level >= max(min_epi, 2 * self._succ_hist_curr.maxlen)) and (self._successes_at_level <= 1):
                should_drop = True

            if should_drop:
                # 比例化下调：低于阈值越多，减幅越接近 dec_max
                frac = min(1.0, max(0.0, (low_g - rate_c) / max(1e-6, low_g)))
                delta = dec_max * frac
                if delta > 0:
                    self.r_max = max(self.r_max - delta, floor)
                    changed = True
                    reason = f"down:{delta:.3f} (rc={rate_c:.3f})"
                    self._after_change(episode_idx)

        info = {
            "rate_global": float(rate_g),
            "rate_curr": float(rate_c),
            "episodes_at_level": int(self._episodes_at_level),
            "successes_at_level": int(self._successes_at_level),
            "changed": bool(changed),
            "reason": reason,
            "r_min": float(self.r_min),
            "r_max": float(self.r_max),
        }
        return float(self.r_min), float(self.r_max), changed, info

    # ---- 难度变更后的重置逻辑 ----
    def _after_change(self, episode_idx: int):
        """变更难度后，重置当前级别窗口与计数，防止旧成绩推动新难度连跳。"""
        self._last_change_ep = int(episode_idx)
        self._succ_hist_curr.clear()
        self._episodes_at_level = 0
        self._successes_at_level = 0
        self._ema_curr = None  # 当前级别 EMA 也重置

    # ---- 只读接口 ----
    def get_window(self) -> Tuple[float, float]:
        return float(self.r_min), float(self.r_max)

    def state(self) -> Dict:
        """导出当前状态（便于日志/保存）。"""
        return {
            "cfg": dict(self.cfg),
            "r_min": float(self.r_min),
            "r_max": float(self.r_max),
            "episodes_at_level": int(self._episodes_at_level),
            "successes_at_level": int(self._successes_at_level),
            "last_change_ep": int(self._last_change_ep),
            "succ_hist_global_len": int(len(self._succ_hist_global)),
            "succ_hist_curr_len": int(len(self._succ_hist_curr)),
            "ema_global": None if self._ema_global is None else float(self._ema_global),
            "ema_curr": None if self._ema_curr is None else float(self._ema_curr),
        }

    def load_state(self, state: Dict):
        """从保存的状态中恢复（可选）。"""
        if not isinstance(state, dict):
            return
        self.cfg.update(state.get("cfg", {}))
        self.r_min = float(state.get("r_min", self.r_min))
        self.r_max = float(state.get("r_max", self.r_max))
        self._episodes_at_level = int(state.get("episodes_at_level", 0))
        self._successes_at_level = int(state.get("successes_at_level", 0))
        self._last_change_ep = int(state.get("last_change_ep", -10**9))
        self._ema_global = state.get("ema_global", None)
        self._ema_curr = state.get("ema_curr", None)
        # 滑窗长度由 cfg 控制，具体样本不恢复（可按需扩展）
