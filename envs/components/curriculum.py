# envs/components/curriculum.py
DEFAULT_CFG = {
    "r_min": 1.0,
    "r_max": 1.5,
    "inc": 0.3,          # 成功率高时 r_max 增量
    "dec": 0.2,          # 成功率低时 r_max 减量
    "r_max_cap": 8.0,    # r_max 上限
    "r_max_floor": 1.2,  # r_max 下限
    "high_thresh": 0.6,  # 成功率高阈值
    "low_thresh": 0.3,   # 成功率低阈值
}

class CurriculumPolicy(object):
    def __init__(self, cfg_dict=None):
        # 合并默认配置
        cfg = dict(DEFAULT_CFG)
        if isinstance(cfg_dict, dict):
            cfg.update(cfg_dict)

        self.cfg = cfg
        self.r_min = float(cfg.get("r_min", DEFAULT_CFG["r_min"]))
        self.r_max = float(cfg.get("r_max", DEFAULT_CFG["r_max"]))

    @classmethod
    def from_dict(cls, cfg_dict):
        """从 dict 创建策略；cfg_dict 可为 None。"""
        return cls(cfg_dict)

    def update_by_success_rate(self, success_rate):
        """根据成功率调整窗口，仅调整 r_max；r_min 保持不变（更稳定）。"""
        try:
            rate = float(success_rate)
        except Exception:
            rate = 0.0

        high = float(self.cfg.get("high_thresh", DEFAULT_CFG["high_thresh"]))
        low  = float(self.cfg.get("low_thresh", DEFAULT_CFG["low_thresh"]))
        inc  = float(self.cfg.get("inc", DEFAULT_CFG["inc"]))
        dec  = float(self.cfg.get("dec", DEFAULT_CFG["dec"]))
        cap  = float(self.cfg.get("r_max_cap", DEFAULT_CFG["r_max_cap"]))
        floor= float(self.cfg.get("r_max_floor", DEFAULT_CFG["r_max_floor"]))

        if rate > high:
            self.r_max = min(self.r_max + inc, cap)
        elif rate < low:
            self.r_max = max(self.r_max - dec, floor)
        # 介于 low~high 不变

        return self.r_min, self.r_max

    def get_window(self):
        """返回当前 (r_min, r_max)。"""
        return float(self.r_min), float(self.r_max)

    def state(self):
        """便于日志/保存。"""
        return {"r_min": float(self.r_min), "r_max": float(self.r_max)}
