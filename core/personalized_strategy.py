#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""个性化安慰策略选择引擎（UCB1 多臂老虎机）

把每种安慰策略当作一个"摇臂"，用 UCB1 算法平衡探索与利用：
  - 探索：尝试不熟悉的策略，发现潜在更优选择
  - 利用：优先选择历史表现好的策略

UCB1 公式：
  UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
  Q(a): 策略a的平均奖励
  N(a): 策略a被选择的次数
  t:    总轮次
  c:    探索系数（默认1.414，越大越偏向探索）

奖励信号：
  用户情绪明显好转 → +1
  用户情绪轻微好转 → +0.5
  用户情绪不变   → 0
  用户情绪轻微恶化 → -0.5
  用户情绪明显恶化 → -1

按情绪分组：不同情绪状态下分别学习最优策略，
"焦虑"时效果最好的策略不一定适用于"悲伤"。
"""

import math
import json
import os
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 安慰策略定义（与 text_gen.py COMFORT_FOCUS_MAP 对齐）
# ---------------------------------------------------------------------------

STRATEGIES = {
    1:  "deep_empathy",       # 深度共情
    2:  "normalization",      # 正常化
    3:  "perspective_shift",  # 视角转换
    4:  "affirm_strength",    # 肯定力量
    5:  "warm_close",         # 温暖收尾
    6:  "deep_layer",         # 深层共情
    7:  "grounding",          # 当下锚定
    8:  "encourage_express",  # 鼓励表达
    9:  "inner_resource",     # 内在资源
    10: "unconditional",      # 无条件陪伴
    11: "imagery_metaphor",   # 意象隐喻
    12: "glimmer_hope",       # 微光希望
}

STRATEGY_NAMES_CN = {
    1:  "深度共情",
    2:  "正常化",
    3:  "视角转换",
    4:  "肯定力量",
    5:  "温暖收尾",
    6:  "深层共情",
    7:  "当下锚定",
    8:  "鼓励表达",
    9:  "内在资源",
    10: "无条件陪伴",
    11: "意象隐喻",
    12: "微光希望",
}

# 情绪分组：相似情绪共享学习经验
EMOTION_GROUPS = {
    "悲伤": "loss", "心碎": "loss", "失恋": "loss", "失落": "loss",
    "绝望": "despair", "崩溃": "despair", "痛苦": "despair",
    "愤怒": "anger", "委屈": "anger",
    "恐惧": "fear", "无助": "fear",
    "焦虑": "anxiety", "压力大": "anxiety", "心烦": "anxiety",
    "迷茫": "confusion", "无奈": "confusion",
    "孤独": "lonely", "孤单": "lonely",
    "疲惫": "tired", "内耗": "tired", "麻木": "tired",
    "挫败": "frustration", "后悔": "frustration", "自卑": "frustration", "沮丧": "frustration",
    "嫉妒": "frustration", "思念": "frustration",
    "开心": "positive", "平静": "positive", "满足": "positive",
    "期待": "positive", "放松": "positive",
}

DEFAULT_GROUP = "other"


def get_emotion_group(emotion: str) -> str:
    # 先直接查中文键，再尝试英文别名转中文
    group = EMOTION_GROUPS.get(emotion)
    if group:
        return group
    # 英文别名转中文再查
    from core.emotion_fusion import ALIAS_MAP
    cn = ALIAS_MAP.get(emotion)
    if cn:
        return EMOTION_GROUPS.get(cn, DEFAULT_GROUP)
    return DEFAULT_GROUP


# ---------------------------------------------------------------------------
# UCB1 引擎
# ---------------------------------------------------------------------------

class UCB1Bandit:
    """UCB1 多臂老虎机

    按情绪分组独立学习，每组维护一组臂的统计数据。
    """

    def __init__(self, explore_coeff=1.414, min_plays=2):
        self.explore_coeff = explore_coeff
        self.min_plays = min_plays  # 每臂最少尝试次数才参与UCB计算
        # {group: {strategy_id: {"plays": N, "reward_sum": R}}}
        self.arms: Dict[str, Dict[int, Dict]] = {}
        self.total_plays: Dict[str, int] = {}

    def _ensure_group(self, group: str):
        if group not in self.arms:
            self.arms[group] = {}
            self.total_plays[group] = 0
            for sid in STRATEGIES:
                self.arms[group][sid] = {"plays": 0, "reward_sum": 0.0}

    def select(self, emotion: str, exclude: Optional[List[int]] = None) -> int:
        """选择最优策略

        emotion: 当前情绪
        exclude: 排除的策略ID列表
        返回: 策略ID
        """
        group = get_emotion_group(emotion)
        self._ensure_group(group)

        arms = self.arms[group]
        t = self.total_plays[group]

        # 有未充分探索的臂 → 优先探索
        unexplored = [sid for sid in STRATEGIES
                     if arms[sid]["plays"] < self.min_plays
                     and (exclude is None or sid not in exclude)]
        if unexplored:
            # 从未探索臂中随机选
            import random
            return random.choice(unexplored)

        # 全部探索过 → UCB1 选择
        if t == 0:
            import random
            candidates = [sid for sid in STRATEGIES
                         if exclude is None or sid not in exclude]
            return random.choice(candidates) if candidates else 1

        best_id = None
        best_ucb = -float("inf")
        for sid in STRATEGIES:
            if exclude and sid in exclude:
                continue
            arm = arms[sid]
            if arm["plays"] == 0:
                continue
            q = arm["reward_sum"] / arm["plays"]
            ucb = q + self.explore_coeff * math.sqrt(math.log(t) / arm["plays"])
            if ucb > best_ucb:
                best_ucb = ucb
                best_id = sid

        return best_id if best_id is not None else 1

    def update(self, emotion: str, strategy_id: int, reward: float):
        """更新策略奖励

        emotion:     用户当前情绪
        strategy_id: 使用的策略ID
        reward:      奖励值 -1~+1
        """
        group = get_emotion_group(emotion)
        self._ensure_group(group)

        if strategy_id not in self.arms[group]:
            return

        self.arms[group][strategy_id]["plays"] += 1
        self.arms[group][strategy_id]["reward_sum"] += reward
        self.total_plays[group] += 1

    def get_stats(self, emotion: str) -> Dict:
        """获取某情绪组的策略统计"""
        group = get_emotion_group(emotion)
        self._ensure_group(group)

        result = {}
        for sid, arm in self.arms[group].items():
            if arm["plays"] > 0:
                avg = arm["reward_sum"] / arm["plays"]
                result[sid] = {
                    "name": STRATEGY_NAMES_CN.get(sid, str(sid)),
                    "plays": arm["plays"],
                    "avg_reward": round(avg, 3),
                }
        return result

    def get_best_strategy(self, emotion: str) -> Optional[int]:
        """获取某情绪组的最优策略（纯利用，无探索）"""
        group = get_emotion_group(emotion)
        self._ensure_group(group)

        best_id = None
        best_avg = -float("inf")
        for sid, arm in self.arms[group].items():
            if arm["plays"] < self.min_plays:
                continue
            avg = arm["reward_sum"] / arm["plays"]
            if avg > best_avg:
                best_avg = avg
                best_id = sid
        return best_id

    # ---- 持久化 ----

    def to_dict(self) -> dict:
        return {
            "explore_coeff": self.explore_coeff,
            "min_plays": self.min_plays,
            "arms": self.arms,
            "total_plays": self.total_plays,
        }

    def load_dict(self, data: dict):
        self.explore_coeff = data.get("explore_coeff", 1.414)
        self.min_plays = data.get("min_plays", 2)
        # JSON序列化将int key转为str，加载时还原
        raw_arms = data.get("arms", {})
        self.arms = {}
        for group, group_data in raw_arms.items():
            self.arms[group] = {}
            for sid_key, arm_data in group_data.items():
                sid = int(sid_key) if isinstance(sid_key, str) and sid_key.isdigit() else sid_key
                self.arms[group][sid] = arm_data
        raw_total = data.get("total_plays", {})
        self.total_plays = {k: v for k, v in raw_total.items()}

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.load_dict(json.load(f))


# ---------------------------------------------------------------------------
# 奖励计算
# ---------------------------------------------------------------------------

def compute_reward(prev_emotion: str, prev_intensity: float,
                   curr_emotion: str, curr_intensity: float) -> float:
    """根据情绪变化计算奖励

    好转（强度下降或情绪转正面）→ 正奖励
    恶化（强度上升或情绪转负面）→ 负奖励
    """
    # 强度变化（越高越严重，下降=好转）
    intensity_delta = prev_intensity - curr_intensity

    # 情绪分组变化
    POSITIVE_GROUPS = {"positive"}
    prev_group = get_emotion_group(prev_emotion)
    curr_group = get_emotion_group(curr_emotion)

    # 从负面→正面：额外加分
    group_bonus = 0.0
    if prev_group not in POSITIVE_GROUPS and curr_group in POSITIVE_GROUPS:
        group_bonus = 0.5
    elif prev_group in POSITIVE_GROUPS and curr_group not in POSITIVE_GROUPS:
        group_bonus = -0.5

    # 综合奖励（强度变化归一化到 [-1, 1]，加上组别奖励）
    reward = (intensity_delta / 5.0) + group_bonus
    return max(-1.0, min(1.0, reward))


# ---------------------------------------------------------------------------
# 全局实例与接口
# ---------------------------------------------------------------------------

_bandit = None
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ucb1_state.json"
)


def get_bandit() -> UCB1Bandit:
    global _bandit
    if _bandit is None:
        _bandit = UCB1Bandit()
        _bandit.load(_STATE_FILE)
    return _bandit


def select_strategy(emotion: str, exclude: Optional[List[int]] = None) -> int:
    """选择最优策略ID"""
    return get_bandit().select(emotion, exclude)


def update_strategy_reward(emotion: str, strategy_id: int, reward: float):
    """更新策略奖励"""
    b = get_bandit()
    b.update(emotion, strategy_id, reward)
    b.save(_STATE_FILE)


def update_strategy_from_emotion(prev_emotion: str, prev_intensity: float,
                                  curr_emotion: str, curr_intensity: float,
                                  strategy_id: int):
    """根据情绪变化自动计算奖励并更新"""
    reward = compute_reward(prev_emotion, prev_intensity,
                            curr_emotion, curr_intensity)
    update_strategy_reward(curr_emotion, strategy_id, reward)


def strategy_hint(emotion: str) -> str:
    """生成策略提示词（注入 system prompt）

    告诉AI当前选择了什么策略、历史最优策略是什么
    """
    b = get_bandit()
    selected = b.select(emotion)
    stats = b.get_stats(emotion)

    if not stats:
        return ""

    best_id = b.get_best_strategy(emotion)
    group = get_emotion_group(emotion)

    lines = [f"[UCB1策略] 当前情绪组({group})，推荐策略: "
             f"{STRATEGY_NAMES_CN.get(selected, str(selected))}"]

    if best_id and best_id != selected:
        lines.append(f"历史最优: {STRATEGY_NAMES_CN.get(best_id, str(best_id))}")

    # 展示Top3策略
    sorted_stats = sorted(stats.items(), key=lambda x: -x[1]["avg_reward"])[:3]
    if sorted_stats:
        top_str = ", ".join(
            f"{v['name']}({v['avg_reward']:+.2f}/{v['plays']}次)"
            for _, v in sorted_stats
        )
        lines.append(f"Top3: {top_str}")

    return "\n".join(lines)


def reset_bandit():
    """重置学习器"""
    global _bandit
    _bandit = UCB1Bandit()
