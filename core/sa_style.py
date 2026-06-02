#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""话术风格自适应引擎（模拟退火）
根据用户反馈动态调整话术参数，找到最适合当前用户的安慰风格。

参数维度：
  warmth:     温柔程度 0~1（高→柔软共情，低→理性引导）
  depth:      共情深度 0~1（高→深挖感受，低→轻描淡写）
  directive:  建议力度 0~1（高→给方向，低→纯倾听）
  imagery:    意象使用 0~1（高→比喻隐喻，低→直白朴素）

反馈信号：
  用户情绪好转 → 正反馈（当前参数得分+1）
  用户情绪不变 → 零反馈
  用户情绪恶化 → 负反馈（当前参数得分-1）

模拟退火：
  温度高时大胆探索新风格，温度低时收敛到已知好风格
  每N轮退火一步，防止风格固化
"""

import math
import random
import time
import json
import os


class StyleParameters:
    """话术风格参数向量"""

    DIMS = ("warmth", "depth", "directive", "imagery")

    def __init__(self, warmth=0.6, depth=0.5, directive=0.3, imagery=0.4):
        self.warmth = warmth
        self.depth = depth
        self.directive = directive
        self.imagery = imagery

    def to_dict(self):
        return {d: round(getattr(self, d), 3) for d in self.DIMS}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d.get(k, 0.5) for k in cls.DIMS})

    def perturb(self, step_size=0.15):
        """随机扰动，生成邻居状态"""
        new = StyleParameters(**self.to_dict())
        for dim in self.DIMS:
            delta = random.gauss(0, step_size)
            val = getattr(new, dim) + delta
            setattr(new, dim, max(0.0, min(1.0, val)))
        return new

    def distance(self, other):
        """欧氏距离"""
        return math.sqrt(sum(
            (getattr(self, d) - getattr(other, d)) ** 2
            for d in self.DIMS
        ))


class SimulatedAnnealing:
    """模拟退火优化器"""

    def __init__(self, initial_temp=1.0, cooling_rate=0.92,
                 min_temp=0.15, reheat_threshold=8):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.reheat_threshold = reheat_threshold
        self.reset()

    def reset(self):
        self.temperature = self.initial_temp
        self.current_params = StyleParameters()
        self.best_params = StyleParameters()
        self.best_score = 0.0
        self.current_score = 0.0
        self.rounds_since_improve = 0
        self.total_rounds = 0
        self.score_history = []

    def _accept(self, new_score, current_score):
        """Metropolis准则：接受更优解；较劣解以概率接受"""
        delta = new_score - current_score
        if delta >= 0:
            return True
        if self.temperature <= 1e-6:
            return False
        prob = math.exp(delta / self.temperature)
        return random.random() < prob

    def step(self, feedback):
        """根据用户反馈执行一步退火

        feedback: -1(恶化) / 0(不变) / 1(好转)
        返回: (StyleParameters, accepted: bool)
        """
        self.total_rounds += 1
        new_score = self.current_score + feedback
        self.score_history.append(feedback)

        # 生成邻居候选
        candidate = self.current_params.perturb(
            step_size=0.1 + 0.2 * self.temperature
        )

        accepted = self._accept(new_score, self.current_score)
        if accepted:
            self.current_params = candidate
            self.current_score = new_score
            if new_score > self.best_score:
                self.best_score = new_score
                self.best_params = StyleParameters(**candidate.to_dict())
                self.rounds_since_improve = 0
            else:
                self.rounds_since_improve += 1
        else:
            self.rounds_since_improve += 1

        # 退火冷却
        self.temperature = max(
            self.min_temp,
            self.temperature * self.cooling_rate
        )

        # 长期无改善→重新加热（避免局部最优）
        if self.rounds_since_improve >= self.reheat_threshold:
            self.temperature = min(
                self.initial_temp * 0.7,
                self.temperature + 0.3
            )
            self.rounds_since_improve = 0

        return self.current_params, accepted

    def get_style_params(self):
        """获取当前推荐参数（温度低时用最优解，温度高时用当前探索解）"""
        if self.temperature < 0.3:
            return self.best_params
        return self.current_params

    def to_dict(self):
        return {
            "temperature": round(self.temperature, 4),
            "current_params": self.current_params.to_dict(),
            "best_params": self.best_params.to_dict(),
            "current_score": self.current_score,
            "best_score": self.best_score,
            "total_rounds": self.total_rounds,
            "rounds_since_improve": self.rounds_since_improve,
        }

    def load_dict(self, d):
        self.temperature = d.get("temperature", self.initial_temp)
        self.current_params = StyleParameters.from_dict(d.get("current_params", {}))
        self.best_params = StyleParameters.from_dict(d.get("best_params", {}))
        self.current_score = d.get("current_score", 0.0)
        self.best_score = d.get("best_score", 0.0)
        self.total_rounds = d.get("total_rounds", 0)
        self.rounds_since_improve = d.get("rounds_since_improve", 0)


# 全局实例
_sa_engine = None
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "sa_style_state.json"
)


def get_sa_engine():
    global _sa_engine
    if _sa_engine is None:
        _sa_engine = SimulatedAnnealing()
        _load_state(_sa_engine)
    return _sa_engine


def _load_state(sa):
    try:
        if os.path.exists(_STATE_FILE):
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                sa.load_dict(json.load(f))
            print(f"[模拟退火] 已加载状态: 温度={sa.temperature:.3f}, "
                  f"最优分={sa.best_score:.1f}, 轮次={sa.total_rounds}")
    except Exception as e:
        print(f"[模拟退火] 加载状态失败，使用默认: {e}")


def _save_state(sa):
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sa.to_dict(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[模拟退火] 保存状态失败: {e}")


def evaluate_feedback(prev_emotion, prev_intensity, curr_emotion, curr_intensity):
    """根据情绪变化计算反馈信号

    返回: -1(恶化) / 0(不变) / 1(好转)
    """
    from config import EMOTION_INTENSITY, POSITIVE_EMOTIONS

    prev_level = EMOTION_INTENSITY.get(prev_emotion, 5)
    curr_level = EMOTION_INTENSITY.get(curr_emotion, 5)

    # 综合考虑关键词强度和传入强度
    prev_val = (prev_level + prev_intensity) / 2
    curr_val = (curr_level + curr_intensity) / 2

    # 正面情绪特殊处理
    if curr_emotion in POSITIVE_EMOTIONS and prev_emotion not in POSITIVE_EMOTIONS:
        return 1
    if prev_emotion in POSITIVE_EMOTIONS and curr_emotion not in POSITIVE_EMOTIONS:
        return -1

    diff = prev_val - curr_val
    if diff >= 1.5:
        return 1
    elif diff <= -1.5:
        return -1
    return 0


def update_style(prev_emotion, prev_intensity, curr_emotion, curr_intensity):
    """外部调用入口：根据情绪变化更新话术风格

    返回: StyleParameters（当前推荐参数）
    """
    sa = get_sa_engine()
    feedback = evaluate_feedback(
        prev_emotion, prev_intensity, curr_emotion, curr_intensity
    )
    params, accepted = sa.step(feedback)
    _save_state(sa)

    p = params.to_dict()
    print(f"  [模拟退火] 反馈={feedback}, 接受={accepted}, "
          f"温度={sa.temperature:.3f}, 参数={p}")
    return params


def get_current_style():
    """获取当前话术风格参数（不触发退火）"""
    return get_sa_engine().get_style_params()


def style_to_prompt_hints(params=None):
    """将风格参数转化为 DeepSeek prompt 风格提示词

    返回: str，可直接拼接到 system prompt
    """
    if params is None:
        params = get_current_style()
    p = params.to_dict()

    hints = []

    # 温柔程度
    w = p["warmth"]
    if w >= 0.7:
        hints.append("语气极度温柔柔软，像轻声耳语")
    elif w >= 0.4:
        hints.append("语气温和自然，像朋友聊天")
    else:
        hints.append("语气平和理性，像咨询师引导")

    # 共情深度
    d = p["depth"]
    if d >= 0.7:
        hints.append("深挖用户没说出口的感受，直击内心")
    elif d >= 0.4:
        hints.append("适度共情，既理解又不过度解读")
    else:
        hints.append("点到为止，不过度深入情绪")

    # 建议力度
    dr = p["directive"]
    if dr >= 0.7:
        hints.append("可以主动给方向或小建议")
    elif dr >= 0.4:
        hints.append("偶尔暗示可能性，不强加建议")
    else:
        hints.append("纯倾听陪伴，不主动给建议")

    # 意象使用
    im = p["imagery"]
    if im >= 0.7:
        hints.append("多用自然意象和比喻（海、天空、季节、光）")
    elif im >= 0.4:
        hints.append("偶尔用比喻点缀，以直白为主")
    else:
        hints.append("直白朴素，不用比喻和意象")

    return "【风格自适应】" + "；".join(hints) + "。"
