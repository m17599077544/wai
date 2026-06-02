#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对话节奏感知引擎（半隐马尔可夫模型 HSMM）

识别对话节奏模式，动态调整回复时机和话术长度：
  - 观察：用户回复间隔、回复长度、情绪强度变化
  - 隐状态：快节奏(urgent)/中节奏(normal)/慢节奏(reflective)/沉默(silent)
  - 根据当前节奏状态建议回复策略（快速短句/正常/缓慢长句/静默陪伴）

实现简化版 HSMM：
  - 4个隐状态，3个观测特征
  - Viterbi解码推断当前节奏
  - 指数衰减的驻留时间分布（区别于HMM的几何分布）
"""

import math
import time
import json
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# 隐状态与观测定义
# ---------------------------------------------------------------------------

STATES = ("urgent", "normal", "reflective", "silent")
STATE_NAMES_CN = {
    "urgent": "急促",
    "normal": "正常",
    "reflective": "沉思",
    "silent": "沉默",
}

# 观测特征分箱
INTERVAL_BINS = ("fast", "medium", "slow", "timeout")  # 回复间隔
LENGTH_BINS = ("short", "medium", "long")               # 回复长度
EMOTION_DELTA = ("worse", "stable", "better")            # 情绪变化方向

# 发射概率 P(观测|状态) — 简化为对各观测维度的独立分布
EMISSION_PROBS = {
    "urgent":    {"interval": [0.60, 0.25, 0.10, 0.05], "length": [0.30, 0.50, 0.20], "delta": [0.40, 0.35, 0.25]},
    "normal":    {"interval": [0.15, 0.55, 0.25, 0.05], "length": [0.15, 0.55, 0.30], "delta": [0.20, 0.55, 0.25]},
    "reflective":{"interval": [0.05, 0.20, 0.55, 0.20], "length": [0.50, 0.35, 0.15], "delta": [0.15, 0.50, 0.35]},
    "silent":    {"interval": [0.02, 0.08, 0.30, 0.60], "length": [0.80, 0.15, 0.05], "delta": [0.10, 0.70, 0.20]},
}

# 转移概率（对角占优，因为节奏状态倾向于持续）
TRANS_PROBS = {
    "urgent":    [0.40, 0.35, 0.10, 0.15],
    "normal":    [0.15, 0.55, 0.20, 0.10],
    "reflective":[0.05, 0.20, 0.55, 0.20],
    "silent":    [0.05, 0.10, 0.25, 0.60],
}

# 驻留时间参数（指数衰减，状态持续轮数的期望值）
DURATION_PARAMS = {
    "urgent": 2.5,
    "normal": 4.0,
    "reflective": 5.0,
    "silent": 3.0,
}

# 初始概率
INIT_PROBS = [0.10, 0.60, 0.20, 0.10]


# ---------------------------------------------------------------------------
# 观测特征提取
# ---------------------------------------------------------------------------

def _bin_interval(seconds):
    """将回复间隔（秒）分箱"""
    if seconds < 3:
        return 0   # fast
    elif seconds < 10:
        return 1   # medium
    elif seconds < 30:
        return 2   # slow
    else:
        return 3   # timeout

def _bin_length(text_len):
    """将回复长度分箱"""
    if text_len < 6:
        return 0   # short
    elif text_len < 25:
        return 1   # medium
    else:
        return 2   # long

def _bin_emotion_delta(prev_intensity, curr_intensity):
    """将情绪变化分箱"""
    diff = curr_intensity - prev_intensity
    if diff >= 1:
        return 2   # better
    elif diff <= -1:
        return 0   # worse
    return 1       # stable


# ---------------------------------------------------------------------------
# HSMM 节奏检测器
# ---------------------------------------------------------------------------

class RhythmDetector:
    """对话节奏检测器（简化 HSMM）

    用 Viterbi 解码推断最近N轮的节奏状态序列，
    取最后一轮的状态作为当前节奏。
    """

    def __init__(self, window=8):
        self.window = window
        self._observations = []  # [(interval_bin, length_bin, delta_bin), ...]
        self._raw_data = []      # [(timestamp, text_len, intensity), ...]
        self._last_timestamp = None
        self._current_rhythm = "normal"
        self._rhythm_history = []

    def observe(self, text_length, intensity, timestamp=None):
        """记录一轮对话的观测

        text_length: 用户回复字符数
        intensity: 情绪强度 1-10
        timestamp: 时间戳（默认当前时间）
        """
        if timestamp is None:
            timestamp = time.time()

        # 计算回复间隔
        if self._last_timestamp is not None:
            interval = timestamp - self._last_timestamp
        else:
            interval = 5.0  # 默认中等间隔
        self._last_timestamp = timestamp

        # 计算情绪变化
        if self._raw_data:
            prev_intensity = self._raw_data[-1][2]
        else:
            prev_intensity = intensity

        obs = (
            _bin_interval(interval),
            _bin_length(text_length),
            _bin_emotion_delta(prev_intensity, intensity),
        )
        self._observations.append(obs)
        self._raw_data.append((timestamp, text_length, intensity))

        # 保持窗口大小
        if len(self._observations) > self.window:
            self._observations = self._observations[-self.window:]
            self._raw_data = self._raw_data[-self.window:]

        # 推断节奏
        self._current_rhythm = self._decode()
        self._rhythm_history.append(self._current_rhythm)
        if len(self._rhythm_history) > 50:
            self._rhythm_history = self._rhythm_history[-50:]

    def _decode(self):
        """Viterbi 解码（简化版，忽略驻留时间）"""
        n = len(self._observations)
        if n == 0:
            return "normal"

        n_states = len(STATES)

        # 初始化
        probs = [INIT_PROBS[s] * self._emit_prob(s, self._observations[0])
                 for s in range(n_states)]

        # 递推
        for t in range(1, n):
            obs = self._observations[t]
            new_probs = []
            for s in range(n_states):
                emit = self._emit_prob(s, obs)
                trans_max = max(
                    probs[prev] * TRANS_PROBS[STATES[prev]][s]
                    for prev in range(n_states)
                )
                new_probs.append(emit * trans_max)
            # 归一化防止下溢
            total = sum(new_probs)
            if total > 1e-12:
                probs = [p / total for p in new_probs]
            else:
                probs = new_probs

        # 取最大概率状态
        best = max(range(n_states), key=lambda s: probs[s])
        return STATES[best]

    @staticmethod
    def _emit_prob(state_idx, obs):
        """计算发射概率 P(obs|state)"""
        state = STATES[state_idx]
        interval_bin, length_bin, delta_bin = obs
        p = (EMISSION_PROBS[state]["interval"][interval_bin] *
             EMISSION_PROBS[state]["length"][length_bin] *
             EMISSION_PROBS[state]["delta"][delta_bin])
        return max(p, 1e-10)

    @property
    def current_rhythm(self):
        return self._current_rhythm

    @property
    def rhythm_cn(self):
        return STATE_NAMES_CN.get(self._current_rhythm, "正常")

    def get_strategy(self):
        """根据当前节奏返回回复策略

        返回: dict {
            "rhythm": str,          # 当前节奏
            "rhythm_cn": str,       # 中文名
            "reply_speed": str,     # 回复速度建议
            "reply_length": str,    # 回复长度建议
            "silence_ok": bool,     # 是否允许沉默
            "hint": str,            # 简短提示
        }
        """
        r = self._current_rhythm
        if r == "urgent":
            return {
                "rhythm": r, "rhythm_cn": "急促",
                "reply_speed": "fast",
                "reply_length": "short",
                "silence_ok": False,
                "hint": "用户在急切表达，快速简短回应，不要打断节奏",
            }
        elif r == "reflective":
            return {
                "rhythm": r, "rhythm_cn": "沉思",
                "reply_speed": "slow",
                "reply_length": "long",
                "silence_ok": True,
                "hint": "用户在深度思考，可以慢慢说，允许停顿和沉默",
            }
        elif r == "silent":
            return {
                "rhythm": r, "rhythm_cn": "沉默",
                "reply_speed": "slow",
                "reply_length": "medium",
                "silence_ok": True,
                "hint": "用户可能不想说话，用极简短句陪伴，不强求回应",
            }
        else:  # normal
            return {
                "rhythm": r, "rhythm_cn": "正常",
                "reply_speed": "normal",
                "reply_length": "medium",
                "silence_ok": False,
                "hint": "正常对话节奏，自然回应即可",
            }

    def reset(self):
        """重置检测状态"""
        self._observations.clear()
        self._raw_data.clear()
        self._last_timestamp = None
        self._current_rhythm = "normal"
        self._rhythm_history.clear()


# ---------------------------------------------------------------------------
# 全局实例与接口
# ---------------------------------------------------------------------------

_detector = None
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "rhythm_state.json"
)


def get_detector():
    global _detector
    if _detector is None:
        _detector = RhythmDetector()
    return _detector


def observe_rhythm(text_length, intensity, timestamp=None):
    """记录对话观测（外部调用）"""
    get_detector().observe(text_length, intensity, timestamp)


def get_current_rhythm():
    """获取当前节奏状态"""
    return get_detector().current_rhythm


def get_rhythm_strategy():
    """获取当前节奏策略"""
    return get_detector().get_strategy()


def rhythm_hint():
    """生成节奏提示词（注入 system prompt）"""
    strategy = get_rhythm_strategy()
    if strategy["rhythm"] == "normal":
        return ""
    return f"【对话节奏】当前节奏：{strategy['rhythm_cn']}。{strategy['hint']}。"


def reset_rhythm():
    """重置节奏检测"""
    get_detector().reset()
