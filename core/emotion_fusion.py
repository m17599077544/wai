#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多模态情绪融合引擎（D-S证据理论）

融合多个情绪来源（文本/面部/语音）的判断，解决冲突：
  - 同一情绪多个源一致 → 增强置信度
  - 不同源冲突 → 降低置信度，输出更谨慎的判断

D-S证据理论核心：
  - 基本概率分配 m(A): 证据对命题A的支持程度
  - 组合规则: m12(A) = [m1(A)*m2(A)] / [1 - K]
  - K = 冲突系数，K越大说明证据间冲突越大

当前模态：
  1. 文本情绪（DeepSeek + 关键词检测）
  2. 面部表情（DeepFace）
  3. 语音情绪（预留接口）
"""

import math
import json
import os
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 情绪类别（与项目 emotion.py 对齐）
# ---------------------------------------------------------------------------

EMOTION_CATEGORIES = [
    "悲伤", "焦虑", "愤怒", "恐惧", "迷茫", "孤独",
    "疲惫", "烦躁", "无助", "崩溃", "内疚", "羞耻",
    "开心", "平静", "感激", "期待",
]

# 映射：将各类来源的情绪名映射到标准类别
ALIAS_MAP = {
    "sad": "悲伤", "anxious": "焦虑", "anxiety": "焦虑",
    "angry": "愤怒", "anger": "愤怒", "fear": "恐惧",
    "confused": "迷茫", "lonely": "孤独", "tired": "疲惫",
    "irritable": "烦躁", "helpless": "无助", "devastated": "崩溃",
    "guilty": "内疚", "ashamed": "羞耻",
    "happy": "开心", "calm": "平静", "grateful": "感激", "excited": "期待",
    "neutral": "平静", "surprise": "期待",
}


# ---------------------------------------------------------------------------
# D-S 证据组合
# ---------------------------------------------------------------------------

def combine_masses(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """Dempster 组合规则

    m1, m2: 基本概率分配 {emotion: probability}
    所有值的和应为 1.0（或接近，含 "unknown" 项）
    """
    # 计算冲突系数 K
    K = 0.0
    keys1 = set(m1.keys())
    keys2 = set(m2.keys())
    for a in keys1:
        for b in keys2:
            if a != b and a != "unknown" and b != "unknown":
                K += m1.get(a, 0) * m2.get(b, 0)

    # 避免除零
    if K >= 1.0:
        K = 0.999

    # 组合
    result = {}
    all_keys = keys1 | keys2
    for a in all_keys:
        if a == "unknown":
            continue
        # m1(a)*m2(a) + m1(a)*m2(unknown) + m1(unknown)*m2(a)
        val = (m1.get(a, 0) * m2.get(a, 0) +
               m1.get(a, 0) * m2.get("unknown", 0) +
               m1.get("unknown", 0) * m2.get(a, 0))
        result[a] = val / (1.0 - K)

    # 归一化
    total = sum(result.values())
    if total > 1e-9:
        result = {k: v / total for k, v in result.items()}

    return result


def emotion_to_mass(emotion: str, confidence: float) -> Dict[str, float]:
    """将单一情绪判断转为基本概率分配

    emotion: 情绪名（中文或英文别名）
    confidence: 置信度 0~1
    """
    # 映射别名
    emo = ALIAS_MAP.get(emotion, emotion)
    if emo not in EMOTION_CATEGORIES:
        emo = "平静"  # 未知情绪默认平静

    mass = {emo: confidence, "unknown": 1.0 - confidence}
    return mass


# ---------------------------------------------------------------------------
# 多模态融合器
# ---------------------------------------------------------------------------

class MultimodalFusion:
    """多模态情绪融合

    输入各模态的情绪判断和置信度，
    输出融合后的情绪和置信度。
    """

    def __init__(self, min_confidence=0.25):
        self.min_confidence = min_confidence
        self._last_fusion = None  # 最近一次融合结果
        self._conflict_history = []  # 冲突历史

    def fuse(self, sources: Dict[str, Tuple[str, float]]) -> Dict:
        """融合多个来源的情绪判断

        sources: {"text": ("悲伤", 0.8), "face": ("开心", 0.5), ...}
                 每个源提供 (emotion, confidence) 对

        返回: {
            "emotion": str,       # 融合后情绪
            "confidence": float,  # 融合后置信度
            "conflict": float,    # 冲突系数 0~1
            "sources": dict,      # 各源原始判断
            "distribution": dict, # 融合后概率分布
        }
        """
        if not sources:
            return {
                "emotion": "平静", "confidence": 0.0,
                "conflict": 0.0, "sources": {}, "distribution": {},
            }

        # 只有一个源
        if len(sources) == 1:
            name, (emo, conf) = list(sources.items())[0]
            emo = ALIAS_MAP.get(emo, emo)
            return {
                "emotion": emo, "confidence": conf,
                "conflict": 0.0, "sources": sources,
                "distribution": {emo: conf},
            }

        # 逐步组合各源证据
        masses = []
        for name, (emo, conf) in sources.items():
            mass = emotion_to_mass(emo, conf)
            masses.append((name, mass))

        # 两两组合
        combined = masses[0][1]
        total_conflict = 0.0
        for i in range(1, len(masses)):
            # 计算冲突
            K = self._compute_conflict(combined, masses[i][1])
            total_conflict = max(total_conflict, K)
            combined = combine_masses(combined, masses[i][1])

        # 提取结果
        distribution = {k: v for k, v in combined.items()
                       if k != "unknown" and v > 0.01}
        if not distribution:
            distribution = {"平静": 1.0}

        best_emotion = max(distribution, key=distribution.get)
        best_conf = distribution[best_emotion]

        # 冲突高时降低置信度
        if total_conflict > 0.5:
            best_conf *= (1.0 - total_conflict * 0.3)

        self._last_fusion = {
            "emotion": best_emotion,
            "confidence": round(best_conf, 3),
            "conflict": round(total_conflict, 3),
            "sources": {k: v for k, v in sources.items()},
            "distribution": {k: round(v, 3) for k, v in sorted(
                distribution.items(), key=lambda x: -x[1])},
        }
        self._conflict_history.append(total_conflict)
        if len(self._conflict_history) > 20:
            self._conflict_history = self._conflict_history[-20:]

        return self._last_fusion

    @staticmethod
    def _compute_conflict(m1, m2):
        """计算两个mass函数的冲突系数"""
        K = 0.0
        for a in m1:
            for b in m2:
                if a != b and a != "unknown" and b != "unknown":
                    K += m1.get(a, 0) * m2.get(b, 0)
        return K

    def get_fusion_hint(self):
        """生成融合提示词（注入 system prompt）"""
        if not self._last_fusion:
            return ""

        conflict = self._last_fusion.get("conflict", 0)
        if conflict < 0.3:
            return ""

        emo = self._last_fusion.get("emotion", "")
        conf = self._last_fusion.get("confidence", 0)

        if conflict >= 0.6:
            return (f"【多模态冲突】不同信号源对用户情绪判断不一致"
                    f"（冲突度{conflict:.0%}），融合判断为{emo}（置信度{conf:.0%}）。"
                    f"请更谨慎地确认用户状态，不要急于下结论。")
        elif conflict >= 0.3:
            return (f"【多模态提示】信号源存在轻微分歧"
                    f"（冲突度{conflict:.0%}），综合判断为{emo}。"
                    f"注意观察用户后续反应来验证判断。")
        return ""

    def reset(self):
        self._last_fusion = None
        self._conflict_history.clear()


# ---------------------------------------------------------------------------
# 全局实例与接口
# ---------------------------------------------------------------------------

_fusion = None
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fusion_state.json"
)


def get_fusion():
    global _fusion
    if _fusion is None:
        _fusion = MultimodalFusion()
    return _fusion


def fuse_emotions(sources):
    """融合多模态情绪（外部调用）

    sources: {"text": ("悲伤", 0.8), "face": ("开心", 0.5), ...}
    """
    return get_fusion().fuse(sources)


def fusion_hint():
    """获取融合提示词"""
    return get_fusion().get_fusion_hint()


def reset_fusion():
    """重置融合器"""
    get_fusion().reset()
