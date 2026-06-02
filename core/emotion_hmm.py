#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HMM 情绪状态转移预测

隐状态=情绪强度级别(5级)，观测=情绪类别(30种)。
MLE + 拉普拉斯平滑训练，增量更新，历史不足时优雅降级。

集成点：
  1. dialogue/AgentMemory.update_emotion() → train_hmm()
  2. text_gen 的 CoT system message → predict_emotion_trend() / predict_next_emotion()
"""

import json, os
from typing import Optional, List, Dict, Tuple
from config import ALL_EMOTIONS, EMOTION_LEVELS

# HMM 隐状态
_LEVELS = ["high", "mid_high", "mid", "low", "positive"]
_LEVEL_IDX = {lv: i for i, lv in enumerate(_LEVELS)}
_N_STATES = len(_LEVELS)

# 观测符号
_OBS_IDX = {e: i for i, e in enumerate(ALL_EMOTIONS)}
_N_OBS = len(ALL_EMOTIONS)

_SMOOTH = 1e-4        # 拉普拉斯平滑
_MIN_HISTORY = 5      # 最少训练样本
_HMM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hmm_model.json")


def _emotion_to_level(emotion: str) -> str:
    for level, emotions in EMOTION_LEVELS.items():
        if emotion in emotions:
            return level
    return "mid"


def _normalize_rows(matrix: List[List[float]]) -> List[List[float]]:
    result = []
    for row in matrix:
        total = sum(row)
        n = len(row)
        result.append([v / total for v in row] if total > 0 else [1.0 / n] * n)
    return result


def _smooth_matrix(matrix: List[List[float]], smooth: float = _SMOOTH) -> List[List[float]]:
    result = []
    for row in matrix:
        smoothed = [v + smooth for v in row]
        total = sum(smoothed)
        result.append([v / total for v in smoothed])
    return result


class EmotionHMM:
    """5隐状态×30观测的离散 HMM"""

    def __init__(self, filepath: str = _HMM_FILE):
        self.filepath = filepath
        self.A = [[1.0 / _N_STATES] * _N_STATES for _ in range(_N_STATES)]
        self.B = [[1.0 / _N_OBS] * _N_OBS for _ in range(_N_STATES)]
        self.pi = [1.0 / _N_STATES] * _N_STATES
        self._trained_len = 0
        self._train_rounds = 0
        self.load()

    def load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.A = data.get("A", self.A)
            self.B = data.get("B", self.B)
            self.pi = data.get("pi", self.pi)
            self._trained_len = data.get("trained_len", 0)
            self._train_rounds = data.get("train_rounds", 0)
            print(f"[EmotionHMM] 已加载模型 | 训练轮次={self._train_rounds} | 训练长度={self._trained_len}")
        except Exception as e:
            print(f"[EmotionHMM] 加载失败，使用默认参数: {e}")

    def save(self):
        try:
            data = {
                "A": self.A, "B": self.B, "pi": self.pi,
                "trained_len": self._trained_len,
                "train_rounds": self._train_rounds,
            }
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[EmotionHMM] 保存失败: {e}")

    def train(self, emotion_history: List[Dict], force: bool = False):
        """MLE 训练（频率计数 + 平滑）。增量模式：新增 ≤3 条时跳过。"""
        if len(emotion_history) < _MIN_HISTORY:
            print(f"[EmotionHMM] 历史不足 ({len(emotion_history)}/{_MIN_HISTORY})，跳过训练")
            return

        new_len = len(emotion_history)
        if not force and new_len <= self._trained_len + 3:
            return

        seq = []
        for record in emotion_history:
            emotion = record.get("emotion", "")
            level = _emotion_to_level(emotion)
            level_idx = _LEVEL_IDX.get(level, 2)
            obs_idx = _OBS_IDX.get(emotion, 0)
            seq.append((level_idx, obs_idx))

        if len(seq) < 2:
            return

        # 转移矩阵 A
        trans_count = [[0] * _N_STATES for _ in range(_N_STATES)]
        for t in range(len(seq) - 1):
            trans_count[seq[t][0]][seq[t + 1][0]] += 1
        self.A = _smooth_matrix(trans_count)

        # 发射矩阵 B
        emit_count = [[0] * _N_OBS for _ in range(_N_STATES)]
        for level_idx, obs_idx in seq:
            emit_count[level_idx][obs_idx] += 1
        self.B = _smooth_matrix(emit_count)

        # 初始分布 pi
        pi_count = [0] * _N_STATES
        if seq:
            pi_count[seq[0][0]] += 1
        if sum(pi_count) < 2:
            self.pi = [1.0 / _N_STATES] * _N_STATES
        else:
            smoothed = [v + _SMOOTH for v in pi_count]
            total = sum(smoothed)
            self.pi = [v / total for v in smoothed]

        self._trained_len = new_len
        self._train_rounds += 1
        self.save()
        print(f"[EmotionHMM] 训练完成 | 轮次={self._train_rounds} | 历史长度={new_len}")

    def predict(self, current_emotion: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """预测下一步最可能的状态级别，返回 [(level, probability), ...]"""
        level = _emotion_to_level(current_emotion)
        i = _LEVEL_IDX.get(level, 2)
        indexed = sorted(enumerate(self.A[i]), key=lambda x: -x[1])
        return [(_LEVELS[idx], prob) for idx, prob in indexed[:top_k]]

    def predict_emotion(self, current_emotion: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """预测下一步具体情绪：P(o) = Σ_j A[s][j]·B[j][o]"""
        level = _emotion_to_level(current_emotion)
        s = _LEVEL_IDX.get(level, 2)

        emotion_probs = [0.0] * _N_OBS
        for j in range(_N_STATES):
            for o in range(_N_OBS):
                emotion_probs[o] += self.A[s][j] * self.B[j][o]

        total = sum(emotion_probs)
        if total > 0:
            emotion_probs = [p / total for p in emotion_probs]

        indexed = sorted(enumerate(emotion_probs), key=lambda x: -x[1])
        return [(ALL_EMOTIONS[idx], prob) for idx, prob in indexed[:top_k]]

    def get_trend_hint(self, current_emotion: str) -> Optional[str]:
        """情绪趋势提示（供 text_gen 注入 CoT prompt）"""
        if self._train_rounds == 0 and self._trained_len < _MIN_HISTORY:
            return None

        predictions = self.predict(current_emotion, top_k=2)
        if not predictions:
            return None

        current_level = _emotion_to_level(current_emotion)
        current_idx = _LEVEL_IDX.get(current_level, 2)
        top_level, top_prob = predictions[0]
        top_idx = _LEVEL_IDX[top_level]

        level_names = {
            "high": "高强度", "mid_high": "中高强度",
            "mid": "中等强度", "low": "低强度", "positive": "正面积极"
        }

        if top_idx < current_idx:
            hint = f"用户情绪可能趋向加重（{level_names.get(top_level, top_level)}，概率{top_prob:.0%}），话术应更多共情和支持，减少理性分析"
        elif top_idx > current_idx:
            hint = f"用户情绪可能趋向缓解（{level_names.get(top_level, top_level)}，概率{top_prob:.0%}），话术可适当引导积极方向"
        else:
            hint = f"用户情绪可能持续在{level_names.get(current_level, current_level)}，话术应保持当前策略"

        print(f"  [EmotionHMM] 趋势 | {current_level}→{top_level}({top_prob:.0%})")
        return hint

    def get_emotion_prediction_hint(self, current_emotion: str) -> Optional[str]:
        """下一轮情绪预测提示"""
        if self._train_rounds == 0 and self._trained_len < _MIN_HISTORY:
            return None

        predictions = self.predict_emotion(current_emotion, top_k=3)
        if not predictions:
            return None

        top3 = "、".join(f"{e}({p:.0%})" for e, p in predictions)
        return f"基于历史模式，用户下一轮情绪可能是：{top3}"


# 全局单例
_hmm_instance: Optional[EmotionHMM] = None

def get_hmm() -> EmotionHMM:
    global _hmm_instance
    if _hmm_instance is None:
        _hmm_instance = EmotionHMM()
    return _hmm_instance


def train_hmm(emotion_history: List[Dict], force: bool = False):
    get_hmm().train(emotion_history, force=force)


def predict_emotion_trend(current_emotion: str) -> Optional[str]:
    return get_hmm().get_trend_hint(current_emotion)


def predict_next_emotion(current_emotion: str) -> Optional[str]:
    return get_hmm().get_emotion_prediction_hint(current_emotion)
