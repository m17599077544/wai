#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""话题回避检测（TF-IDF + 余弦相似度）

检测用户是否在回避某个话题：
  - 记录机器人提出的问题（话题锚点）和用户回答的语义距离
  - 连续N次回答与问题低相关 → 判定为回避
  - 回避检测触发后自动调整策略（换话题/降低深度/暂停追问）

依赖：dialogue.TransformerSemanticEngine（可用时用语义向量，不可用时降级为关键词重叠）
"""

import math
import os
import json
import re
from collections import Counter


# ---------------------------------------------------------------------------
# TF-IDF 向量化器（轻量实现，不依赖 sklearn）
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    """简易中文 TF-IDF 向量化器"""

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocab = {}       # token -> index
        self.idf = {}         # token -> idf值
        self._doc_count = 0
        self._doc_freq = Counter()

    @staticmethod
    def _tokenize(text):
        """中文分词：字符级 bigram + 单字，兼顾无分词器场景"""
        text = re.sub(r'[^\u4e00-\u9fff\w]', '', text.lower())
        tokens = list(text)
        bigrams = [text[i:i+2] for i in range(len(text) - 1)]
        return tokens + bigrams

    def fit(self, documents):
        """根据语料拟合 IDF"""
        self._doc_count = len(documents)
        self._doc_freq.clear()
        for doc in documents:
            seen = set(self._tokenize(doc))
            for t in seen:
                self._doc_freq[t] += 1
        # 选 top-N 高频词作为词表
        top = self._doc_freq.most_common(self.max_features)
        self.vocab = {t: i for i, (t, _) in enumerate(top)}
        # 计算 IDF = log((1+N)/(1+df)) + 1
        self.idf = {
            t: math.log((1 + self._doc_count) / (1 + df)) + 1
            for t, df in top
        }
        return self

    def transform(self, text):
        """将文本转为 TF-IDF 稀疏向量（dict: index -> value）"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {}
        for t, count in tf.items():
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] = (count / total) * self.idf.get(t, 1.0)
        return vec

    def fit_transform_one(self, reference_docs, query):
        """对单条查询做即时向量化（用参考文档拟合IDF）"""
        self.fit(reference_docs + [query])
        return self.transform(query)


# ---------------------------------------------------------------------------
# 余弦相似度
# ---------------------------------------------------------------------------

def cosine_sim(v1, v2):
    """两个稀疏向量(dict)的余弦相似度"""
    if not v1 or not v2:
        return 0.0
    common = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# 话题回避检测器
# ---------------------------------------------------------------------------

class TopicAvoidanceDetector:
    """话题回避检测器

    工作流程：
    1. record_question(text)  — 记录机器人提出的话题/问题
    2. record_answer(text)    — 记录用户回答
    3. check_avoidance()      — 返回回避状态

    回避判定：
    - 连续 N 次回答与问题的语义相似度 < threshold
    - 且回答长度偏短（< avg_len * 0.5）
    """

    def __init__(self, window_size=3, threshold=0.18,
                 short_answer_ratio=0.5, consecutive_trigger=2):
        self.window_size = window_size
        self.threshold = threshold
        self.short_answer_ratio = short_answer_ratio
        self.consecutive_trigger = consecutive_trigger

        self._questions = []      # 最近的问题
        self._answers = []        # 最近的回答
        self._similarities = []   # 最近的相关度
        self._avoidance_count = 0 # 连续回避次数

        # 尝试加载语义引擎
        self._semantic = None
        try:
            from dialogue import _TRANSFORMER_DIALOGUE_MANAGER
            if _TRANSFORMER_DIALOGUE_MANAGER and _TRANSFORMER_DIALOGUE_MANAGER.semantic.available:
                self._semantic = _TRANSFORMER_DIALOGUE_MANAGER.semantic
        except Exception:
            pass

        self._tfidf = TfidfVectorizer()

    def record_question(self, text):
        """记录机器人提问"""
        self._questions.append(text)
        if len(self._questions) > self.window_size * 2:
            self._questions = self._questions[-self.window_size * 2:]

    def record_answer(self, text):
        """记录用户回答，计算与最近问题的相关度"""
        self._answers.append(text)
        if len(self._answers) > self.window_size * 2:
            self._answers = self._answers[-self.window_size * 2:]

        # 计算与最近问题的相关度
        sim = self._compute_similarity(text)
        self._similarities.append(sim)
        if len(self._similarities) > self.window_size * 2:
            self._similarities = self._similarities[-self.window_size * 2:]

        # 更新连续回避计数
        if sim < self.threshold:
            self._avoidance_count += 1
        else:
            self._avoidance_count = 0

    def _compute_similarity(self, answer):
        """计算回答与最近问题的相关度"""
        if not self._questions:
            return 0.5  # 无问题参照，给默认值

        last_q = self._questions[-1]

        # 优先用 Transformer 语义引擎
        if self._semantic and self._semantic.available:
            try:
                return self._semantic.similarity(last_q, answer)
            except Exception:
                pass

        # 降级：TF-IDF 余弦相似度
        try:
            ref_docs = self._questions[-self.window_size:]
            q_vec = self._tfidf.fit_transform_one(ref_docs, last_q)
            a_vec = self._tfidf.transform(answer)
            return cosine_sim(q_vec, a_vec)
        except Exception:
            # 最终降级：关键词重叠
            return self._keyword_overlap(last_q, answer)

    @staticmethod
    def _keyword_overlap(text1, text2):
        """关键词重叠率（Jaccard）"""
        def _chars(t):
            return set(re.sub(r'[^\u4e00-\u9fff]', '', t))
        s1, s2 = _chars(text1), _chars(text2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def check_avoidance(self):
        """检测当前是否在回避话题

        返回: dict {
            "is_avoiding": bool,       # 是否回避
            "consecutive": int,        # 连续回避次数
            "avg_similarity": float,   # 近期平均相关度
            "suggestion": str          # 策略建议
        }
        """
        recent_sims = self._similarities[-self.window_size:]
        avg_sim = sum(recent_sims) / len(recent_sims) if recent_sims else 0.5

        is_avoiding = self._avoidance_count >= self.consecutive_trigger

        # 平均回答长度
        recent_answers = self._answers[-self.window_size:]
        avg_len = sum(len(a) for a in recent_answers) / len(recent_answers) if recent_answers else 20

        # 短回答也是回避信号
        is_short = (avg_len < 8 and len(recent_answers) >= 2)

        if is_avoiding or is_short:
            level = "high" if self._avoidance_count >= 3 else "medium"
        else:
            level = "none"

        suggestion = self._get_suggestion(level, avg_sim)

        return {
            "is_avoiding": is_avoiding or is_short,
            "level": level,
            "consecutive": self._avoidance_count,
            "avg_similarity": round(avg_sim, 3),
            "avg_answer_length": round(avg_len, 1),
            "suggestion": suggestion,
        }

    @staticmethod
    def _get_suggestion(level, avg_sim):
        """根据回避程度返回策略建议"""
        if level == "high":
            return "stop_probing"    # 停止追问，切换轻松话题
        elif level == "medium":
            return "soften_approach" # 软化提问方式，降低深度
        return "continue"            # 正常继续

    def reset(self):
        """重置检测状态（情绪切换时调用）"""
        self._questions.clear()
        self._answers.clear()
        self._similarities.clear()
        self._avoidance_count = 0


# ---------------------------------------------------------------------------
# 全局实例与接口
# ---------------------------------------------------------------------------

_detector = None
_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "avoidance_state.json"
)


def get_detector():
    global _detector
    if _detector is None:
        _detector = TopicAvoidanceDetector()
    return _detector


def record_question(text):
    """记录机器人提问（外部调用）"""
    get_detector().record_question(text)


def record_answer(text):
    """记录用户回答（外部调用）"""
    get_detector().record_answer(text)


def check_avoidance():
    """检测回避状态（外部调用）"""
    return get_detector().check_avoidance()


def reset_avoidance():
    """重置回避检测（情绪切换时调用）"""
    get_detector().reset()


def avoidance_hint():
    """生成回避状态提示词（注入 system prompt）

    返回: str，可直接拼接到 prompt
    """
    result = check_avoidance()
    if not result["is_avoiding"]:
        return ""

    level = result["level"]
    if level == "high":
        return ("【话题回避警告】用户可能不愿深入当前话题。"
                "立即停止追问，切换到轻松安全的话题，如日常、天气、美食。")
    elif level == "medium":
        return ("【话题回避提示】用户对当前话题回应偏少。"
                "尝试换一种更柔和的方式提问，或给出自己的小故事代替直接追问。")
    return ""
