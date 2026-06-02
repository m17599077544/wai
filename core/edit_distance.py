#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编辑距离模糊匹配——语音识别容错

集成点：emotion.py detect_intent()/detect_emotion() 中关键词精确匹配失败后的兜底。

算法：Levenshtein Distance，dp[i][j]，滚动数组 O(min(m,n)) 空间。"""

from typing import Optional, Tuple

# 预设指令词表：[(关键词列表, 意图类型)]
INTENT_VOCAB = [
    (["唱歌", "唱首歌", "唱一首歌", "唱支歌", "唱个歌"], "play_song"),
    (["放歌", "放音乐", "听歌", "听音乐", "播歌", "播音乐", "播放音乐", "来首歌", "放首歌"], "play_music"),
    (["跳舞", "跳个舞", "舞蹈", "跳支舞", "表演舞", "跳一段舞", "跳一首舞"], "dance"),
    (["俯卧撑", "做俯卧撑", "做个俯卧撑", "做一个俯卧撑", "做运动", "运动一下",
      "来几个俯卧撑", "做几个俯卧撑"], "pushup"),
    (["看表演", "表演", "想看", "想表演", "来一个"], "demand"),
    (["不用了", "不用", "谢谢", "感谢", "结束", "停止", "可以了", "没事了", "退出", "再见"], "stop"),
]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离，滚动数组实现"""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    if m < n:
        a, b = b, a
        m, n = n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


def similarity(a: str, b: str) -> float:
    """归一化相似度 = 1 - edit_distance / max(len(a), len(b))"""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - edit_distance(a, b) / max_len


def _dynamic_threshold(keyword_len: int) -> float:
    """短词容错阈值：2字→0.5, 3字→0.6, 4字→0.65, 5+字→0.7"""
    if keyword_len <= 2:
        return 0.5
    elif keyword_len == 3:
        return 0.6
    elif keyword_len == 4:
        return 0.65
    else:
        return 0.7


def _is_fuzzy_match(substr: str, keyword: str) -> Tuple[bool, float]:
    """子串与关键词的模糊判定，动态阈值 + 首字同+ed=1 降级匹配"""
    sim = similarity(substr, keyword)
    threshold = _dynamic_threshold(len(keyword))

    if abs(len(substr) - len(keyword)) > 1:
        return False, sim

    if sim >= threshold:
        return True, sim

    # 首字相同且编辑距离为1时降级通过（中文语音首字通常准确）
    if len(keyword) <= 4 and edit_distance(substr, keyword) == 1:
        if substr and keyword and substr[0] == keyword[0]:
            return True, sim

    return False, sim


def fuzzy_match(text: str) -> Optional[Tuple[str, float, str]]:
    """意图模糊匹配，滑动窗口找最相似子串

    Returns:
        (intent_type, similarity, matched_keyword) 或 None
    """
    text_lower = text.lower().strip()
    text_len = len(text_lower)
    if text_len == 0:
        return None

    best_intent = None
    best_sim = 0.0
    best_keyword = ""

    for keywords, intent_type in INTENT_VOCAB:
        for keyword in keywords:
            if keyword in text_lower:
                continue

            kw_len = len(keyword)
            win_min = max(1, kw_len - 1)
            win_max = min(text_len + 1, kw_len + 2)

            for win_len in range(win_min, win_max):
                for start in range(text_len - win_len + 1):
                    substr = text_lower[start:start + win_len]
                    is_match, sim = _is_fuzzy_match(substr, keyword)

                    if is_match and sim > best_sim:
                        best_sim = sim
                        best_intent = intent_type
                        best_keyword = keyword

    if best_intent:
        return (best_intent, best_sim, best_keyword)
    return None


def fuzzy_match_emotion(text: str) -> Optional[Tuple[str, float]]:
    """情绪关键词模糊匹配（严格策略：ed=1 且首字相同）

    严格于意图匹配，避免"很好"→"平静"等假阳性。

    Returns:
        (emotion, similarity) 或 None
    """
    from config import EMOTION_KEY_WORDS

    text_lower = text.lower().strip()
    text_len = len(text_lower)
    if text_len == 0:
        return None

    best_emotion = None
    best_sim = 0.0

    for emotion, keywords in EMOTION_KEY_WORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                continue

            kw_len = len(keyword)
            win_min = max(1, kw_len - 1)
            win_max = min(text_len + 1, kw_len + 2)

            for win_len in range(win_min, win_max):
                for start in range(text_len - win_len + 1):
                    substr = text_lower[start:start + win_len]
                    ed = edit_distance(substr, keyword)
                    sim = similarity(substr, keyword)

                    if ed == 1 and substr and keyword and substr[0] == keyword[0]:
                        if sim > best_sim:
                            best_sim = sim
                            best_emotion = emotion

    if best_emotion:
        return (best_emotion, best_sim)
    return None
