#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""危机预警：多维加权评分 + 分级响应"""
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# 一级危机词（高风险，直接触发）
CRISIS_KEYWORDS_LEVEL1 = [
    "自杀", "不想活", "死了算了", "活够了", "活着没意思",
    "不想活了", "活腻了", "不想再活", "想死", "想一死了之",
]

# 二级危机词（中高风险）
CRISIS_KEYWORDS_LEVEL2 = [
    "割腕", "跳楼", "安眠药", "轻生", "了结自己",
    "解脱", "离开这个世界", "结束一切", "没有我更好",
    "活着是负担", "不配活着", "去死", "该死",
]

# 三级危机词（情感崩溃指标）
CRISIS_KEYWORDS_LEVEL3 = [
    "绝望", "崩溃", "痛苦", "生不如死", "痛不欲生",
    "无人能救", "没救了", "彻底完了", "万念俱灰",
]

CRISIS_PERSISTENT_KEYWORDS = [
    "一直", "总是", "从来", "每次", "永远",
    "一直这样", "永远这样", "总是这样",
]


@dataclass
class CrisisConfig:
    weight_intensity: float = 0.30
    weight_keywords: float = 0.40
    weight_trend: float = 0.20
    weight_history: float = 0.10

    threshold_safe: float = 0.3
    threshold_attention: float = 0.5
    threshold_warning: float = 0.7
    threshold_critical: float = 0.85

    trend_window_size: int = 5
    trend_increase_threshold: float = 0.3


@dataclass
class EmotionRecord:
    emotion: str
    intensity: float  # 0-1 归一化
    timestamp: float = field(default_factory=time.time)
    text: str = ""


class CrisisDetector:
    """危机预警系统"""

    def __init__(self, config: CrisisConfig = None):
        self.config = config or CrisisConfig()
        self.emotion_history: deque = deque(maxlen=100)
        self.crisis_history: List[Dict] = []
        self.user_profile: Dict = {
            'baseline_intensity': 0.3,
            'crisis_count': 0,
            'last_crisis': None,
            'total_sessions': 0,
        }
        self.session_start: Optional[float] = None
        self.session_emotion_records: List[EmotionRecord] = []

    def start_session(self):
        self.session_start = time.time()
        self.session_emotion_records = []
        self.user_profile['total_sessions'] += 1

    def _check_crisis_keywords(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        for kw in CRISIS_KEYWORDS_LEVEL1:
            if kw in text_lower: return 0.9
        for kw in CRISIS_KEYWORDS_LEVEL2:
            if kw in text_lower: return 0.7
        for kw in CRISIS_KEYWORDS_LEVEL3:
            if kw in text_lower: return 0.5
        return 0.0

    def _normalize_intensity(self, intensity: float) -> float:
        return min(1.0, max(0.0, intensity / 10.0))

    def _calculate_intensity_score(self, intensity: float, duration_minutes: float) -> float:
        normalized = self._normalize_intensity(intensity)
        base_score = normalized
        duration_bonus = min(0.15, duration_minutes * 0.03)
        if intensity >= 9:
            base_score = min(1.0, base_score + 0.1)
        return min(1.0, base_score + duration_bonus)

    def _calculate_trend_score(self) -> float:
        if len(self.emotion_history) < 3:
            return 0.0
        recent = list(self.emotion_history)[-self.config.trend_window_size:]
        intensities = [r.intensity for r in recent]
        increases = sum(
            1 for i in range(len(intensities) - 1)
            if intensities[i + 1] > intensities[i] + self.config.trend_increase_threshold)
        is_continuously_worsening = all(
            intensities[i] <= intensities[i + 1] for i in range(len(intensities) - 1))
        if is_continuously_worsening and len(recent) >= 3:
            return 0.85
        if increases >= len(intensities) / 2:
            return 0.6
        return 0.15

    def _calculate_history_score(self) -> float:
        score = 0.0
        one_week_ago = time.time() - 7 * 24 * 3600
        recent_crisis = sum(1 for c in self.crisis_history if c['timestamp'] > one_week_ago)
        score += min(0.35, recent_crisis * 0.1)
        if self.user_profile['last_crisis']:
            hours_since = (time.time() - self.user_profile['last_crisis']) / 3600
            if hours_since < 24:
                score += 0.4
        score += min(0.15, self.user_profile['crisis_count'] * 0.02)
        return min(1.0, score)

    def add_emotion_record(self, emotion: str, intensity: float, text: str = ""):
        record = EmotionRecord(
            emotion=emotion,
            intensity=self._normalize_intensity(intensity),
            text=text)
        self.emotion_history.append(record)
        self.session_emotion_records.append(record)

    def detect_crisis(self, current_intensity: float, current_emotion: str, user_text: str = "") -> Dict:
        """综合危机检测，返回风险评分+等级+建议"""
        duration_minutes = 0.0
        if self.session_start:
            duration_minutes = (time.time() - self.session_start) / 60.0

        intensity_score = self._calculate_intensity_score(current_intensity, duration_minutes)
        keyword_score = self._check_crisis_keywords(user_text)
        trend_score = self._calculate_trend_score()
        history_score = self._calculate_history_score()

        risk_score = (
            intensity_score * self.config.weight_intensity +
            keyword_score * self.config.weight_keywords +
            trend_score * self.config.weight_trend +
            history_score * self.config.weight_history)

        if risk_score >= self.config.threshold_critical:
            level = 'critical'
        elif risk_score >= self.config.threshold_warning:
            level = 'warning'
        elif risk_score >= self.config.threshold_attention:
            level = 'attention'
        else:
            level = 'safe'

        self.add_emotion_record(current_emotion, current_intensity, user_text)

        if level in ('critical', 'warning'):
            self.user_profile['last_crisis'] = time.time()
            self.user_profile['crisis_count'] += 1
            self.crisis_history.append({
                'timestamp': time.time(), 'risk_score': risk_score, 'level': level,
                'emotion': current_emotion, 'intensity': current_intensity,
                'user_text': user_text[:100] if user_text else ''})

        return {
            'risk_score': round(risk_score, 3), 'level': level,
            'is_crisis': level in ('critical', 'warning'),
            'components': {
                'intensity': {'score': round(intensity_score, 3), 'weight': self.config.weight_intensity},
                'keywords': {'score': round(keyword_score, 3), 'weight': self.config.weight_keywords},
                'trend': {'score': round(trend_score, 3), 'weight': self.config.weight_trend},
                'history': {'score': round(history_score, 3), 'weight': self.config.weight_history},
            },
            'recommendation': self._get_recommendation(level),
            'suggested_action': self._get_suggested_action(level, current_emotion),
        }

    def _get_recommendation(self, level: str) -> str:
        recommendations = {
            'critical': '【紧急】立即启动危机干预协议',
            'warning': '【警告】加强陪伴频率，增加互动性安慰',
            'attention': '【关注】保持高频关注',
            'safe': '【正常】继续正常陪伴',
        }
        return recommendations.get(level, '')

    def _get_suggested_action(self, level: str, emotion: str) -> str:
        actions = {
            'critical': 'emergency_comfort',
            'warning': 'intensive_comfort',
            'attention': 'keep_attention',
            'safe': 'normal',
        }
        return actions.get(level, 'normal')

    def get_user_report(self) -> Dict:
        recent = list(self.emotion_history)[-10:]
        return {
            'profile': self.user_profile.copy(),
            'recent_emotions': [
                {'emotion': r.emotion, 'intensity': r.intensity,
                 'time': datetime.fromtimestamp(r.timestamp).strftime('%H:%M')}
                for r in recent],
            'crisis_count': len(self.crisis_history),
            'avg_intensity': sum(r.intensity for r in self.emotion_history) / len(self.emotion_history) if self.emotion_history else 0,
            'current_trend': 'worsening' if self._calculate_trend_score() > 0.5 else 'stable',
        }

    def reset(self):
        self.emotion_history.clear()
        self.crisis_history.clear()
        self.session_start = None
        self.session_emotion_records = []


# 危机响应话术库
CRISIS_COMFORT_PHRASES = {
    'critical': [
        "朋友，我注意到你现在可能非常痛苦。我想告诉你，你的生命很珍贵。",
        "我知道你现在很难受，但请相信，总有人在乎你，愿意陪你度过。",
        "无论发生什么，你的存在对这个世界很重要。让我们一起想办法。",
    ],
    'warning': [
        "我感觉到你现在的情绪很强烈，请让我陪着你。",
        "痛苦不会永远持续，我会在这里一直支持你。",
        "你不需要一个人面对，我们一起慢慢来。",
    ],
    'attention': [
        "我在这里认真倾听你，告诉我更多好吗？",
        "谢谢你愿意告诉我这些，我会一直陪着你。",
    ],
}

CRISIS_SUGGESTED_MOTIONS = {
    'critical': ['H_Str_B', 'H_Rise_B'],
    'warning': ['H_Wave_B', 'H_Bec_B'],
    'attention': ['H_Wave_R', 'H_Bec_L'],
    'safe': ['RaiseRightHand', 'Victory'],
}


def get_crisis_response(level: str) -> Dict:
    return {
        'phrases': CRISIS_COMFORT_PHRASES.get(level, CRISIS_COMFORT_PHRASES['safe']),
        'motions': CRISIS_SUGGESTED_MOTIONS.get(level, CRISIS_SUGGESTED_MOTIONS['safe']),
    }
