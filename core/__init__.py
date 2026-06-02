"""核心逻辑：DeepSeek API、情绪检测、文本生成、音乐推荐、ComfortBot主类"""

from core import edit_distance, emotion, emotion_hmm
from core import cbt_engine, cbt_text_gen, crisis_detection
from core import deepface_config, deepseek, text_gen
from core import music_recommender

try:
    from core import comfort_bot
except Exception:
    comfort_bot = None
