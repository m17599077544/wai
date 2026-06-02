#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪检测：关键词置信度 → 编辑距离模糊匹配 → DeepSeek 兜底"""
import re
from config import EMOTION_KEY_WORDS, ALL_EMOTIONS, STOP_WORDS, MOTION_CN_MAP, DANCE_MOTION_POOL
from core.deepseek import _call_deepseek
from core.edit_distance import fuzzy_match, fuzzy_match_emotion

CONFIDENCE_THRESHOLD = 0.6

# 舞名关键词→motion名映射（用于"跳支XX舞"识别具体舞蹈）
DANCE_NAME_MAP = {}
for _motion_name in DANCE_MOTION_POOL:
    _cn = MOTION_CN_MAP.get(_motion_name, _motion_name)
    DANCE_NAME_MAP[_cn] = _motion_name
    _kw = _cn.rstrip("舞")
    if _kw and _kw != _cn:
        DANCE_NAME_MAP[_kw] = _motion_name
    DANCE_NAME_MAP[_motion_name] = _motion_name


def _detect_emotion_keyword_with_confidence(text):
    """关键词检测，返回(情绪, 置信度)。1词=0.7，≥2词=0.9"""
    tl = text.lower().strip()
    best_emo = None
    best_count = 0
    
    for emo, kws in EMOTION_KEY_WORDS.items():
        count = sum(1 for kw in kws if kw in tl)
        if count > best_count:
            best_count = count
            best_emo = emo
    
    if best_emo:
        confidence = 0.9 if best_count >= 2 else 0.7
        return best_emo, confidence
    return None, 0.0


def _detect_emotion_keyword(text):
    emo, _ = _detect_emotion_keyword_with_confidence(text)
    return emo


def _is_stop(text):
    """结束词检测，排除好转表述"""
    improvement_patterns = ["好了很多", "好多了", "好一些", "好一点", "好不少", "好了吧",
                            "好些了", "好点了", "好很多了", "好一些了", "好了一点"]
    for p in improvement_patterns:
        if p in text:
            return False
    for w in STOP_WORDS:
        if w in text:
            return True
    return False


def detect_emotion(text):
    """三级情绪检测：关键词置信度 → 模糊匹配 → DeepSeek"""
    if _is_stop(text): return "__STOP__"
    
    # 1. 关键词置信度
    kw, kw_confidence = _detect_emotion_keyword_with_confidence(text)
    if kw and kw_confidence >= CONFIDENCE_THRESHOLD:
        print(f"  [情绪检测] 关键词置信度 {kw_confidence:.1f} >= 阈值{CONFIDENCE_THRESHOLD:.1f}，直接返回: {kw}")
        return kw
    
    # 2. 编辑距离模糊匹配
    fuzzy_emo_result = fuzzy_match_emotion(text)
    if fuzzy_emo_result:
        fuzzy_emo, fuzzy_sim = fuzzy_emo_result
        fuzzy_confidence = 0.6 + fuzzy_sim * 0.2
        if fuzzy_confidence >= CONFIDENCE_THRESHOLD:
            print(f"  [情绪检测] 编辑距离模糊匹配: {fuzzy_emo} (相似度{fuzzy_sim:.2f}, 置信度{fuzzy_confidence:.2f})")
            return fuzzy_emo
    
    # 3. DeepSeek
    emotions_str = "、".join(ALL_EMOTIONS)
    system = f"""你是一位深谙心理学的情绪分析专家，擅长从日常语言中捕捉微妙的情绪信号。
可选情绪类别（{len(ALL_EMOTIONS)}种），请严格区分：
【高强度9-10】绝望、崩溃、心碎、痛苦 —— 极度负面，需要高度关注
【中高强度7-8】悲伤、失恋、愤怒、恐惧、委屈、无助 —— 明显痛苦，但尚有自知力
【中强度5-7】焦虑、孤独、迷茫、挫败、自卑、内耗、疲惫、后悔、嫉妒、思念、压力大、失落、沮丧
【低强度3-5】心烦、无奈、孤单、麻木 —— 轻微不适，情绪波动
【正面2-3】开心、平静、满足、期待、放松

分析要点：
- 注意语境和程度词，如"好多了""好一些"→正面情绪，而非"好了"→结束
- "想太多"倾向内耗而非焦虑；"睡不着"倾向焦虑而非疲惫
- "一个人"结合上下文判断是孤独还是客观陈述
- 区分"后悔"（对过去）与"焦虑"（对未来）与"内耗"（反复纠结）
只返回一个最贴切的情绪词，不要解释，不要加标点。"""
    result = _call_deepseek(f'用户说："{text}"\n情绪是什么？', system=system, max_tokens=20, temp=0.2)
    
    if result:
        e = result.strip().strip('"')
        if e in ALL_EMOTIONS: 
            print(f"  [情绪检测] DeepSeek返回: {e} (置信度0.5)")
            return e
        for known in ALL_EMOTIONS:
            if known in e or e in known: 
                print(f"  [情绪检测] DeepSeek匹配: {known} (置信度0.5)")
                return known
    
    return None


def detect_intent(text, current_emotion=None):
    """意图检测：关键词 → 模糊匹配 → DeepSeek
    
    返回：("emotion", emotion, text) | ("play_song", song_name) | ("play_music",) 
          | ("dance",) | ("pushup",) | ("demand",) | ("stop",) | ("continue",)
    """
    text_lower = text.lower().strip()
    
    # 0. 结束意图快速路径
    if _is_stop(text):
        print(f"  [意图检测] 结束意图（结束词匹配）")
        return ("stop",)
    
    # 1. 情绪关键词
    kw_emotion, kw_confidence = _detect_emotion_keyword_with_confidence(text)
    if kw_emotion and kw_emotion in ALL_EMOTIONS:
        if kw_emotion != current_emotion:
            print(f"  [意图检测] 情绪关键词: {kw_emotion} (置信度{kw_confidence:.1f})")
            return ("emotion", kw_emotion, text)
    
    # 2. 表演关键词
    # 2a. "XX唱的YY" → 提取 YY 为歌名（处理"周杰伦唱的稻香"）
    m_singer = re.search(r'(.+?)唱的(.{2,8})', text_lower)
    if m_singer and not _is_stop(text):
        song_name = m_singer.group(2).strip().rstrip('的')
        if song_name and len(song_name) >= 2:
            print(f"  [意图检测] 播放指定歌曲: {song_name}（从'{m_singer.group(1)}唱的'提取）")
            return ("play_song", song_name)

    # 先匹配"指定歌曲"模式（含歌名），再匹配"随机放歌"模式
    _BAD_SONGS = ('一下', '一下歌', '音乐', '首歌', '一首', '首好听',
                  '别的', '不一样的', '不一样', '其他的', '换首', '随便', '随便一首',
                  '好听的', '一首好听的', '歌吧', '歌啊', '好听', '别的歌', '好听歌')
    sing_patterns = [
        r'唱[首|个|支|曲]?(.+?)[歌|曲]', r'唱(.+?)[歌|曲]', r'放[一|首]?[个]?(.+?)[歌|曲]',
        r'(?:放|来|播)首(.+?)(?:给|让|帮|[我你他]|听|唱|播|\Z)',
        r'(?:放|来|播)一首(.+?)(?:给|让|帮|[我你他]|听|唱|播|\Z)',
        r'(?:给|让|帮)我(?:放|播|来|听)首?(.+?)(?:给|让|帮|[我你他]|[，。！？,.!?\s]|\Z)',
        r'(?:放|播|来|听)(.+?)(?:给|让|帮)[我你]',
        r'想听(.{2,}?)(?:的?歌|[，。！？,.!?\s]|\Z)',
        r'播放一首(.{2,}?)(?:给|让|帮|[我你他]|[，。！？,.!?\s]|\Z)',
        r'播放(.{2,}?)(?:的?歌|[，。！？,.!?\s]|\Z)',
        r'放一下(.{2,}?)(?:给|让|帮|[我你他]|[，。！？,.!?\s]|\Z)',
    ]
    for pattern in sing_patterns:
        match = re.search(pattern, text_lower)
        if match:
            song_name = match.group(1).strip()
            # "周杰伦的歌" -> "周杰伦"，去掉末尾"的"再判断
            if song_name.endswith('的'):
                song_name = song_name[:-1].strip()
            if song_name and len(song_name) >= 2 and song_name not in _BAD_SONGS:
                print(f"  [意图检测] 播放指定歌曲: {song_name}")
                return ("play_song", song_name)
    
    if any(kw in text_lower for kw in ["放歌", "放音乐", "听歌", "听音乐", "播歌", "播音乐", "播放音乐", "来首歌", "放首歌"]):
        print(f"  [意图检测] 放音乐（随机）")
        return ("play_music",)

    # 2a. 检测具体舞名（"跳支起飞舞""跳个小苹果舞""跳江南style"等）
    for kw, motion_name in DANCE_NAME_MAP.items():
        if kw in text_lower:
            print(f"  [意图检测] 跳指定舞蹈: {motion_name}（从'{kw}'匹配）")
            return ("dance", motion_name)

    if any(kw in text_lower for kw in ["跳舞", "跳个舞", "舞蹈", "跳支舞", "表演舞", "跳一段舞", "跳一首舞"]):
        print(f"  [意图检测] 跳舞（随机）")
        return ("dance",)
    
    if any(kw in text_lower for kw in ["俯卧撑", "做俯卧撑", "做个俯卧撑", "做一个俯卧撑", "做运动", "运动一下", "来几个俯卧撑", "做几个俯卧撑"]):
        print(f"  [意图检测] 俯卧撑")
        return ("pushup",)
    
    if any(kw in text_lower for kw in ["看表演", "表演", "想看", "想表演", "来一个"]):
        print(f"  [意图检测] 通用表演需求")
        return ("demand",)
    
    # 2.5 编辑距离模糊匹配
    fuzzy_result = fuzzy_match(text_lower)
    if fuzzy_result:
        intent, sim_score, matched_kw = fuzzy_result
        print(f"  [意图检测] 编辑距离模糊匹配: 意图={intent}, 相似度={sim_score:.2f}, 匹配词={matched_kw}")
        if intent == "play_song":
            return ("play_song", None)
        elif intent == "play_music":
            return ("play_music",)
        elif intent == "dance":
            return ("dance",)
        elif intent == "pushup":
            return ("pushup",)
        elif intent == "demand":
            return ("demand",)
        elif intent == "stop":
            print(f"  [意图检测] 结束意图（模糊匹配）")
            return ("stop",)
    
    # 3. DeepSeek
    print(f"  [意图检测] 无关键词，调用DeepSeek分析...")
    system = f"""你是一位意图识别专家。用户正在与机器人进行情绪陪伴对话。
当前用户情绪状态：{current_emotion or '未知'}

分析用户下一句的意图，返回格式：
- emotion_xxx（切换到情绪xxx，如emotion_孤单、emotion_开心）
- play_song:歌名（用户想听特定歌曲，返回歌名，如 play_song:稻香、play_song:晴天）
- play_music（用户想放音乐，不指定具体歌名，如"放首歌""来点音乐"）
- dance（用户想看跳舞）
- pushup（用户想做俯卧撑/运动）
- demand（用户想看表演但不明确类型）
- stop（用户想结束对话，如不用了/谢谢/再见/可以了）
- continue（用户只是正常回应，继续对话）

注意：
- "放首周杰伦唱的稻香" → play_song:稻香（歌名是稻香，不是周杰伦）
- "播放一首歌" → play_music（没指定歌名）
- 只返回一行结果，不要解释。"""

    result = _call_deepseek(f'用户说："{text}"\n意图是什么？', system=system, max_tokens=50, temp=0.3)
    if result:
        result = result.strip()
        print(f"  [意图检测] DeepSeek返回: {result}")

        if result.lower().startswith("emotion_"):
            emo = result.lower().replace("emotion_", "").strip()
            if emo in ALL_EMOTIONS:
                return ("emotion", emo, text)
        elif result.lower().startswith("play_song"):
            # 提取歌名: play_song:歌名
            if ":" in result:
                song = result.split(":", 1)[1].strip()
                if song and len(song) >= 2:
                    return ("play_song", song)
            return ("play_song", None)
        elif "play_music" in result or "音乐" in result:
            return ("play_music",)
        elif "dance" in result or "跳舞" in result:
            return ("dance",)
        elif "pushup" in result or "运动" in result:
            return ("pushup",)
        elif "demand" in result or "表演" in result:
            return ("demand",)
        elif "stop" in result or "结束" in result:
            print(f"  [意图检测] 结束意图（DeepSeek判定）")
            return ("stop",)
    
    return ("continue",)


def check_if_user_wants_to_stop(text):
    """结束意图判断（结束词快速路径 + DeepSeek 兜底），用于连续表演时检测退出意愿"""
    if _is_stop(text):
        return True
    
    system = """你是一个对话结束意图判断专家。
判断用户是否想结束当前对话。

用户可能说：
- 直接说再见/不聊了/就这样/结束吧
- 表示累了/困了/想休息
- 表达不需要陪伴了/可以了/够了
- 问"还有事吗""可以了吗"等询问是否结束
- 只是简单回应但不想继续

但要注意：
- "好多了""好一些"是好转，不是结束
- "再来一次""再表演一个"是想继续，不是结束
- 正常回应问题不是想结束

只返回：yes（想结束）或 no（不想结束），不要解释。"""
    
    result = _call_deepseek(f'用户说："{text}"\n用户想结束对话吗？', system=system, max_tokens=10, temp=0.1)
    if result:
        result = result.strip().lower()
        print(f"  [结束意图检测] DeepSeek返回: {result}")
        if "yes" in result or "是" in result or "对" in result:
            return True
    
    return False
