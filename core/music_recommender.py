#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多维音乐推荐引擎：情绪×风格×时段×去重"""
import random
from datetime import datetime

# ── 情绪→音乐多维映射（tags/genres均为纯词无空格）──
EMOTION_MUSIC_MAP = {
    "绝望": {
        "tags": ["励志", "重生", "不屈", "破晓", "逆袭"],
        "genres": ["流行", "民谣", "摇滚"],
        "tempo": "中速渐强",
        "style": "力量感",
    },
    "崩溃": {
        "tags": ["释放", "治愈", "安抚", "平复", "拥抱"],
        "genres": ["流行", "抒情", "民谣"],
        "tempo": "先激烈后舒缓",
        "style": "宣泄转温暖",
    },
    "心碎": {
        "tags": ["自愈", "坚强", "释怀", "重生", "时间"],
        "genres": ["流行", "民谣", "R&B"],
        "tempo": "慢速抒情",
        "style": "治愈系",
    },
    "痛苦": {
        "tags": ["治愈", "陪伴", "温暖", "走出", "黎明"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "舒缓",
        "style": "抚慰",
    },
    "悲伤": {
        "tags": ["治愈", "温暖", "希望", "阳光", "春天"],
        "genres": ["流行", "民谣", "古风"],
        "tempo": "中速抒情",
        "style": "温柔治愈",
    },
    "失恋": {
        "tags": ["自愈", "成长", "放下", "向前", "独立"],
        "genres": ["流行", "民谣", "R&B"],
        "tempo": "中速",
        "style": "成长系",
    },
    "愤怒": {
        "tags": ["平静", "释放", "和解", "自由", "开阔"],
        "genres": ["摇滚", "电子", "说唱"],
        "tempo": "先快后缓",
        "style": "宣泄转平静",
    },
    "恐惧": {
        "tags": ["勇气", "安全", "力量", "守护", "无畏"],
        "genres": ["流行", "民谣", "摇滚"],
        "tempo": "中速坚定",
        "style": "力量感",
    },
    "委屈": {
        "tags": ["共情", "理解", "被看见", "抱抱", "心疼"],
        "genres": ["流行", "抒情", "民谣"],
        "tempo": "舒缓",
        "style": "被理解",
    },
    "无助": {
        "tags": ["陪伴", "力量", "支持", "并肩", "同行"],
        "genres": ["流行", "民谣", "摇滚"],
        "tempo": "中速温暖",
        "style": "陪伴系",
    },
    "焦虑": {
        "tags": ["平静", "放松", "安宁", "深呼吸", "静心"],
        "genres": ["古风", "民谣", "流行"],
        "tempo": "慢速",
        "style": "减压舒缓",
    },
    "孤独": {
        "tags": ["陪伴", "温暖", "共鸣", "不孤单", "灯火"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "中速",
        "style": "温暖陪伴",
    },
    "迷茫": {
        "tags": ["方向", "希望", "启程", "出发", "星光"],
        "genres": ["流行", "民谣", "摇滚"],
        "tempo": "中速递进",
        "style": "启明系",
    },
    "挫败": {
        "tags": ["鼓励", "坚持", "成长", "再来", "逆风"],
        "genres": ["摇滚", "流行", "说唱"],
        "tempo": "中速偏快",
        "style": "鼓舞系",
    },
    "自卑": {
        "tags": ["自信", "接纳", "独特", "闪光", "值得"],
        "genres": ["流行", "民谣", "R&B"],
        "tempo": "中速",
        "style": "自我肯定",
    },
    "内耗": {
        "tags": ["解脱", "轻松", "接纳", "停一停", "自在"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "慢速",
        "style": "放松解压",
    },
    "疲惫": {
        "tags": ["放松", "舒缓", "治愈", "休息", "晚安"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "慢速",
        "style": "休息系",
    },
    "后悔": {
        "tags": ["释怀", "接纳", "成长", "向前", "翻篇"],
        "genres": ["流行", "民谣", "古风"],
        "tempo": "中速",
        "style": "释怀系",
    },
    "嫉妒": {
        "tags": ["自我成长", "接纳", "平和", "专注自己", "自信"],
        "genres": ["流行", "摇滚", "说唱"],
        "tempo": "中速偏快",
        "style": "自我提升",
    },
    "思念": {
        "tags": ["温暖", "回忆", "珍藏", "重逢", "美好"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "慢速抒情",
        "style": "温柔回忆",
    },
    "压力大": {
        "tags": ["放松", "舒缓", "自由", "逃离", "深呼吸"],
        "genres": ["流行", "民谣", "古风"],
        "tempo": "慢速到中速",
        "style": "减压释放",
    },
    "失落": {
        "tags": ["安慰", "温暖", "力量", "重新出发", "曙光"],
        "genres": ["流行", "民谣", "摇滚"],
        "tempo": "中速",
        "style": "重拾力量",
    },
    "沮丧": {
        "tags": ["鼓励", "阳光", "力量", "希望", "振作"],
        "genres": ["流行", "摇滚", "民谣"],
        "tempo": "中速偏快",
        "style": "鼓舞系",
    },
    "心烦": {
        "tags": ["平静", "舒缓", "清心", "放空", "悠然"],
        "genres": ["古风", "民谣", "流行"],
        "tempo": "慢速",
        "style": "静心系",
    },
    "无奈": {
        "tags": ["释怀", "接纳", "随缘", "顺其自然", "自在"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "中速",
        "style": "淡然系",
    },
    "孤单": {
        "tags": ["陪伴", "温暖", "治愈", "星空", "灯火"],
        "genres": ["民谣", "流行", "古风"],
        "tempo": "中速",
        "style": "陪伴温暖",
    },
    "麻木": {
        "tags": ["唤醒", "感受", "温柔", "复苏", "春天"],
        "genres": ["流行", "民谣", "摇滚"],
        "tempo": "中速渐强",
        "style": "唤醒系",
    },
    "开心": {
        "tags": ["快乐", "轻快", "阳光", "幸福", "庆祝"],
        "genres": ["流行", "电子", "摇滚"],
        "tempo": "快速",
        "style": "欢快系",
    },
    "平静": {
        "tags": ["安宁", "放松", "舒适", "惬意", "微风"],
        "genres": ["民谣", "古风", "流行"],
        "tempo": "慢速",
        "style": "宁静系",
    },
    "满足": {
        "tags": ["温暖", "幸福", "感恩", "知足", "美好"],
        "genres": ["流行", "民谣", "古风"],
        "tempo": "中速",
        "style": "温馨系",
    },
    "期待": {
        "tags": ["希望", "阳光", "憧憬", "美好", "启程"],
        "genres": ["流行", "摇滚", "电子"],
        "tempo": "中速偏快",
        "style": "期待系",
    },
    "放松": {
        "tags": ["轻松", "惬意", "舒缓", "舒适", "悠然"],
        "genres": ["民谣", "古风", "流行"],
        "tempo": "慢速",
        "style": "松弛系",
    },
}

# ── 时段推荐策略 ──
TIME_STRATEGY = {
    "morning": {
        "tempo_hint": "清新活力",
        "genre_boost": ["流行", "民谣", "古风"],
        "mood_hint": "新的一天",
    },
    "afternoon": {
        "tempo_hint": "中速节奏",
        "genre_boost": ["流行", "摇滚", "R&B"],
        "mood_hint": "午后时光",
    },
    "evening": {
        "tempo_hint": "温暖抒情",
        "genre_boost": ["民谣", "流行", "古风"],
        "mood_hint": "傍晚温馨",
    },
    "night": {
        "tempo_hint": "安静舒缓",
        "genre_boost": ["民谣", "古风", "流行"],
        "mood_hint": "夜晚安宁",
    },
}

# ── 风格轮换池（每种风格对应补充搜索词，避免同质）──
STYLE_VARIETY_POOL = {
    "力量感": ["不屈", "倔强", "逆袭", "怒放", "烈火"],
    "治愈系": ["温暖", "治愈", "阳光", "春天", "微风"],
    "陪伴系": ["陪伴", "并肩", "同行", "灯火", "不孤单"],
    "宣泄转温暖": ["释放", "呐喊", "破茧", "飞翔", "自由"],
    "宣泄转平静": ["释放", "呐喊", "破茧", "飞翔", "自由"],
    "鼓舞系": ["加油", "振作", "崛起", "向前", "燃烧"],
    "释怀系": ["放下", "翻篇", "随风", "告别", "新生"],
    "宁静系": ["安宁", "静心", "禅意", "山水", "月光"],
    "唤醒系": ["破晓", "重生", "苏醒", "蜕变", "绽放"],
    "欢快系": ["快乐", "派对", "舞动", "阳光", "大笑"],
    "减压释放": ["放空", "深呼吸", "漫步", "逃离", "发呆"],
    "成长系": ["蜕变", "破茧", "新生", "前行", "绽放"],
    "抚慰": ["抚慰", "安心", "依靠", "港湾", "避风"],
    "被理解": ["懂你", "共鸣", "倾诉", "倾听", "诉说"],
    "减压舒缓": ["放空", "深呼吸", "漫步", "逃离", "发呆"],
    "温暖陪伴": ["灯火", "相依", "守候", "牵挂", "惦念"],
    "启明系": ["曙光", "启航", "远方", "星辰", "破晓"],
    "自我肯定": ["闪光", "独特", "骄傲", "自信", "绽放"],
    "放松解压": ["放空", "漫步", "发呆", "慢生活", "闲适"],
    "休息系": ["晚安", "入眠", "摇篮", "星光", "月色"],
    "自我提升": ["超越", "突破", "蜕变", "进化", "逆袭"],
    "温柔回忆": ["珍藏", "余温", "旧时光", "美好", "怀念"],
    "重拾力量": ["曙光", "重新出发", "涅槃", "归来", "崛起"],
    "静心系": ["清心", "空灵", "禅意", "无为", "止水"],
    "淡然系": ["随缘", "云淡风轻", "自在", "闲适", "无执"],
    "陪伴温暖": ["相依", "守候", "牵挂", "惦念", "灯火"],
    "期待系": ["憧憬", "蓄势", "待发", "崭新", "破晓"],
    "温馨系": ["幸福", "甜蜜", "圆满", "知足", "暖阳"],
    "松弛系": ["慵懒", "慢生活", "闲适", "悠然", "漫无目的"],
}

# 默认映射（情绪不在MAP中时使用）
_DEFAULT_MUSIC = {
    "tags": ["治愈", "温暖", "希望", "陪伴"],
    "genres": ["流行", "民谣"],
    "tempo": "中速",
    "style": "治愈系",
}


def _get_time_period():
    """获取当前时段"""
    h = datetime.now().hour
    if 6 <= h < 12:
        return "morning"
    elif 12 <= h < 18:
        return "afternoon"
    elif 18 <= h < 22:
        return "evening"
    else:
        return "night"


class MusicRecommender:
    """多维音乐推荐器

    推荐维度:
    1. 情绪维度: 情绪→风格标签+曲风+节奏
    2. 时段维度: 早晨/下午/傍晚/深夜→不同节奏偏好
    3. 多样性维度: 风格轮换、标签轮换、曲风轮换
    4. 去重维度: 已播歌曲排除、已用标签降权
    """

    def __init__(self):
        self.played_songs = []
        self.used_tags = []
        self.genre_index = {}   # 每种情绪的曲风轮换指针
        self.tag_index = {}     # 每种情绪的标签轮换指针

    def update_played(self, song_name):
        """记录已播放歌曲"""
        if song_name and song_name not in self.played_songs:
            self.played_songs.append(song_name)
            if len(self.played_songs) > 30:
                self.played_songs = self.played_songs[-30:]

    def _rotate_tags(self, emotion, tags):
        """轮换标签,每次选不同组合"""
        if emotion not in self.tag_index:
            self.tag_index[emotion] = 0
        idx = self.tag_index[emotion] % len(tags)
        self.tag_index[emotion] += 1
        selected = []
        for i in range(2):
            selected.append(tags[(idx + i) % len(tags)])
        return selected

    def _rotate_genre(self, emotion, genres):
        """轮换曲风,每次选不同类型"""
        if emotion not in self.genre_index:
            self.genre_index[emotion] = 0
        idx = self.genre_index[emotion] % len(genres)
        self.genre_index[emotion] += 1
        return genres[idx]

    def _get_variety_tags(self, style):
        """从风格轮换池获取补充标签"""
        pool = STYLE_VARIETY_POOL.get(style, [])
        if not pool:
            return []
        unused = [t for t in pool if t not in self.used_tags[-10:]]
        if not unused:
            unused = pool
        pick = random.choice(unused[:3])
        self.used_tags.append(pick)
        if len(self.used_tags) > 50:
            self.used_tags = self.used_tags[-50:]
        return [pick]

    def build_prompt_params(self, emotion, user_text=""):
        """构建推荐参数

        Returns:
            dict: tags/genre/tempo/style/time_hint/variety_tag/played_hint
        """
        info = EMOTION_MUSIC_MAP.get(emotion, _DEFAULT_MUSIC)
        period = _get_time_period()
        ts = TIME_STRATEGY[period]

        tags = self._rotate_tags(emotion, info["tags"])
        genre = self._rotate_genre(emotion, info["genres"])
        variety = self._get_variety_tags(info.get("style", "治愈系"))

        played_hint = ""
        if self.played_songs:
            played_hint = ",".join(self.played_songs[-8:])

        return {
            "tags": tags,
            "genre": genre,
            "tempo": info["tempo"],
            "style": info.get("style", "治愈系"),
            "time_hint": ts["mood_hint"],
            "variety_tag": variety,
            "played_hint": played_hint,
        }

    def build_deepseek_prompt(self, emotion, user_text=""):
        """构建DeepSeek推荐prompt(多维+多样)"""
        p = self.build_prompt_params(emotion, user_text)
        tags_str = ",".join(p["tags"] + p["variety_tag"])
        genre = p["genre"]
        tempo = p["tempo"]
        time_hint = p["time_hint"]

        played_line = ""
        if p["played_hint"]:
            played_line = f"\n排除已播:{p['played_hint']}"

        prompt = (
            f"输出3首真实华语歌曲名。\n"
            f"情绪:{emotion}|风格:{tags_str}|曲风:{genre}|节奏:{tempo}|时段:{time_hint}"
            f"{played_line}\n"
            f"规则:\n"
            f"1.必须真实存在的知名歌曲\n"
            f"2.每行仅歌名,无歌手/序号/标点/解释\n"
            f"3.3首风格各不相同(如:一首抒情+一首轻快+一首经典)\n"
            f"4.不要重复已播歌曲\n"
            f"\n示例:\n晴天\n夜空中最亮的星\n平凡之路"
        )
        return prompt

    def build_search_keywords(self, emotion):
        """构建直接搜索用的关键词列表(fallback时用)"""
        p = self.build_prompt_params(emotion)
        primary = f"{p['genre']}{p['tags'][0]}"
        secondary = p["variety_tag"][0] if p["variety_tag"] else p["tags"][1]
        return [primary, secondary]


# 全局单例
_recommender = None


def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = MusicRecommender()
    return _recommender


def recommend_music_v2(emotion, user_text="", played_songs=None):
    """新版多维音乐推荐

    Args:
        emotion: 当前情绪
        user_text: 用户输入文本
        played_songs: 已播放歌曲列表(旧接口兼容)

    Returns:
        list[str]: 推荐歌名列表(最多3个)
    """
    import re
    from core.text_gen import _call_deepseek

    rec = get_recommender()

    # 同步已播列表
    if played_songs:
        for s in played_songs:
            rec.update_played(s)

    # ---- 尝试1: DeepSeek多维推荐 ----
    for _try in range(2):
        prompt = rec.build_deepseek_prompt(emotion, user_text)
        text = _call_deepseek(prompt, max_tokens=80, temp=0.9)
        if not text:
            continue

        raw_lines = text.strip().split("\n")
        lines = []
        _skip_kws = (
            "推荐", "情绪", "以下", "为您", "根据", "需要", "需求", "风格",
            "歌曲", "这几", "希望", "适合", "这是一", "下面", "您可以",
            "第", "首", "首是", "首歌",
        )
        for l in raw_lines:
            # 去序号前缀
            l = re.sub(r"^\d+[\.\、\)\）\】\：\:\s]*", "", l)
            # 去歌手名分隔(-/—/~/|等后全部)
            l = re.sub(r"[-\—~|\+]+.*$", "", l)
            # 去括号标注 (Live/现场等)
            l = re.sub(r"[\(\[【].*?[\)\]】]", "", l)
            # 去各类引号和装饰符号
            l = re.sub(r"[\*《》\[\]【】\(\)\"\"\'\'\u201c\u201d\u2018\u2019\u300c\u300d\-\—\~\|\+\s]+", "", l)
            l = l.strip()
            if not l or len(l) < 2 or len(l) > 15:
                continue
            if any(kw in l for kw in _skip_kws):
                continue
                lines.append(l)
        if lines:
            for s in lines[:3]:
                rec.update_played(s)
            print(f"  [音乐推荐] {lines[:3]}")
            return lines[:3]

    # ---- 尝试2: fallback关键词搜索 ----
    kws = rec.build_search_keywords(emotion)
    print(f"  [音乐] DeepSeek无结果,关键词搜索: {kws}")
    return kws
