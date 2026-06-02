#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CBT认知行为疗法对话引擎
状态机：IDENTIFY → CHALLENGE → RESTRUCTURE → CLOSE
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from core.deepseek import _call_deepseek


@dataclass
class CognitiveDistortion:
    """认知扭曲类型"""
    id: str
    name_cn: str
    description: str
    keywords: List[str]
    example: str
    challenge_template: str


# 10种认知扭曲（基于Beck认知治疗理论）
COGNITIVE_DISTORTIONS = [
    CognitiveDistortion(
        id="all_or_nothing", name_cn="非黑即白",
        description="用非此即彼的方式看待事物，没有中间地带",
        keywords=["肯定", "绝对", "永远", "从来", "从不", "总是", "一定", "必须", "完全", "彻底"],
        example="我考试肯定要挂",
        challenge_template="你说「{keyword}」，事情真的只有这两种可能吗？有没有中间的情况？"
    ),
    CognitiveDistortion(
        id="catastrophizing", name_cn="灾难化",
        description="自动预期最坏的结果，把小问题放大成灾难",
        keywords=["完了", "糟了", "完蛋", "死定了", "没救了", "彻底完了", "全完了", "这下完了"],
        example="这次面试没过，我这辈子就完了",
        challenge_template="你想到的是最坏的情况，但最可能发生的是什么呢？"
    ),
    CognitiveDistortion(
        id="overgeneralization", name_cn="过度泛化",
        description="从一次经历得出普遍性结论，一个例子代表全部",
        keywords=["每次", "总是这样", "谁都", "都不", "没人", "永远都", "从来都"],
        example="每次面试都被拒",
        challenge_template="你说「{keyword}」，是每一次都这样吗？有没有例外？"
    ),
    CognitiveDistortion(
        id="mental_filter", name_cn="心理过滤",
        description="只关注负面细节，忽略积极的部分",
        keywords=["但是", "可是", "只是", "可惜", "虽然好", "就算"],
        example="老师表扬了我，但那只是客套话",
        challenge_template="我注意到你特别关注了负面的一面，有没有被你忽略的积极部分？"
    ),
    CognitiveDistortion(
        id="disqualifying", name_cn="否定积极",
        description="把积极经历变得不值一提，不承认自己的成功",
        keywords=["不算", "不算什么", "碰巧", "运气", "侥幸", "没什么", "不是我的功劳", "那只是"],
        example="通过了只是运气好",
        challenge_template="你说那只是运气，你自己的努力在其中占了多少呢？"
    ),
    CognitiveDistortion(
        id="mind_reading", name_cn="读心术",
        description="武断地认为别人在想什么，没有证据就下结论",
        keywords=["觉得他们", "肯定觉得", "一定觉得", "肯定认为", "都看不起", "肯定在", "一定在笑"],
        example="他们肯定在背后笑我",
        challenge_template="你觉得别人在那样想，有什么具体的证据吗？"
    ),
    CognitiveDistortion(
        id="fortune_telling", name_cn="算命式预测",
        description="确信事情会变糟，但没有依据",
        keywords=["肯定会", "一定会", "注定", "迟早", "结果还是", "到头来"],
        example="我这辈子注定一事无成",
        challenge_template="你说事情「{keyword}」会那样，你有多少把握？有没有别的可能性？"
    ),
    CognitiveDistortion(
        id="emotional_reasoning", name_cn="情绪化推理",
        description="因为感觉如此，就认为一定如此",
        keywords=["感觉", "觉得我就是", "感觉自己", "我就是个", "我好像就是"],
        example="我感觉自己很没用，所以我一定很没用",
        challenge_template="感觉和事实有时候不是一回事。你感觉到的，有多少是事实支撑的？"
    ),
    CognitiveDistortion(
        id="should_statements", name_cn="应该思维",
        description="用「应该」「必须」要求自己或他人，做不到就自责",
        keywords=["应该", "必须", "不该", "本该", "早就应该", "本来应该", "怎么能不"],
        example="我应该更努力才对",
        challenge_template="你说「{keyword}」，这个「应该」是谁定的标准？如果没做到，真的就那么不可接受吗？"
    ),
    CognitiveDistortion(
        id="labeling", name_cn="贴标签",
        description="给自己或他人贴上固定的负面标签",
        keywords=["我就是个", "我是个", "我这种人", "废物", "失败者", "没用的人", "笨蛋", "loser"],
        example="我就是个废物",
        challenge_template="你给自己贴了这个标签，但一个人真的能被一个标签定义吗？"
    ),
]

# 关键词→认知扭曲倒排索引
_DISTORTION_KEYWORD_INDEX: Dict[str, List[CognitiveDistortion]] = {}


def _build_keyword_index():
    global _DISTORTION_KEYWORD_INDEX
    if _DISTORTION_KEYWORD_INDEX:
        return
    for d in COGNITIVE_DISTORTIONS:
        for kw in d.keywords:
            if kw not in _DISTORTION_KEYWORD_INDEX:
                _DISTORTION_KEYWORD_INDEX[kw] = []
            _DISTORTION_KEYWORD_INDEX[kw].append(d)


class CBTPhase:
    IDENTIFY = "identify"
    CHALLENGE = "challenge"
    RESTRUCTURE = "restructure"
    CLOSE = "close"


@dataclass
class CBTSession:
    phase: str = CBTPhase.IDENTIFY
    emotion: str = ""
    user_text: str = ""
    detected_distortions: List[CognitiveDistortion] = field(default_factory=list)
    primary_distortion: Optional[CognitiveDistortion] = None
    automatic_thought: str = ""
    challenge_history: List[str] = field(default_factory=list)
    user_responses: List[str] = field(default_factory=list)
    round_count: int = 0
    max_rounds: int = 5
    start_time: float = 0.0

    def __post_init__(self):
        if not self.start_time:
            self.start_time = time.time()


class CBTEngine:
    """CBT引擎：should_activate → start_session → generate_next_response → close"""

    def __init__(self):
        _build_keyword_index()
        self.session: Optional[CBTSession] = None

    def should_activate(self, emotion: str, user_text: str) -> bool:
        """中高强度(5-8) + 语言含认知扭曲关键词 → 激活"""
        from config import EMOTION_INTENSITY
        intensity = EMOTION_INTENSITY.get(emotion, 5)
        if intensity < 5 or intensity > 8:
            return False
        return len(self._detect_distortions(user_text)) > 0

    def get_activation_confidence(self, emotion: str, user_text: str) -> float:
        """CBT激活置信度 (0~1)"""
        from config import EMOTION_INTENSITY
        intensity = EMOTION_INTENSITY.get(emotion, 5)
        if intensity < 5 or intensity > 8:
            return 0.0

        distortions = self._detect_distortions(user_text)
        if not distortions:
            return 0.0

        base = min(len(distortions) / 3.0, 1.0) * 0.6
        if 6 <= intensity <= 7:
            base += 0.3
        elif intensity == 5 or intensity == 8:
            base += 0.1
        return min(base, 1.0)

    def _detect_distortions(self, text: str) -> List[CognitiveDistortion]:
        """从用户语言检测认知扭曲，按匹配优先级排序"""
        _build_keyword_index()
        text_lower = text.lower()
        matched = {}

        for kw, distortions in _DISTORTION_KEYWORD_INDEX.items():
            if kw in text_lower:
                for d in distortions:
                    if d.id not in matched:
                        matched[d.id] = {"distortion": d, "count": 0, "max_kw_len": 0}
                    matched[d.id]["count"] += 1
                    matched[d.id]["max_kw_len"] = max(matched[d.id]["max_kw_len"], len(kw))

        if not matched:
            return []
        sorted_items = sorted(matched.values(), key=lambda x: (x["count"], x["max_kw_len"]), reverse=True)
        return [item["distortion"] for item in sorted_items]

    def start_session(self, emotion: str, user_text: str) -> CBTSession:
        distortions = self._detect_distortions(user_text)
        primary = distortions[0] if distortions else None
        self.session = CBTSession(
            phase=CBTPhase.IDENTIFY, emotion=emotion, user_text=user_text,
            detected_distortions=distortions, primary_distortion=primary,
            automatic_thought=user_text, round_count=0,
        )
        print(f"  [CBT] 会话启动 | 情绪: {emotion} | 扭曲: {[d.name_cn for d in distortions]}")
        return self.session

    def end_session(self):
        if self.session:
            print(f"  [CBT] 会话结束 | 轮次: {self.session.round_count}")
        self.session = None

    @property
    def is_active(self) -> bool:
        return self.session is not None and self.session.phase != CBTPhase.CLOSE

    def generate_next_response(self, user_reply: str = "") -> Tuple[str, str]:
        """生成下一轮CBT对话，返回 (text, phase)"""
        if not self.session:
            return "出了点问题，我们换个话题聊聊吧。", CBTPhase.CLOSE

        self.session.round_count += 1
        self.session.user_responses.append(user_reply)

        if self.session.round_count > self.session.max_rounds:
            return self._generate_close(), CBTPhase.CLOSE

        if self.session.phase == CBTPhase.IDENTIFY:
            text, _ = self._generate_identify(user_reply)
        elif self.session.phase == CBTPhase.CHALLENGE:
            text, _ = self._generate_challenge(user_reply)
        elif self.session.phase == CBTPhase.RESTRUCTURE:
            text, _ = self._generate_restructure(user_reply)
        elif self.session.phase == CBTPhase.CLOSE:
            return self._generate_close(), CBTPhase.CLOSE
        else:
            text = "我们换个话题聊聊吧。"
            self.session.phase = CBTPhase.CLOSE

        return text, self.session.phase

    def _generate_identify(self, user_reply: str) -> Tuple[str, str]:
        distortion = self.session.primary_distortion
        if self.session.round_count == 1:
            if distortion:
                prompt = f"""用户情绪是「{self.session.emotion}」，说了：「{self.session.user_text}」
你识别到用户可能存在「{distortion.name_cn}」的认知扭曲（{distortion.description}）。

请用一句话帮助用户看到自己的这个思维模式。要求：
- 不超过40字，温柔地说"你有没有发现，你刚才在想..."
- 不是指责，是帮他自己看见
- 语气像朋友轻轻点了一下，不是心理医生在诊断
- 不要说"认知扭曲""非理性"等专业术语
只返回这句话。"""
            else:
                prompt = f"""用户情绪是「{self.session.emotion}」，说了：「{self.session.user_text}」
请用一句温柔的话帮用户看到自己的自动思维。
- 不超过40字，不说专业术语，像朋友轻轻提醒
只返回这句话。"""

            result = _call_deepseek(prompt, max_tokens=80, temp=0.7)
            text = result or "你有没有发现，你刚才好像在往最坏的方向想？"
            self.session.phase = CBTPhase.CHALLENGE
            return text, CBTPhase.IDENTIFY

        self.session.phase = CBTPhase.CHALLENGE
        return "你觉得这个想法有多真实？", CBTPhase.IDENTIFY

    def _generate_challenge(self, user_reply: str) -> Tuple[str, str]:
        distortion = self.session.primary_distortion

        challenge_history_str = ""
        if self.session.challenge_history:
            challenge_history_str = f"\n\n之前你问过的（绝对不能重复）：\n" + "\n".join(
                f"  第{i+1}次: {q}" for i, q in enumerate(self.session.challenge_history))

        user_history_str = ""
        valid_responses = [r for r in self.session.user_responses if r.strip()]
        if valid_responses:
            user_history_str = f"\n\n用户之前回答过：\n" + "\n".join(f"  用户: {r}" for r in valid_responses[-3:])

        if distortion:
            prompt = f"""你是"心伴"，正在进行CBT质疑阶段。

用户情绪: {self.session.emotion}
用户最初说: {self.session.user_text}
认知扭曲: {distortion.name_cn}（{distortion.description}）

任务：用苏格拉底式提问引导用户发现思维偏差。{challenge_history_str}{user_history_str}

要求：
- 只问一个问题，≤35字
- 像朋友好奇地问，不是审问
- 不要机械套模板，根据用户回答自然追问
- 不说专业术语，不重复之前问过的
只返回这个问题。"""
        else:
            prompt = f"""用户情绪: {self.session.emotion}
用户说: {self.session.user_text}
用户刚回答: {user_reply}

请用温柔的追问帮用户重新看待想法。≤35字，只问一个问题。只返回这个问题。"""

        result = _call_deepseek(prompt, max_tokens=80, temp=0.8)
        text = result or "有没有什么证据支持你刚才的想法呢？"
        self.session.challenge_history.append(text)

        # 检查思维松动 → 进入重建
        if self.session.round_count >= 3:
            loosening_signals = ["也许", "可能", "其实", "好像", "不确定", "不一定",
                         "也许吧", "说得也是", "你说的有道理", "没想过", "换个角度",
                         "确实", "想想", "回头", "仔细想"]
            if user_reply and any(s in user_reply for s in loosening_signals):
                self.session.phase = CBTPhase.RESTRUCTURE
                print(f"  [CBT] 检测到思维松动，进入重建")

        if self.session.round_count >= 4 and self.session.phase != CBTPhase.RESTRUCTURE:
            self.session.phase = CBTPhase.RESTRUCTURE
            print(f"  [CBT] 质疑轮次达上限，进入重建")

        return text, CBTPhase.CHALLENGE

    def _generate_restructure(self, user_reply: str) -> Tuple[str, str]:
        distortion = self.session.primary_distortion
        valid_responses = [r for r in self.session.user_responses if r.strip()]
        user_history_str = "\n".join(f"  用户: {r}" for r in valid_responses[-3:]) if valid_responses else ""

        prompt = f"""你是"心伴"，正在进行CBT认知重建。

用户情绪: {self.session.emotion}
最初说: {self.session.user_text}
扭曲: {distortion.name_cn if distortion else '未明确'}
质疑过程中用户说过的:
{user_history_str}

帮用户形成更平衡的替代性思维。≤50字，先肯定觉察再温和引导，不说专业术语。只返回这句话。"""

        result = _call_deepseek(prompt, max_tokens=100, temp=0.8)
        text = result or "也许事情不是只有那一种可能，换个角度看也许会不一样。"
        self.session.phase = CBTPhase.CLOSE
        return text, CBTPhase.RESTRUCTURE

    def _generate_close(self) -> str:
        distortion = self.session.primary_distortion
        prompt = f"""用户刚完成了一次自我探索。
最初说: {self.session.user_text}
扭曲: {distortion.name_cn if distortion else '未明确'}

说一句温暖的收尾话，≤30字，肯定觉察力，像朋友拍拍肩。只返回这句话。"""

        result = _call_deepseek(prompt, max_tokens=60, temp=0.8)
        text = result or "你能看到自己的想法模式，这本身就很了不起。"
        self.session.phase = CBTPhase.CLOSE
        return text


# 全局实例 + 快捷接口
cbt_engine = CBTEngine()

def should_use_cbt(emotion: str, user_text: str) -> bool:
    return cbt_engine.should_activate(emotion, user_text)

def start_cbt_session(emotion: str, user_text: str) -> CBTSession:
    return cbt_engine.start_session(emotion, user_text)

def get_cbt_response(user_reply: str = "") -> Tuple[str, str]:
    return cbt_engine.generate_next_response(user_reply)

def end_cbt_session():
    cbt_engine.end_session()

def is_cbt_active() -> bool:
    return cbt_engine.is_active

def get_cbt_session() -> Optional[CBTSession]:
    return cbt_engine.session


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("CBT引擎 - 独立测试")
    print("--- 认知扭曲检测 ---")
    test_texts = [
        "我考试肯定要挂", "每次面试都被拒，我这辈子完了",
        "老师表扬我但那只是客套话", "我感觉自己很没用",
        "我应该更努力才对", "他们肯定在背后笑我", "今天天气不错",
    ]
    for text in test_texts:
        distortions = cbt_engine._detect_distortions(text)
        names = [d.name_cn for d in distortions]
        confidence = cbt_engine.get_activation_confidence("焦虑", text)
        print(f"  「{text}」→ 扭曲: {names} | 置信度: {confidence:.2f}")
