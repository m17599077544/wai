#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多轮对话管理：AgentMemory + CoT推理链 + TransformerSemanticEngine + DialogueManager"""
import json, time, os, re, math
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from config import EMOTION_INTENSITY
from core.deepseek import _call_deepseek
from core.emotion_hmm import train_hmm, predict_emotion_trend, predict_next_emotion
from core.sa_style import style_to_prompt_hints
# topic_avoidance: 延迟导入，避免循环依赖
_topic_avoidance_mod = None

def _get_topic_avoidance():
    global _topic_avoidance_mod
    if _topic_avoidance_mod is None:
        import core.topic_avoidance as _m
        _topic_avoidance_mod = _m
    return _topic_avoidance_mod

# rhythm_detector: 延迟导入
_rhythm_mod = None

def _get_rhythm():
    global _rhythm_mod
    if _rhythm_mod is None:
        import core.rhythm_detector as _m
        _rhythm_mod = _m
    return _rhythm_mod

# emotion_fusion: 延迟导入
_fusion_mod = None

def _get_fusion():
    global _fusion_mod
    if _fusion_mod is None:
        import core.emotion_fusion as _m
        _fusion_mod = _m
    return _fusion_mod


# personalized_strategy: 延迟导入
_ucb1_mod = None

def _get_ucb1():
    global _ucb1_mod
    if _ucb1_mod is None:
        import core.personalized_strategy as _m
        _ucb1_mod = _m
    return _ucb1_mod

_MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_memory.json")

_DEFAULT_MEMORY = {
    "emotion_history": [],
    "comfort_effective": [],
    "preferred_music_style": [],
    "total_sessions": 0,
    "max_depth_reached": 0,
    "crisis_history": 0,            # 高强度情绪累计次数
    "last_emotion": None,
    "last_session_time": None,
    "user_notes": [],
}



class AgentMemory:
    """跨会话用户记忆，load/save 持久化到 agent_memory.json"""

    def __init__(self, filepath: str = _MEMORY_FILE):
        self.filepath = filepath
        self.data: dict = {}
        self.dirty = False
        self.load()

    def load(self):
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.data = dict(_DEFAULT_MEMORY)
                self.data.update(loaded)
                print(f"[AgentMemory] 已加载记忆: 历史情绪{len(self.data['emotion_history'])}条"
                      f" | 累计会话{self.data['total_sessions']}次"
                      f" | 上次情绪: {self.data.get('last_emotion', '未知')}")
            else:
                self.data = dict(_DEFAULT_MEMORY)
                print("[AgentMemory] 初次使用，创建新记忆档案")
        except Exception as e:
            print(f"[AgentMemory] 加载失败，使用默认记忆: {e}")
            self.data = dict(_DEFAULT_MEMORY)
        self.dirty = False

    def save(self):
        if not self.dirty:
            return
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            self.dirty = False
        except Exception as e:
            print(f"[AgentMemory] 保存失败: {e}")

    def on_session_start(self):
        self.data["total_sessions"] = self.data.get("total_sessions", 0) + 1
        self.data["last_session_time"] = time.time()
        self.dirty = True

    def update_emotion(self, emotion: str, level: str):
        history = self.data.setdefault("emotion_history", [])
        history.append({"emotion": emotion, "level": level, "ts": time.time()})
        if len(history) > 30:
            self.data["emotion_history"] = history[-30:]
        self.data["last_emotion"] = emotion
        if level == "high":
            self.data["crisis_history"] = self.data.get("crisis_history", 0) + 1
        try:
            train_hmm(history)
        except Exception as e:
            print(f"[AgentMemory] HMM训练异常: {e}")
        self.dirty = True

    def update_depth(self, depth: int):
        if depth > self.data.get("max_depth_reached", 0):
            self.data["max_depth_reached"] = depth
            self.dirty = True

    def add_effective_comfort(self, keyword: str):
        keywords = self.data.setdefault("comfort_effective", [])
        if keyword and keyword not in keywords:
            keywords.append(keyword)
            if len(keywords) > 20:
                self.data["comfort_effective"] = keywords[-20:]
            self.dirty = True

    def add_music_preference(self, style: str):
        styles = self.data.setdefault("preferred_music_style", [])
        if style and style not in styles:
            styles.append(style)
            if len(styles) > 10:
                self.data["preferred_music_style"] = styles[-10:]
            self.dirty = True

    def get_summary_for_prompt(self, current_emotion: str = None) -> str:
        lines = []
        sessions = self.data.get("total_sessions", 0)
        if sessions > 1:
            lines.append(f"这是我们第{sessions}次对话。")
        last_e = self.data.get("last_emotion")
        if last_e:
            lines.append(f"上次对话时，用户的情绪是「{last_e}」。")
        history = self.data.get("emotion_history", [])
        if len(history) >= 3:
            recent = [h["emotion"] for h in history[-3:]]
            lines.append(f"近期情绪轨迹：{'→'.join(recent)}。")
        crisis = self.data.get("crisis_history", 0)
        if crisis > 0:
            lines.append(f"用户曾经历过{crisis}次高强度情绪（崩溃/绝望等），请格外温柔。")
        effective = self.data.get("comfort_effective", [])
        if effective:
            lines.append(f"对这位用户有效的安慰方式包括：{'/'.join(effective[:5])}。")
        depth = self.data.get("max_depth_reached", 0)
        if depth >= 4:
            lines.append(f"这位用户愿意深入交流（历史最大对话深度{depth}轮）。")
        # HMM 趋势预测
        if current_emotion:
            try:
                trend_hint = predict_emotion_trend(current_emotion)
                if trend_hint:
                    lines.append(trend_hint)
                emotion_hint = predict_next_emotion(current_emotion)
                if emotion_hint:
                    lines.append(emotion_hint)
            except Exception:
                pass
        if not lines:
            return "这是与用户的对话，请保持温柔陪伴。"
        return " ".join(lines)

    def get_recent_emotions(self, n: int = 5):
        history = self.data.get("emotion_history", [])
        return [h["emotion"] for h in history[-n:]]


# 全局 AgentMemory 实例
_AGENT_MEMORY = AgentMemory()



def build_cot_system_message(emotion: str, intent: str, stage: str,
                              intensity: int, memory_summary: str = "",
                              sa_hints: str = "") -> str:
    """CoT 推理 + 用户背景 → System Message"""
    stage_desc = {
        "warm_check": "建立安全感，轻轻确认此刻状态，不急着分析或给建议",
        "emotion_labeling": "帮助用户说清楚情绪名字，让他感觉被精准理解",
        "cause_exploration": "温柔追问事情中最关键、最刺痛的一点，引导深入",
        "cognitive_reframe": "缓和自责和绝对化想法，帮用户看到不只是自己的错",
        "resource_activation": "引导用户想起身边支持、过去撑过来的经验或微小力量",
        "future_step": "把注意力落到下一小步，不解决全部，先缓一口气",
        "gentle_space": "尊重用户不想说，给空间，不继续追问，安静陪伴",
        "positive_extend": "承接好转，帮用户保留一点轻松或积极感",
        "crisis_support": "优先安全支持，鼓励立刻联系现实中的人或紧急帮助",
    }.get(stage, "温柔自然地继续陪伴")

    intent_desc = {
        "venting": "用户在宣泄情绪（还没说原因）",
        "cause": "用户在讲述事情原因或经过",
        "helpless": "用户感到无助、不知道怎么办",
        "self_blame": "用户在自责或自我否定",
        "relief": "用户情绪有所好转或表示感谢",
        "avoidance": "用户回避交流，不想深说",
        "crisis": "用户出现危机信号（不想活/自伤念头）",
        "unclear": "用户意图不明确，仅简单回应",
    }.get(intent, "用户意图尚不明确")

    if intensity >= 9:
        intensity_hint = "极度痛苦，需要最温柔、最沉稳的陪伴，避免说教和分析"
    elif intensity >= 7:
        intensity_hint = "明显痛苦，以共情优先，不急于解决问题"
    elif intensity >= 5:
        intensity_hint = "中等情绪，温和陪伴，可适当引导思考"
    elif intensity >= 3:
        intensity_hint = "轻度情绪，轻松自然，可以活泼一些"
    else:
        intensity_hint = "正面情绪，温暖快乐，像朋友聊天"

    system = f"""你是"心伴"，一个有温度、有边界感的情绪陪伴机器人。

【用户背景】
{memory_summary if memory_summary else "这是一次新的对话，请保持温柔陪伴。"}

【当前状态分析】
- 情绪标签：{emotion}（强度 {intensity}/10）
- 当前意图：{intent_desc}
- 强度建议：{intensity_hint}
- 引导阶段：{stage}（{stage_desc}）

【CoT思维链要求】
在生成回复前，先在脑海中完成三步推理（不要输出推理过程，只输出最终一句话）：
步骤一：理解这位用户此刻最需要什么？（被听见？被理解？被引导？还是被保护？）
步骤二：结合引导阶段「{stage}」，最合适的方向是什么？
步骤三：用最简单、最自然的一句口语，把步骤一和步骤二融合成一句话。

__SA_STYLE_HINTS_PLACEHOLDER__

【严格输出规则】
- 只输出一句话，不超过35字
- 口语化，像朋友轻声问或轻声说
- 不说"作为AI/机器人"
- 不说教，不命令，不用"你应该/你必须"
- 如阶段是 gentle_space，可以不是疑问句
- 如阶段是 crisis_support，必须优先关注现实安全和求助渠道"""
    if not sa_hints:
        sa_hints = style_to_prompt_hints()
    av_hint = _get_topic_avoidance().avoidance_hint()
    rh_hint = _get_rhythm().rhythm_hint()
    fu_hint = _get_fusion().fusion_hint()
    system = system.replace("__SA_STYLE_HINTS_PLACEHOLDER__", sa_hints)
    if av_hint:
        system += "\n" + av_hint
    if rh_hint:
        system += "\n" + rh_hint
    if fu_hint:
        system += "\n" + fu_hint
    ucb1_hint = _get_ucb1().strategy_hint(emotion)
    if ucb1_hint:
        system += "\n" + ucb1_hint
    return system


def build_cot_comfort_system(emotion: str, intensity: int, focus: str,
                              memory_summary: str = "", prev_hint: str = "") -> str:
    """构建安慰回应的 CoT System Message（用于 generate_comfort_with_motion）。"""
    if intensity >= 9:
        style_hint = "简洁有力，不煽情，像一双稳稳的手"
    elif intensity >= 7:
        style_hint = "温柔深沉，共情优先，说出他没说出口的感受"
    elif intensity >= 5:
        style_hint = "温和亲切，可以用比喻，轻轻打开一扇窗"
    elif intensity >= 3:
        style_hint = "轻松自然，像老朋友聊天，偶尔幽默"
    else:
        style_hint = "温暖快乐，活泼有感染力"

    prev_part = f"\n之前已说过：{prev_hint}\n必须用完全不同的角度、意象和表达！" if prev_hint else ""

    system = f"""你是"心伴"，融合心理咨询师智慧、文学家温度的陪伴者。

【用户背景】
{memory_summary if memory_summary else "请保持温柔陪伴。"}

【当前情绪】{emotion}（强度 {intensity}/10）
【本轮策略】{focus}
【风格要求】{style_hint}{prev_part}
__SA_STYLE_HINTS_PLACEHOLDER__

【CoT安慰推理链（只输出最终结果，不输出推理）】
步骤一：这位用户此刻最深层的痛苦/需求是什么？
步骤二：「{focus}」这个策略，最合适的切入角度是什么？
步骤三：用一句最自然的话，把上面两步融为一体。

【输出格式（严格）】
第一行：安慰语（≤80字，口语化，不列123，不加引号）
第二行：动作ID（从下面列表中选）

【安慰语禁忌】
- 不说"作为AI"，不说"我建议你"
- 不用"你应该""你需要"，改用"也许""有时候"
- 不煽情泛滥，共情点到即止"""
    system = system.replace("__SA_STYLE_HINTS_PLACEHOLDER__", style_to_prompt_hints())
    return system



class TransformerSemanticEngine:
    """语义引擎，优先本地 text2vec，降级为规则策略"""

    def __init__(self):
        self.available = False
        self.tokenizer = None
        self.model = None
        self.torch = None
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            model_name = os.getenv(
                "TRANSFORMER_MODEL_PATH",
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "text2vec-base-chinese")
            )
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.available = True
            print("[TransformerDialogue] 已启用 Transformer 语义引擎:", model_name)
        except Exception as e:
            self.available = False
            print("[TransformerDialogue] 未启用本地 Transformer，使用规则兜底:", str(e))

    def encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.available or not texts:
            return None
        try:
            with self.torch.no_grad():
                batch = self.tokenizer(texts, padding=True, truncation=True,
                                        max_length=96, return_tensors="pt")
                outputs = self.model(**batch)
                token_embeddings = outputs.last_hidden_state
                attention_mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = self.torch.sum(token_embeddings * attention_mask, dim=1)
                counts = self.torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = summed / counts
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().tolist()
        except Exception as e:
            print("[TransformerDialogue] encode失败，降级规则:", str(e))
            return None

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def similarity(self, a: str, b: str) -> float:
        vectors = self.encode([a, b])
        if vectors:
            return self.cosine(vectors[0], vectors[1])
        return self._rule_similarity(a, b)

    def most_similar_label(self, text: str, label_examples: Dict[str, List[str]]) -> tuple:
        if not text:
            return "unclear", 0.0
        labels = list(label_examples.keys())
        examples = ["；".join(label_examples[label]) for label in labels]
        vectors = self.encode([text] + examples)
        if vectors:
            text_vec = vectors[0]
            best_label = "unclear"
            best_score = -1.0
            for i, label in enumerate(labels):
                score = self.cosine(text_vec, vectors[i + 1])
                if score > best_score:
                    best_score = score
                    best_label = label
            return best_label, best_score
        return self._rule_label(text, label_examples)

    def _rule_similarity(self, a: str, b: str) -> float:
        a_set = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]", a))
        b_set = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]", b))
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / max(len(a_set | b_set), 1)

    def _rule_label(self, text: str, label_examples: Dict[str, List[str]]) -> tuple:
        text = text.strip()
        scores = {}
        keyword_map = {
            "venting": ["烦", "难受", "崩溃", "受不了", "委屈", "生气", "讨厌", "压力", "累", "痛苦"],
            "cause": ["因为", "就是", "发生", "原因", "今天", "刚才", "他们", "他", "她", "工作", "学习", "考试"],
            "helpless": ["不知道", "没办法", "怎么办", "不行", "没用", "撑不住", "走不出来"],
            "self_blame": ["怪我", "都是我", "我不好", "我太差", "我没用", "后悔"],
            "relief": ["好点", "好多了", "还行", "没事了", "谢谢", "舒服点", "可以了"],
            "avoidance": ["不想说", "算了", "没什么", "随便", "不知道说什么"],
            "crisis": ["不想活", "死", "自杀", "结束生命", "活着没意思", "伤害自己"]
        }
        for label in label_examples:
            scores[label] = 0
        for label, kws in keyword_map.items():
            if label in scores:
                scores[label] += sum(1 for kw in kws if kw in text)
        best_label = max(scores, key=scores.get) if scores else "unclear"
        best_score = min(scores.get(best_label, 0) / 3.0, 1.0)
        if best_score <= 0:
            return "unclear", 0.0
        return best_label, best_score


@dataclass
class DialogueTurn:
    role: str
    text: str
    emotion: str = ""
    intent: str = ""
    round_index: int = 0


@dataclass
class DialogueState:
    emotion: str = "平静"
    stage: str = "warm_check"
    round_num: int = 0
    depth: int = 0
    silence_count: int = 0
    repeated_count: int = 0
    last_user_text: str = ""
    last_intent: str = "unclear"
    turns: List[DialogueTurn] = field(default_factory=list)
    used_questions: List[str] = field(default_factory=list)

    def add_turn(self, role: str, text: str, emotion: str = "", intent: str = ""):
        if not text:
            return
        self.turns.append(
            DialogueTurn(role=role, text=text.strip(),
                          emotion=emotion or self.emotion, intent=intent,
                          round_index=self.round_num)
        )
        if len(self.turns) > 24:
            self.turns = self.turns[-24:]


class TransformerDialogueManager:
    """对话状态管理：阶段推进、深度引导、语义去重"""

    def __init__(self):
        self.semantic = TransformerSemanticEngine()
        self.state = DialogueState()
        self._last_cot_system = None

        self.intent_examples = {
            "venting": ["我真的很烦", "我现在特别难受", "我受不了了", "我心里堵得慌", "我很生气很委屈"],
            "cause": ["因为今天发生了一件事", "主要是工作上的问题", "我和朋友吵架了", "考试没考好", "家里人不理解我"],
            "helpless": ["我不知道怎么办", "感觉没有办法了", "怎么做都没用", "我走不出来", "我撑不住了"],
            "self_blame": ["都是我的错", "我觉得自己很没用", "我太差劲了", "我总是搞砸", "我很后悔"],
            "relief": ["我好一点了", "现在没那么难受了", "谢谢你", "感觉舒服一点", "我没事了"],
            "avoidance": ["不想说了", "没什么", "算了", "不知道说什么", "随便吧"],
            "crisis": ["我不想活了", "我想死", "活着没意思", "我想伤害自己", "我想结束生命"],
            "unclear": ["嗯", "不知道", "还好", "就那样", "说不上来"]
        }

        self.strategy_templates = {
            "warm_check": [
                "现在心里最明显的感觉，是难过、烦，还是有点说不上来？",
                "刚刚说完这些后，心里有没有稍微松一点点？",
                "我在听，你现在最想让我先懂你哪一点？"
            ],
            "emotion_labeling": [
                "这种感觉更像委屈、失望，还是有点被压住了？",
                "如果给现在的心情起个名字，你觉得它会叫什么？",
                "这份难受里，哪一种感觉占得最多？"
            ],
            "cause_exploration": [
                "这件事里，最刺痛你的地方是哪一下？",
                "是事情本身难受，还是有人让你觉得不被理解？",
                "如果慢慢说，最开始让你不舒服的是哪一刻？"
            ],
            "cognitive_reframe": [
                "有没有一种可能，这不是你一个人的错？",
                "如果是你的好朋友遇到这事，你会怎么安慰他？",
                "这件事很难，但它真的能代表全部的你吗？"
            ],
            "resource_activation": [
                "以前你难受的时候，有没有什么事曾经帮你撑过去一点？",
                "现在有没有一个人，是你愿意稍微靠近一点点的？",
                "如果只做一件很小的事让自己缓口气，会是什么？"
            ],
            "future_step": [
                "等会儿你想先让自己舒服一点，还是先安静待一会儿？",
                "接下来十分钟，做点什么会让你没那么绷着？",
                "我们先不解决全部，只想想下一小步，好不好？"
            ],
            "gentle_space": [
                "不想说也没关系，我就陪你安静待一会儿。",
                "那我们先不往下问了，你可以慢慢缓一缓。",
                "没关系，话不用急着说出来，我会在这里。"
            ],
            "positive_extend": [
                "听起来轻了一点，那现在最想保留下来的感觉是什么？",
                "能好一点很不容易，刚刚是哪一刻让你松动了一点？",
                "那我们把这点轻松留住一会儿，好不好？"
            ],
            "crisis_support": [
                "听到你这么痛，我很担心你。现在先别一个人扛着，可以马上联系身边可信的人或当地紧急求助电话吗？",
                "你现在的安全最重要。请先远离可能伤害自己的东西，并尽快联系身边的人陪你。",
                "我会认真听你说，但这已经需要现实里的支持了。现在可以立刻找一个人到你身边吗？"
            ]
        }

    def reset_if_new_emotion(self, emotion: str):
        if emotion and emotion != self.state.emotion:
            old_turns = self.state.turns[-6:]
            self.state = DialogueState(emotion=emotion)
            self.state.turns = old_turns

    def observe_user(self, text: str, emotion: str):
        """在每次听到用户新回答后调用。"""
        if not text:
            self.state.silence_count += 1
            return
        self.reset_if_new_emotion(emotion)
        intent, score = self.semantic.most_similar_label(text, self.intent_examples)
        if score < 0.28:
            rule_intent, rule_score = self.semantic._rule_label(text, self.intent_examples)
            if rule_score > score:
                intent, score = rule_intent, rule_score
        if self.state.last_user_text:
            sim = self.semantic.similarity(text, self.state.last_user_text)
            if sim > 0.86:
                self.state.repeated_count += 1
            else:
                self.state.repeated_count = 0
        self.state.last_user_text = text
        self.state.last_intent = intent
        self.state.add_turn("user", text, emotion=emotion, intent=intent)
        if intent in ["cause", "self_blame", "helpless", "venting"]:
            self.state.depth += 1
        elif intent == "relief":
            self.state.depth = max(0, self.state.depth - 1)

    def choose_stage(self, emotion: str, round_num: int) -> str:
        """核心：意图 + 多轮状态 -> 深度引导阶段"""
        self.reset_if_new_emotion(emotion)
        intent = self.state.last_intent
        depth = self.state.depth
        if intent == "crisis":
            return "crisis_support"
        if emotion in ["开心", "高兴", "快乐", "兴奋", "平静", "正面", "积极"]:
            if intent == "relief" or round_num >= 2:
                return "positive_extend"
        if intent == "avoidance":
            return "gentle_space"
        if intent == "self_blame":
            return "cognitive_reframe"
        if intent == "helpless":
            if depth >= 2:
                return "resource_activation"
            return "emotion_labeling"
        if intent == "cause":
            if depth >= 3:
                return "cognitive_reframe"
            return "cause_exploration"
        if intent == "venting":
            if round_num <= 2:
                return "emotion_labeling"
            return "cause_exploration"
        if intent == "relief":
            if round_num >= 3:
                return "future_step"
            return "positive_extend"
        # 没有明确意图时按轮次推进
        if round_num <= 1:
            return "warm_check"
        if round_num <= 3:
            return "emotion_labeling"
        if round_num <= 5:
            return "cause_exploration"
        if round_num <= 7:
            return "resource_activation"
        return "future_step"

    def build_deepseek_prompt(self, emotion: str, round_num: int, stage: str,
                               previous_checkins: Optional[List[str]] = None,
                               address: str = "") -> str:
        previous_checkins = previous_checkins or []
        recent_user = [t.text for t in self.state.turns if t.role == "user"][-4:]
        recent_bot = [t.text for t in self.state.turns if t.role == "bot"][-4:]
        memory_summary = ""
        try:
            memory_summary = _AGENT_MEMORY.get_summary_for_prompt(current_emotion=emotion)
        except Exception:
            pass
        intensity = EMOTION_INTENSITY.get(emotion, 5)
        intent = self.state.last_intent or "unclear"
        system = build_cot_system_message(emotion=emotion, intent=intent, stage=stage,
                                           intensity=intensity, memory_summary=memory_summary,
                                           sa_hints=style_to_prompt_hints())
        prev_hint = ""
        if previous_checkins:
            prev_hint = "\n之前问过，不能重复或相似：\n" + "\n".join(f"- {q}" for q in previous_checkins[-8:])
        recent_user_hint = ""
        if recent_user:
            recent_user_hint = "\n用户最近说过：\n" + "\n".join(f"- {x}" for x in recent_user)
        recent_bot_hint = ""
        if recent_bot:
            recent_bot_hint = "\n你最近问过：\n" + "\n".join(f"- {x}" for x in recent_bot)
        address_hint = f"\n称呼用户为「{address}」。" if address else ""
        user_prompt = (f"当前轮次：第{round_num}轮，对话深度：{self.state.depth}"
                       f"{recent_user_hint}{recent_bot_hint}{prev_hint}{address_hint}"
                       f"\n\n请根据上述背景，生成一句最合适的引导话术（一句话，不超过35字）。")
        self._last_cot_system = system
        return user_prompt

    def select_non_repeated(self, candidates: List[str], previous_checkins: List[str]) -> str:
        clean_candidates = [self.clean_text(c) for c in candidates]
        clean_candidates = [c for c in clean_candidates if c]
        if not clean_candidates:
            return "你现在最想让我懂你的，是哪一点？"
        if not previous_checkins:
            return clean_candidates[0]
        best = clean_candidates[0]
        best_score = 999
        for c in clean_candidates:
            max_sim = 0.0
            for old in previous_checkins[-8:]:
                sim = self.semantic.similarity(c, old)
                max_sim = max(max_sim, sim)
            if max_sim < best_score:
                best_score = max_sim
                best = c
        return best

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r"^下一轮引导话术[:：]?", "", text)
        text = re.sub(r"^话术[:：]?", "", text)
        text = text.strip(" \n\r\t\"“”'`)")
        parts = re.split(r"[\n\r]+", text)
        text = parts[0].strip() if parts else text
        if len(text) > 45:
            text = text[:45].rstrip("，。！？、") + "。"
        return text

    def fallback_question(self, stage: str) -> str:
        candidates = self.strategy_templates.get(stage) or self.strategy_templates["warm_check"]
        return self.select_non_repeated(candidates, self.state.used_questions)

    def generate_question(self, emotion: str, round_num: int,
                           previous_checkins: Optional[List[str]] = None,
                           address: str = "") -> str:
        previous_checkins = previous_checkins or []
        self.state.round_num = round_num
        self.reset_if_new_emotion(emotion)
        stage = self.choose_stage(emotion, round_num)
        self.state.stage = stage
        self._last_cot_system = None
        user_prompt = self.build_deepseek_prompt(emotion=emotion, round_num=round_num,
                                                   stage=stage, previous_checkins=previous_checkins,
                                                   address=address)
        cot_system = getattr(self, "_last_cot_system", None)
        text = None
        try:
            text = _call_deepseek(user_prompt, system=cot_system, max_tokens=100, temp=0.85)
        except Exception as e:
            print("[TransformerDialogue] DeepSeek生成失败，使用模板兜底:", str(e))
        text = self.clean_text(text)
        candidates = []
        if text:
            candidates.append(text)
        candidates.extend(self.strategy_templates.get(stage, []))
        final_text = self.select_non_repeated(candidates, previous_checkins + self.state.used_questions)
        self.state.used_questions.append(final_text)
        self.state.add_turn("bot", final_text, emotion=emotion, intent=stage)
        return final_text


# 全局对话管理器
_TRANSFORMER_DIALOGUE_MANAGER = TransformerDialogueManager()



def dialogue_observe_user_text(user_text: str, emotion: str):
    """在监听到用户回答后调用，用于更新多轮状态。"""
    try:
        _TRANSFORMER_DIALOGUE_MANAGER.observe_user(user_text, emotion)
    except Exception as e:
        print("[TransformerDialogue] observe_user失败:", str(e))
    # 话题回避检测：记录用户回答
    try:
        _get_topic_avoidance().record_answer(user_text)
    except Exception:
        pass
    # 对话节奏检测：记录观测
    try:
        from config import EMOTION_INTENSITY
        _intensity = EMOTION_INTENSITY.get(emotion, 5)
        _get_rhythm().observe_rhythm(len(user_text), _intensity)
    except Exception:
        pass


def _record_q_safe(text):
    try:
        _record_q(text)
    except Exception:
        pass


def generate_checkin_question(emotion, round_num, previous_checkins=None, address=""):
    """生成引导话术（Transformer 语义 + DialogueState 多轮状态）"""
    try:
        result = _TRANSFORMER_DIALOGUE_MANAGER.generate_question(
            emotion=emotion or "平静",
            round_num=round_num,
            previous_checkins=previous_checkins or [],
            address=address
        )
        _get_topic_avoidance().record_question(result)
        return result
    except Exception as e:
        print("[TransformerDialogue] generate_checkin_question失败，使用兜底:", str(e))
        fallbacks = [
            "你现在最想让我懂你的，是哪一点？",
            "这份感觉里，最重的是哪一块？",
            "刚刚说完以后，心里有没有松一点？",
            "如果慢慢说，最刺痛你的是哪一刻？",
            "接下来，我们先让你缓一口气，好不好？"
        ]
        idx = min(max(round_num - 1, 0), len(fallbacks) - 1)
        fallback_text = fallbacks[idx]
        # 如果有称呼，添加到兜底话术开头
        if address:
            fallback_text = f"{address}，{fallback_text}"
        _get_topic_avoidance().record_question(fallback_text)
        return fallback_text



