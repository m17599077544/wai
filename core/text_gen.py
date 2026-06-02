#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek文本生成：安慰语、问候语、过渡语、推荐等"""
import re, random
from config import (
    EMOTION_INTENSITY,
    COMFORT_MOTION_OPTIONS, GREETING_MOTION_OPTIONS,
)
from core.deepseek import _call_deepseek
from dialogue import build_cot_comfort_system, _AGENT_MEMORY
from core.sa_style import style_to_prompt_hints

# DeepSeek 智能选动作
def deepseek_select_comfort_motion(emotion):
    """根据情绪选择安慰动作（不用拥抱）"""
    options_desc = "\n".join(f"  {k}：{v}" for k, v in COMFORT_MOTION_OPTIONS.items())
    system = """你是情绪陪伴机器人的动作语言专家，深谙非言语沟通心理学。
身体语言传递的信息占人类沟通的55%以上，你的每个动作都在无声地表达关怀。

动作执行方式：先执行动作→保持最终姿势→边保持边讲话→讲完复位。
选择原则：
- 高强度情绪(绝望/崩溃/心碎)：选沉稳有力的动作，传递稳定安全感
- 中等负面情绪：选温和自然的动作，像朋友默默守在身边
- 情绪较轻时：可选轻快活泼的动作，传递轻松积极的氛围
- 不要选拥抱(Hug)。只返回英文ID。"""
    prompt = f"""用户情绪：「{emotion}」（强度{EMOTION_INTENSITY.get(emotion, 5)}/10）
可用动作：\n{options_desc}
选择最合适的，只返回英文ID。"""
    result = _call_deepseek(prompt, system=system, max_tokens=30, temp=0.5)
    if result:
        mid = result.strip().strip('"')
        if mid in COMFORT_MOTION_OPTIONS:
            return mid, COMFORT_MOTION_OPTIONS[mid]
    intensity = EMOTION_INTENSITY.get(emotion, 5)
    if intensity >= 8:
        return "RaiseRightHand", COMFORT_MOTION_OPTIONS["RaiseRightHand"]
    else:
        return "Victory", COMFORT_MOTION_OPTIONS["Victory"]

def deepseek_select_greeting_motion(emotion=None, is_first=False):
    """选择问候动作，第一次固定举手打招呼"""
    if is_first:
        return "RaiseRightHand", GREETING_MOTION_OPTIONS["RaiseRightHand"]
    options_desc = "\n".join(f"  {k}：{v}" for k, v in GREETING_MOTION_OPTIONS.items())
    emo_desc = f"用户当前情绪「{emotion}」" if emotion else "用户心情有所好转"
    system = """你是情绪陪伴机器人的动作语言专家。选一个最能传递"我在，我在听"的问候动作。
先执行动作→保持→边保持边讲话→复位。
根据用户情绪状态调整：如果对方情绪低落，动作要温柔稳重；如果情绪好转，可以活泼一些。
只返回英文ID，不要解释。"""
    prompt = f"{emo_desc}。\n可用动作：\n{options_desc}\n选择最合适的，只返回英文ID。"
    result = _call_deepseek(prompt, system=system, max_tokens=30, temp=0.5)
    if result:
        mid = result.strip().strip('"')
        if mid in GREETING_MOTION_OPTIONS:
            return mid, GREETING_MOTION_OPTIONS[mid]
    return "Victory", GREETING_MOTION_OPTIONS["Victory"]

# DeepSeek 文本生成
COMFORT_FOCUS_MAP = {
    1: "深度共情——先让他感到被真正理解，不急着解决问题",
    2: "正常化——告诉他这种感受很正常、很普遍，很多人都会经历",
    3: "视角转换——用比喻或新角度帮他从困局中跳出来看",
    4: "肯定力量——肯定他的坚强、勇气或成长，即使他自己没看到",
    5: "温暖收尾——像老朋友般拍拍他的肩，让他感到踏实",
    6: "深层共情——点出他可能没说出口的深层感受",
    7: "当下锚定——引导他把注意力拉回此刻，感受呼吸和身体",
    8: "鼓励表达——温柔鼓励他说更多，做他的安全倾听者",
    9: "内在资源——帮他回忆过去克服困难的经历，发现内在力量",
    10: "无条件陪伴——单纯陪在身边，不分析不说教，只是陪着",
    11: "意象隐喻——用自然意象（海、天空、季节）传递力量和希望",
    12: "微光希望——在困境中找到一丝光亮，不盲目乐观但有真实感",
}

def generate_comfort_response(emotion, user_text, round_num=1, previous_responses=None):
    """生成安慰回应（纯文本版，向后兼容）。
    新代码建议用 generate_comfort_with_motion() 代替，可省去一次DeepSeek API调用。
    """
    text, _, _ = generate_comfort_with_motion(emotion, user_text, round_num, previous_responses)
    return text

def generate_comfort_with_motion(emotion, user_text, round_num=1, previous_responses=None):
    """生成安慰回应 + 推荐动作，使用 CoT 推理链 + AgentMemory 个性化。

    相比旧流程（两次API调用），合并为单次 DeepSeek 调用 + CoT System Message。
    新增：AgentMemory 用户历史摘要注入、三步推理链、更精细的强度风格控制。

    返回: (text: str, motion_id: str, motion_desc: str)
    """
    focus = COMFORT_FOCUS_MAP.get(round_num, "温暖陪伴")
    intensity = EMOTION_INTENSITY.get(emotion, 5)
    prev_hint = ""
    if previous_responses:
        prev_hint = ' | '.join(previous_responses[-5:])

    # AgentMemory 个性化背景
    memory_summary = ""
    try:
        memory_summary = _AGENT_MEMORY.get_summary_for_prompt(current_emotion=emotion)
    except Exception:
        pass

    # CoT System Message（含模拟退火风格提示）
    sa_hints = style_to_prompt_hints()
    system = build_cot_comfort_system(emotion=emotion, intensity=intensity, focus=focus,
                                       memory_summary=memory_summary, prev_hint=prev_hint)
    system += "\n" + sa_hints

    # 构建可用动作列表（嵌入prompt中让模型同时选择）
    motion_list = "\n".join(f"  {k}：{v}" for k, v in COMFORT_MOTION_OPTIONS.items())
    system += f"\n\n可用动作：\n{motion_list}"

    result = _call_deepseek(f'用户说：「{user_text}」', system=system, max_tokens=200, temp=0.9)

    # 解析返回结果：尝试提取文本 + 动作ID
    motion_id = None
    motion_desc = None
    text = None

    if result:
        lines = result.strip().split("\n")
        # 策略1: 多行返回——首行为文本，后续行找动作ID
        if len(lines) >= 2:
            text = lines[0].strip()
            for line in lines[1:]:
                line_stripped = line.strip().strip('"').strip()
                # 直接匹配
                if line_stripped in COMFORT_MOTION_OPTIONS:
                    motion_id = line_stripped
                    break
                # 模糊匹配（模型可能在ID前后加了标点或文字）
                for mid in COMFORT_MOTION_OPTIONS:
                    if mid in line_stripped or line_stripped in mid:
                        motion_id = mid
                        break
                if motion_id:
                    break
        else:
            # 单行返回（模型没给动作ID），整段作为文本
            text = lines[0].strip()

    # 文本兜底
    if not text:
        fallbacks = ["我在这里陪着你。", "你的感受很重要。", "慢慢来，我在这里。",
                     "你已经很勇敢了。", "不管怎样，我都陪着你。"]
        text = fallbacks[(round_num - 1) % len(fallbacks)]

    # 动作兜底（如果解析失败则用规则降级）
    if not motion_id or motion_id not in COMFORT_MOTION_OPTIONS:
        intensity_local = EMOTION_INTENSITY.get(emotion, 5)
        if intensity_local >= 8:
            motion_id = "RaiseRightHand"
        else:
            motion_id = "Victory"

    motion_desc = COMFORT_MOTION_OPTIONS.get(motion_id, "")
    return text, motion_id, motion_desc

def generate_music_intro(emotion, song_name=None):
    """放音乐前的过渡语——温柔好听的告知"""
    if song_name:
        prompt = f"""用户情绪「{emotion}」，你刚为他选了一首歌《{song_name}》。
用一句20-40字的话温柔地告知对方你要放这首歌，可以简单描述一下这首歌的感觉（比如"这是一首很安静的歌"、"这首歌关于成长"）。
重要：必须使用"放首"这个表达，不要说"放一首"。
像朋友轻声说"我给你放首《{song_name}》，这是一首让人心静下来的歌"。
只返回这句话，不要书名号。"""
    else:
        prompt = f"""用户情绪「{emotion}」，你想为他放一首歌。
在放音乐前，用一句温柔的话告知对方。这句话要像一个很在意你的人，轻轻地说"我给你放首歌，闭上眼睛听听看"。
重要：必须使用"放首"这个表达，不要说"放一首"或"放了"。
要求：
- 20-40字，像说话一样自然，不要书面语
- 语气要温暖柔软，像一个拥抱的温度
- 不要说教，不要自我指涉，不要问问题
- 可以暗示这首歌是专门为他选的（"有首歌让我想起你说的"）
只返回这句话。"""
    return _call_deepseek(prompt, max_tokens=80, temp=0.9) or "我给你放首歌，闭上眼睛慢慢听就好～"

def generate_dance_intro(emotion, dance_name=None):
    """跳舞前的过渡语——温柔俏皮的告知"""
    if dance_name:
        prompt = f"""用户情绪「{emotion}」，你刚为他选了一支舞《{dance_name}》。
跳舞前说一句话，告知对方你要表演这个舞蹈，可以简单描述一下这支舞的感觉（比如"这是一支很可爱的舞"、"这支舞跳起来心情会变好"）。
重要：必须使用"跳个"、"跳支"、"表演个"或"表演支"这几个表达之一，不要说"跳一支"或"表演一个"。
像朋友俏皮地说"我给你跳个《{dance_name}》"或"我给你表演支《{dance_name}》"。
只返回这句话，不要书名号。"""
    else:
        prompt = f"""用户情绪「{emotion}」，你想为他表演一个舞蹈。
跳舞前说一句话，像一个调皮的好朋友说"我给你跳个舞，别笑话我啊"。
重要：必须使用"跳个"、"跳支"、"表演个"或"表演支"这几个表达之一，不要说"跳一支"或"表演一个"。
要求：
- 20-40字，自然俏皮，带一点点害羞或可爱
- 语气温暖但不沉重，像在他面前撒了个小娇
- 目的是让他笑一笑、放松一下
- 不要自我指涉，不要说教，不要问问题
只返回这句话。"""
    return _call_deepseek(prompt, max_tokens=80, temp=0.9) or "我给你跳个舞好不好呀？别笑话我～"

def generate_positive_show_intro():
    """正面情绪时问要不要看表演"""
    prompt = """用户心情不错，你想为他表演一个小节目。
说一句话问他要不要看，像一个朋友在你开心时突然说"嘿，我给你表演个绝活！"
要求：15-30字，活泼自然，带一点点得意和俏皮。只返回这句话。"""
    return _call_deepseek(prompt, max_tokens=50, temp=0.9) or "嘿，要不要看我表演一个？算是个小惊喜～"

def generate_cycle_closing(emotion, user_text, previous_closings=None):
    """cycle结尾的过渡鼓励语：给用户温暖收尾，不引导说话（引导说话交给递进询问）"""
    prev_hint = ""
    if previous_closings:
        prev_hint = f"\n之前说过：{' | '.join(previous_closings[-3:])}\n用完全不同的表达、角度或意象！"
    prompt = f"""用户情绪「{emotion}」，他说过「{user_text}」。你刚完成一轮陪伴。
现在说一句温暖有力量的收尾话，像暴风雨后递给他一杯热茶——不用多说什么，他懂。
{prev_hint}
要求：
- 不超过25字，简洁有力，有一种"我在"的笃定感
- 这是本轮收尾，绝对不要用疑问句引导用户继续说话
- 可以用意象结尾（"总会天亮的"），也可以直接温暖收束
- 不要自我指涉，不要说教
只返回鼓励语。"""
    text = _call_deepseek(prompt, max_tokens=60, temp=0.9)
    if text: return text
    fallbacks = ["我一直陪着你呢。", "不管怎样，你不是一个人。", "你的感受我都理解。",
                 "慢慢来，不着急。", "我会一直在这里的。"]
    return random.choice(fallbacks)

def generate_greeting():
    prompt = """你是"心伴"，一个温暖的陪伴者。刚启动，要跟用户打第一个招呼。
这个招呼很重要——它决定了对方愿不愿意对你敞开心扉。
要求：
- 不超过40字，语气像一个老朋友见面，随意但温暖
- 不要自我介绍"我是心伴"，不要说"我是AI"
- 可以用一个小细节切入（"今天过得还好吗"比"你好"好十倍）
- 让他感觉这不是机器在问，而是有人在关心
只返回问候语。"""
    return _call_deepseek(prompt, max_tokens=60, temp=0.9) or "嘿，好久不见～最近过得怎么样？有什么想说的吗？"

def generate_goodbye(emotion):
    prompt = f"""用户情绪「{emotion}」，你们的对话要结束了。
说一句告别语，像一个朋友在你离开时说的最后一句话——不用很长，但要让他感到被在意。
要求：不超过30字，温暖有力量，可以带一点期待下次见面的感觉。只返回告别语。"""
    return _call_deepseek(prompt, max_tokens=50, temp=0.9) or "下次见面的时候，希望看到你笑哦～"

def generate_retry_ask():
    return _call_deepseek("""你刚没听清对方说的话。温柔回应并请他再说一遍。
像一个聊天时没听清的朋友那样，不是机械的"请再说一次"。
不超过20字，自然随意。只返回这句话。""",
                          max_tokens=30, temp=0.8) or "嗯？不好意思，你刚才说什么来着？我重新听一下～"

def generate_timeout_response(emotion):
    prompt = f"""用户情绪「{emotion}」，他已经沉默了30秒没说话。
说一句温暖的话打破沉默——不是催他开口，而是让他知道你在。
像一个安静的陪伴者轻声说"没关系，慢慢来"。
要求：不超过25字，不催促不追问，给他安全感。只返回这句话。"""
    return _call_deepseek(prompt, max_tokens=40, temp=0.9) or "没事，想什么时候说都可以，我就在这里。"

def generate_response_question(emotion, round_num, previous_checkins=None, address=None):
    """正面情绪的回应询问，不同于安慰式询问，语气更轻松活泼。
    round_num: 当前总轮次
    previous_checkins: 之前用过的询问话术列表
    address: 称呼（可选）
    """
# 处理空地址
    if not address:
        address = "朋友"
    
    prev_hint = ""
    if previous_checkins:
        prev_hint = f"\n之前已经问过（绝对不能重复或相似）：{' | '.join(previous_checkins[-5:])}"
    prompt = f"""用户称呼是「{address}」，情绪是「{emotion}」（正面情绪），这是第{round_num}轮陪伴。
本轮询问重点：像朋友一样轻松愉快地问问用户感受，或者问问今天有什么事让你开心？
注意不要像安慰，要像朋友聊天那样自然轻松。{prev_hint}
要求：
- 不超过35字，轻松活泼，像在跟你聊天的感觉
- 必须以「{address}」开头，后面加逗号或感叹号
- 可以问问"有什么开心的事吗""今天怎么样啦"这样自然的问法
- 不要太正经太温柔，像朋友日常聊天那样就好
- 不能重复之前问过的

示例格式：
{address}，今天有啥开心的事吗？
{address}，最近怎么样呀～

只返回询问语，不要任何前缀。"""
    text = _call_deepseek(prompt, max_tokens=80, temp=0.9)
    if text:
        text = text.strip().strip('"').strip()
        # 确保以称呼开头
        if not text.startswith(address):
            text = f"{address}，{text}"
        return text
    fallbacks = [
        f"{address}，有什么开心的事吗？", f"{address}，今天有什么好事呀？", f"{address}，最近怎么样～",
        f"{address}，有什么想分享的吗？", f"{address}，心情不错吧？", f"{address}，聊聊呗～",
    ]
    return fallbacks[round_num % len(fallbacks)]

def generate_post_show_greeting(emotion):
    """表演后重新问候，问新心情"""
    prompt = f"""用户刚看完你的表演，情绪是「{emotion}」。
表演结束后说一句话，像一个朋友表演完回到你身边，带着一点点期待问你"好看吗？"
要求：不超过30字，温暖关切，自然地问问他现在的感受。只返回这句话。"""
    return _call_deepseek(prompt, max_tokens=60, temp=0.9) or "嘿，怎么样？心情有没有好那么一点点？"

def generate_fall_exclamation(direction="front"):
    """摔倒后的感叹语——自然地说出摔倒了，有点疼，准备站起来
    direction: "front"前倒 / "rear"后倒
    前倒不能提屁股，后倒可以说摔到屁股"""
    if direction == "rear":
        prompt = """你是一个可爱的陪伴机器人，刚刚往后摔倒了（摔到了屁股）。
说一句话表达：哎呀摔倒了、屁股好疼、准备站起来。
要求：
- 15-35字，口语化自然，像朋友摔了一跤的反应
- 语气可以带一点点委屈或不好意思，但不要太夸张
- 不要自我指涉"我是机器人"，不要说教，不要问问题
- 让人觉得可爱真实就好
只返回这句话。"""
        fallback = "哎呀！摔到屁股了，有点疼呢，我这就站起来～"
    else:
        prompt = """你是一个可爱的陪伴机器人，刚刚往前摔倒了（面朝下摔倒的）。
说一句话表达：哎呀摔倒了、有点疼、准备站起来。
要求：
- 15-35字，口语化自然，像朋友摔了一跤的反应
- 语气可以带一点点委屈或不好意思，但不要太夸张
- 不要自我指涉"我是机器人"，不要说教，不要问问题
- 绝对不要提到屁股（前倒不会摔到屁股）
- 让人觉得可爱真实就好
只返回这句话。"""
        fallback = "哎呀！不小心扑倒了，有点疼呢，我这就站起来～"
    return _call_deepseek(prompt, max_tokens=60, temp=0.9, timeout=5) or fallback

def generate_repeat_comfort(text, emotion, round_num=1):
    """安慰/打招呼摔倒恢复后，重新说类似意思但不同的话。
    基于之前说的内容，生成一句意思相近但措辞不同的新话。
    """
    prompt = f"""用户情绪「{emotion}」，你之前说过一句话：「{text}」
现在摔倒后站起来了，需要重新说一句话安慰对方。
要求：
- 意思与原话相近，但措辞完全不同
- 15-35字，语气温暖柔和
- 不要重复原话的措辞，换一种说法
- 不要问问题，不要说教
只返回新的一句话。"""
    new_text = _call_deepseek(prompt, max_tokens=80, temp=0.9)
    if new_text and new_text != text:
        return new_text
    return None  # 返回None表示使用默认恢复语

def generate_repeat_greeting(emotion, round_num=1):
    """打招呼摔倒恢复后，重新说类似意思但不同的话。
    基于当前情绪，生成一句打招呼/关心的话。
    """
    prompt = f"""用户情绪「{emotion}」，你之前正在打招呼。
现在摔倒后站起来了，需要重新说一句话。
要求：
- 15-30字，语气轻松自然
- 与之前的打招呼内容意思相近但措辞不同
- 不要问问题，不要说教
只返回新的一句话。"""
    new_text = _call_deepseek(prompt, max_tokens=80, temp=0.9)
    if new_text:
        return new_text
    return None

def generate_standup_recovery(activity_type=None):
    """站起后的恢复语——根据摔倒前的活动类型说不同的话
    activity_type: "dance"/"music"/"comfort"/"greet"/None(等待中)
    """
    if activity_type == "dance":
        prompt = """你是一个可爱的陪伴机器人，刚刚摔倒后成功站起来了，之前正在跳舞。
说一句话表达：我站起来了、继续给你跳舞。
要求：
- 15-30字，口语化自然，轻松带点俏皮
- 不要自我指涉"我是机器人"，不要说教，不要问问题
只返回这句话。"""
        fallback = "嘿嘿，我站起来了！继续给你跳舞哦～"
    elif activity_type == "music":
        prompt = """你是一个可爱的陪伴机器人，刚刚摔倒后成功站起来了，之前正在放音乐。
说一句话表达：我站起来了、继续给你放音乐（必须提到"放音乐"或"放歌"）。
要求：
- 15-30字，口语化自然，轻松带点俏皮
- 必须明确说"继续放音乐"或"继续放歌"，让对方知道音乐要从断点继续播放
- 不要自我指涉"我是机器人"，不要说教，不要问问题
只返回这句话。"""
        fallback = "嘿嘿，我站起来了！继续给你放音乐哦～"
    else:
        # 安慰/打招呼/等待中 — 简单说站起来了就行
        prompt = """你是一个可爱的陪伴机器人，刚刚摔倒后成功站起来了，正在陪人聊天。
说一句话表达：我站起来了、没事。
要求：
- 10-20字，简短自然，像朋友摔了一下拍拍灰尘说"没事"
- 不要自我指涉"我是机器人"，不要说教，不要问问题
只返回这句话。"""
        fallback = "嘿嘿，我站起来了，没事没事～"
    return _call_deepseek(prompt, max_tokens=60, temp=0.9, timeout=5) or fallback

def generate_pushup_intro(emotion):
    """俯卧撑前的过渡语——俏皮自信地告知"""
    prompt = f"""用户情绪「{emotion}」，你想表演一个俯卧撑。
表演前说一句话告知对方，像一个自信的好朋友俏皮地说"嘿，我给你表演个俯卧撑看看，别眨眼"。
要求：
- 15-35字，自然俏皮，带一点点自信和小得意
- 语气活泼但不沉重，目的是让他开心
- 不要自我指涉，不要说教，不要问问题
只返回这句话。"""
    return _call_deepseek(prompt, max_tokens=60, temp=0.9) or "嘿，我给你表演个俯卧撑看看，别眨眼哦！"

def recommend_music(emotion, user_text="", played_songs=None):
    """多维音乐推荐：情绪×风格×时段×去重"""
    from core.music_recommender import recommend_music_v2
    return recommend_music_v2(emotion, user_text, played_songs)

# 带称呼的人脸识别话术生成

def generate_comfort_with_address(emotion, user_text, address, round_num=1, previous_responses=None):
    """生成带称呼的安慰回应（人脸识别版）

    格式: "称呼 + ", " + 看到情绪 + ", " + 安慰话"

    Args:
        emotion: 当前情绪
        user_text: 用户说的话
        address: 称呼（如"小哥哥"、"阿姨"等）
        round_num: 当前轮次
        previous_responses: 之前的回复列表

    Returns:
        (text: str, motion_id: str, motion_desc: str)
    """
# 处理空地址（如果没人脸识别结果，使用默认"朋友"）
    if not address:
        address = "朋友"
    print(f"  [DEBUG-generate_comfort_with_address] address='{address}', emotion='{emotion}', round_num={round_num}")
    
    focus = COMFORT_FOCUS_MAP.get(round_num, "温暖陪伴")
    intensity = EMOTION_INTENSITY.get(emotion, 5)

    prev_hint = ""
    if previous_responses:
        prev_hint = ' | '.join(previous_responses[-5:])

    # AgentMemory 个性化背景
    memory_summary = ""
    try:
        memory_summary = _AGENT_MEMORY.get_summary_for_prompt(current_emotion=emotion)
    except Exception:
        pass

    # 带称呼的 CoT System Message
    emotion_desc_map = {
        "开心": "看起来心情不错",
        "平静": "看起来比较平静",
        "满足": "看起来很满足",
        "期待": "看起来有所期待",
        "放松": "看起来很放松",
        "悲伤": "看起来有点悲伤",
        "焦虑": "看起来有点焦虑",
        "愤怒": "看起来有点生气",
        "恐惧": "看起来有点害怕",
        "绝望": "看起来很绝望",
        "崩溃": "看起来快崩溃了",
        "心碎": "看起来很心碎",
        "孤独": "看起来有点孤单",
        "迷茫": "看起来有点迷茫",
        "疲惫": "看起来很疲惫",
        "内耗": "看起来有点纠结",
    }
    emotion_desc = emotion_desc_map.get(emotion, f"情绪是「{emotion}」")

    system = f"""你是"心伴"，融合心理咨询师智慧、文学家温度的陪伴者。

【用户背景】
{memory_summary if memory_summary else "请保持温柔陪伴。"}

【当前情绪】{emotion}（强度 {intensity}/10）
【本轮策略】{focus}

【CoT安慰推理链（只输出最终结果，不输出推理）】
步骤一：这位用户此刻最深层的痛苦/需求是什么？
步骤二：「{focus}」这个策略，最合适的切入角度是什么？
步骤三：用一句最自然的话，把上面两步融为一体。

【称呼规则（必须遵守）】
- 称呼必须放在句首，后面直接加逗号或感叹号，如："小哥哥，"、"阿姨，"、"朋友，"
- 称呼后要先描述你看到对方的情绪，如"我看到你有点悲伤呢"、"你好像有点焦虑"
- 之后再说安慰的话
- 格式：称呼 + ", " + 看到情绪 + ", " + 安慰话

【输出格式（严格）】
第一行：安慰语（≤80字，口语化，必须以称呼开头，不加引号）
第二行：动作ID（从下面列表中选）

【安慰语禁忌】
- 不说"作为AI"，不说"我建议你"
- 不用"你应该""你需要"，改用"也许""有时候"
- 不煽情泛滥，共情点到即止"""

    # 模拟退火风格提示
    sa_hints = style_to_prompt_hints()
    system += "\n" + sa_hints

    # 构建可用动作列表
    motion_list = "\n".join(f"  {k}：{v}" for k, v in COMFORT_MOTION_OPTIONS.items())
    system += f"\n\n可用动作：\n{motion_list}"

    # 构建 prompt
    prompt = f"""用户称呼是「{address}」，{emotion_desc}。
用户说：「{user_text}」
{f"之前已说过：{prev_hint}" if prev_hint else ""}
请生成一句带称呼的安慰语，必须以「{address}」开头。

示例格式：
小哥哥，我看到你有点悲伤呢，不要难过，我会一直陪着你。
小姐姐，你好像有点焦虑，我们慢慢来，不着急。
朋友，你现在有点累吧，没关系的，休息一下也好。
"""

    result = _call_deepseek(prompt, system=system, max_tokens=150, temp=0.9)

    # 解析返回结果
    motion_id = None
    motion_desc = None
    text = None

    if result:
        lines = result.strip().split("\n")
        # 策略1: 多行返回——首行为文本，后续行找动作ID
        if len(lines) >= 2:
            text = lines[0].strip()
            for line in lines[1:]:
                line_stripped = line.strip().strip('"').strip()
                # 直接匹配
                if line_stripped in COMFORT_MOTION_OPTIONS:
                    motion_id = line_stripped
                    break
                # 模糊匹配
                for mid in COMFORT_MOTION_OPTIONS:
                    if mid in line_stripped or line_stripped in mid:
                        motion_id = mid
                        break
                if motion_id:
                    break
        else:
            text = lines[0].strip()

    # 文本兜底（确保有称呼）
    if not text:
        fallbacks = [
            f"{address}，我在这里陪着你。",
            f"{address}，你的感受很重要。",
            f"{address}，慢慢来，我在这里。",
            f"{address}，不管怎样，我都陪着你。",
        ]
        text = fallbacks[(round_num - 1) % len(fallbacks)]

# 强制确保文本以地址开头（更严格的匹配）
    text_stripped = text.strip()
    # 检查是否以地址开头（支持前后有空格、引号等）
    if address and not text_stripped.startswith(address):
        # 去掉可能的引号后缀后再检查
        clean_text = text_stripped.strip('"').strip('。').strip()
        if not clean_text.startswith(address):
            print(f"  [地址修正] DeepSeek返回未以地址开头，强制添加: {address}")
            text = f"{address}，{text_stripped}"

    # 动作兜底
    if not motion_id or motion_id not in COMFORT_MOTION_OPTIONS:
        intensity_local = EMOTION_INTENSITY.get(emotion, 5)
        if intensity_local >= 8:
            motion_id = "RaiseRightHand"
        else:
            motion_id = "Victory"

    motion_desc = COMFORT_MOTION_OPTIONS.get(motion_id, "")
    return text, motion_id, motion_desc

def generate_checkin_with_address(emotion, address, round_num, previous_checkins=None):
    """生成带称呼的询问话术（人脸识别版）

    Args:
        emotion: 当前情绪
        address: 称呼（如"小哥哥"、"阿姨"等）
        round_num: 当前轮次
        previous_checkins: 之前用过的询问话术列表

    Returns:
        str: 询问话术
    """
# 处理空地址
    if not address:
        address = "朋友"
    
    prev_hint = ""
    if previous_checkins:
        prev_hint = f"\n之前已经问过（绝对不能重复或相似）：{' | '.join(previous_checkins[-5:])}"

    emotion_desc_map = {
        "开心": "看起来心情不错",
        "平静": "看起来比较平静",
        "满足": "看起来很满足",
        "期待": "看起来有所期待",
        "放松": "看起来很放松",
        "悲伤": "看起来有点悲伤",
        "焦虑": "看起来有点焦虑",
        "愤怒": "看起来有点生气",
        "恐惧": "看起来有点害怕",
        "绝望": "看起来很绝望",
        "崩溃": "看起来快崩溃了",
        "心碎": "看起来很心碎",
        "孤独": "看起来有点孤单",
        "迷茫": "看起来有点迷茫",
        "疲惫": "看起来很疲惫",
        "内耗": "看起来有点纠结",
    }
    emotion_desc = emotion_desc_map.get(emotion, f"当前情绪「{emotion}」")

    prompt = f"""用户称呼是「{address}」，{emotion_desc}，这是第{round_num}轮陪伴。
你是一个温暖的陪伴者，要轻轻地问用户此刻的感受。
{prev_hint}
要求：
- 不超过35字，必须以「{address}」开头，后面加逗号或感叹号
- 像朋友一样自然地询问，不要太正式
- 询问时要提到你看到对方的状态（表情/状态/感受）
- 不要重复之前问过的

示例格式：
小哥哥，我看到你有点悲伤呢，愿意说说吗？
阿姨，你看起来有点累，发生什么事了吗？
朋友，你现在感觉怎么样？

只返回询问语，不要前缀。"""

    text = _call_deepseek(prompt, max_tokens=80, temp=0.9)

    if text:
        text = text.strip().strip('"').strip()
        # 确保以称呼开头
        if not text.startswith(address):
            text = f"{address}，{text}"
        return text

    # 兜底
    fallbacks = [
        f"{address}，我看到你有点...还好吗？",
        f"{address}，你愿意说说吗？",
        f"{address}，现在感觉怎么样？",
    ]
    idx = min(max(round_num - 1, 0), len(fallbacks) - 1)
    return fallbacks[idx]

def generate_face_comfort_intro(emotion, address, detected_emotion=None):
    """生成人脸识别后的开场白（看到对方情绪后的反应）

    Args:
        emotion: 当前情绪（项目情绪）
        address: 称呼
        detected_emotion: 人脸识别的情绪（可能与 emotion 不同）

    Returns:
        str: 开场白
    """
# 处理空地址
    if not address:
        address = "朋友"
    
    if detected_emotion and detected_emotion != emotion:
        # 情绪不一致时的处理
        prompt = f"""用户称呼是「{address}」，你通过人脸识别看到对方情绪是「{detected_emotion}」。
这是一句轻轻的回应，表示你注意到了对方的状态。
要求：
- 15-30字，必须以「{address}」开头
- 自然地表达你看到了对方的情绪
- 不要分析或追问，只是温柔地承认

示例：
{address}，我看到你有点悲伤呢。
{address}，你好像有点焦虑的样子。
{address}，你看起来有点累。

只返回这句话。"""
    else:
        prompt = f"""用户称呼是「{address}」，当前情绪「{emotion}」。
这是一句轻轻的回应，表示你注意到了对方的状态。
要求：
- 15-30字，必须以「{address}」开头
- 自然地表达你看到了对方的情绪
- 不要分析或追问，只是温柔地承认

示例：
{address}，我看到你了。
{address}，你看起来有点心事的样子。
{address}，你还好吗？

只返回这句话。"""

    return _call_deepseek(prompt, max_tokens=60, temp=0.9) or f"{address}，我在这里。"

