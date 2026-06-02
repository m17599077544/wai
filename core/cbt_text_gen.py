#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CBT话术生成：开场白、安慰+动作、收尾"""
from core.deepseek import _call_deepseek
from config import COMFORT_MOTION_OPTIONS, EMOTION_INTENSITY


def generate_cbt_intro(emotion, user_text, address="朋友"):
    """CBT开场白：从安慰自然过渡到自我探索"""
    if not address:
        address = "朋友"

    prompt = f"""用户称呼是「{address}」，情绪是「{emotion}」，说了：「{user_text}」
请说一句过渡话，从安慰自然地转向自我探索。
- ≤40字，必须以「{address}」开头
- 像朋友好奇地说"你有没有注意到，你刚才说的那句话里..."
- 不说专业术语
只返回这句话。"""

    result = _call_deepseek(prompt, max_tokens=80, temp=0.8)
    if result:
        result = result.strip().strip('"')
        if not result.startswith(address):
            result = f"{address}，{result}"
        return result
    return f"{address}，你刚才说的那句话，我想跟你多聊聊。"


def generate_cbt_comfort_with_motion(emotion, user_text, address="朋友", round_num=1):
    """CBT模式安慰回应（共情+引导觉察+动作），返回 (text, motion_id, motion_desc)"""
    if not address:
        address = "朋友"

    intensity = EMOTION_INTENSITY.get(emotion, 5)

    system = f"""你是"心伴"，用CBT方式陪伴用户。不说专业术语。

用户称呼: {address}
情绪: {emotion}（强度 {intensity}/10）
用户说: {user_text}

生成既共情又引导觉察的回应。

格式：
第一行：回应语（≤80字，以「{address}」开头，口语化）
第二行：动作ID

回应语要求：
- 先共情再轻轻引导觉察
- 语气温暖柔软，像"我陪你一起看看"

可用动作："""

    motion_list = "\n".join(f"  {k}：{v}" for k, v in COMFORT_MOTION_OPTIONS.items())
    system += f"\n{motion_list}"

    result = _call_deepseek(f'用户说：「{user_text}」', system=system, max_tokens=200, temp=0.9)

    motion_id = None
    text = None

    if result:
        lines = result.strip().split("\n")
        if len(lines) >= 2:
            text = lines[0].strip()
            for line in lines[1:]:
                line_stripped = line.strip().strip('"').strip()
                if line_stripped in COMFORT_MOTION_OPTIONS:
                    motion_id = line_stripped
                    break
                for mid in COMFORT_MOTION_OPTIONS:
                    if mid in line_stripped or line_stripped in mid:
                        motion_id = mid
                        break
                if motion_id:
                    break
        else:
            text = lines[0].strip()

    if not text:
        text = f"{address}，我听到了，你说的话里有一个小小的想法，我们来看看？"
    if address and not text.startswith(address):
        text = f"{address}，{text}"
    if not motion_id or motion_id not in COMFORT_MOTION_OPTIONS:
        motion_id = "H_Bec_L"

    motion_desc = COMFORT_MOTION_OPTIONS.get(motion_id, "")
    return text, motion_id, motion_desc


def generate_cbt_closing_with_summary(emotion, user_text, distortion_name, address="朋友", had_loosening=False):
    """CBT收尾话，had_loosening=用户有松动迹象"""
    if not address:
        address = "朋友"

    if had_loosening:
        prompt = f"""用户称呼「{address}」，刚完成自我探索。
最初说: {user_text}
探索「{distortion_name}」思维模式，用户已开始松动。

说一句温暖收尾。≤35字，以「{address}」开头，肯定觉察，像朋友拍拍肩。只返回这句话。"""
    else:
        prompt = f"""用户称呼「{address}」，刚完成自我探索。
最初说: {user_text}
探索「{distortion_name}」思维模式，用户没明显松动。

说一句温和收尾。≤35字，以「{address}」开头，接纳而非夸大效果，像朋友拍拍肩。只返回这句话。"""

    result = _call_deepseek(prompt, max_tokens=80, temp=0.8)
    if result:
        result = result.strip().strip('"')
        if not result.startswith(address):
            result = f"{address}，{result}"
        return result
    return (f"{address}，你刚才能看到自己的想法，真的很了不起。"
            if had_loosening else f"{address}，这种想法很正常的，不着急，慢慢来。")
