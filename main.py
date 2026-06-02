#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yanshee 智能陪伴机器人 - 模块化版本

模块结构：
  config/       - 配置常量（机器人连接、摔倒检测、情绪体系、动作映射等）
  core/         - 核心逻辑（DeepSeek调用、情绪检测、文本生成、ComfortBot主类）
  dialogue/     - 多轮对话管理（AgentMemory、CoT推理链、Transformer语义引擎）
  voice/        - 讯飞TTS语音合成
  robot/        - Robot类（HTTP/传感器控制、动作执行、摔倒检测线程、视频流SSH管理）
  music/        - 音乐工具（搜索下载、MP3帧级截取、断点续播）
  utils/        - 工具函数（预留）

启动方式：
  cd comfort_bot
  python main.py

环境变量:
  YANSHEE_IP            机器人IP（默认192.168.3.241）
  YANSHEE_SSH_USER      SSH用户名（默认root）
  YANSHEE_SSH_PASSWORD  SSH密码（留空使用密钥认证）
  VIDEO_STREAM_SKIP_SSH 设为1跳过SSH自动部署
"""
import sys
import os

# 确保项目根目录在 sys.path 中，使所有模块可用绝对导入
# 如: from config import ... / from core.deepseek import ...
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import VIDEO_STREAM_SKIP_SSH
from robot.video_stream import ensure_video_stream
from core.comfort_bot import ComfortBot

if __name__ == "__main__":
    # ========== 预检：确保机器人视频流就绪 ==========
    print("=" * 50)
    print("  Yanshee 智能陪伴机器人 v124")
    print("=" * 50)
    print()

    ok, msg = ensure_video_stream(skip_ssh=VIDEO_STREAM_SKIP_SSH)
    if ok:
        print()
    else:
        print(f"\n[警告] 视频流未就绪: {msg}")
        print("[警告] 人脸识别功能可能不可用，继续启动...\n")

    # ========== 启动主流程 ==========
    ComfortBot().run()
