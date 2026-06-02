#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""讯飞TTS配置：音色、音量、语速、语调"""
from config import EMOTION_VOICE, DEFAULT_VOICE, EMOTION_VOLUME

XUNFEI_APP_ID = "10a7c477"
XUNFEI_API_KEY = "fd2d7c87dcc73c631066015647d9e2a7"
XUNFEI_API_SECRET = "MGY4NWY1OGE3YmVhZDVmMTljNDg5MDFl"
USE_XUNFEI_TTS = True

COMFORT_SPEED = 50
COMFORT_PITCH = 50
COMFORT_VOLUME_TTS = 82


def get_voice(e):
    return EMOTION_VOICE.get(e, DEFAULT_VOICE)


def get_volume(e):
    return EMOTION_VOLUME.get(e, 75)
