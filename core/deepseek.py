#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek API 底层调用"""
import requests
from config import DEEPSEEK_API_KEY, DEEPSEEK_URL

def _call_deepseek(prompt, system="", max_tokens=500, temp=0.7, timeout=15):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        r = requests.post(DEEPSEEK_URL,
                          headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                          json={"model": "deepseek-chat", "messages": msgs, "temperature": temp,
                                "max_tokens": max_tokens},
                          timeout=15)
        result = r.json()
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"].strip()
        return None
    except Exception as e:
        print(f"  [DeepSeek错误] {e}")
        return None
