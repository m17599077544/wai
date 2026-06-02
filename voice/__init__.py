#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""讯飞TTS语音合成"""
import json, time, threading, hashlib, base64
import requests, websocket
from datetime import datetime, timezone
import hmac
from config import XUNFEI_APP_ID, XUNFEI_API_KEY, XUNFEI_API_SECRET

class XunfeiTTS:
    def __init__(self, app_id=XUNFEI_APP_ID, api_key=XUNFEI_API_KEY, api_secret=XUNFEI_API_SECRET):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.host = "tts-api.xfyun.cn"
        self.path = "/v2/tts"

    def _create_url(self):
        now = datetime.now(timezone.utc)
        date = now.strftime("%a, %d %b %Y %H:%M:%S %Z")
        sig_orig = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        sig_sha = hmac.new(self.api_secret.encode('utf-8'), sig_orig.encode('utf-8'), hashlib.sha256).digest()
        sig = base64.b64encode(sig_sha).decode('utf-8')
        auth_orig = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{sig}"'
        auth = base64.b64encode(auth_orig.encode('utf-8')).decode('utf-8')
        return f"wss://{self.host}{self.path}?authorization={auth}&date={requests.utils.quote(date)}&host={self.host}"

    def synthesize(self, text, voice="x4_yezi", speed=50, volume=50, pitch=50):
        state = {"chunks": [], "done": False, "ok": False, "wav": b"", "error": ""}
        ws_url = self._create_url()

        def on_msg(ws, msg):
            try:
                d = json.loads(msg)
                if d.get("code", -1) != 0:
                    state["error"] = f"讯飞API错误 code={d.get('code')} msg={d.get('message', '')}"
                    print(f"  [TTS-讯飞] {state['error']}")
                    state["done"] = True;
                    ws.close();
                    return
                if "data" in d and "audio" in d["data"]:
                    b64 = d["data"]["audio"]
                    if b64: state["chunks"].append(b64)
                if d.get("data", {}).get("status") == 2:
                    pcm = b"".join(base64.b64decode(c) for c in state["chunks"])
                    import io
                    buf = io.BytesIO()
                    buf.write(b'RIFF');
                    buf.write((36 + len(pcm)).to_bytes(4, 'little'))
                    buf.write(b'WAVEfmt ');
                    buf.write((16).to_bytes(4, 'little'))
                    buf.write((1).to_bytes(2, 'little'));
                    buf.write((1).to_bytes(2, 'little'))
                    buf.write((16000).to_bytes(4, 'little'));
                    buf.write((32000).to_bytes(4, 'little'))
                    buf.write((2).to_bytes(2, 'little'));
                    buf.write((16).to_bytes(2, 'little'))
                    buf.write(b'data');
                    buf.write(len(pcm).to_bytes(4, 'little'));
                    buf.write(pcm)
                    state["wav"] = buf.getvalue();
                    state["ok"] = True;
                    state["done"] = True;
                    ws.close()
            except Exception as e:
                state["error"] = f"解析讯飞响应异常: {e}"
                print(f"  [TTS-讯飞] {state['error']}")
                state["done"] = True;
                ws.close()

        def on_err(ws, e):
            state["error"] = f"WebSocket错误: {e}"
            print(f"  [TTS-讯飞] WebSocket on_error: {e}")
            state["done"] = True

        def on_close(ws, *a):
            if not state["ok"] and not state["error"]:
                state["error"] = "WebSocket连接关闭但未收到有效音频"
                print(f"  [TTS-讯飞] WebSocket提前关闭，未收到音频数据")
            state["done"] = True

        def on_open(ws):
            def run():
                req = {"common": {"app_id": self.app_id}, "business": {
                    "aue": "raw", "auf": "audio/L16;rate=16000", "vcn": voice,
                    "speed": speed, "volume": volume, "pitch": pitch, "tte": "utf8"
                }, "data": {"status": 2, "text": base64.b64encode(text.encode('utf-8')).decode('utf-8')}}
                ws.send(json.dumps(req))

            threading.Thread(target=run, daemon=True).start()

        try:
            ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_msg, on_error=on_err, on_close=on_close)
            threading.Thread(target=ws.run_forever, daemon=True).start()
            deadline = time.time() + 10
            while not state["done"] and time.time() < deadline: time.sleep(0.05)
            if not state["done"]:
                print(f"  [TTS-讯飞] 超时10s无响应，text={text[:20]}...")
            elif not state["ok"]:
                print(f"  [TTS-讯飞] 合成失败: {state.get('error', '未知原因')}")
            return state["wav"] if state["ok"] else b""
        except Exception:
            return b""


