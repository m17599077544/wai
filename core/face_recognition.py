#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yanshee 机器人人脸识别模块
- 使用 MJPEG/RTSP 流获取机器人摄像头视频
- 使用 DeepFace 库进行本地分析（年龄、性别、情绪）
- 支持多脸识别（优先选择最大人脸）
- 支持 3 层识别优先级

摄像头优先级:
    1. MJPEG 流 (http://{ip}:8080/stream) - 通过自部署 MJPEG 服务器
    2. RTSP 流 (rtsp://{ip}:8554/stream0) - 原生 RTSP 流
    3. HTTP 轮询 (拍照 API) - 最后的降级方案

使用方法:
    from core.face_recognition import init_face_recognition, get_face_manager
    manager = init_face_recognition(robot_ip="192.168.3.215")

    # 检测人脸
    result = manager.detect()
    if result["success"]:
        print(f"识别到: 年龄={result['age']}, 性别={result['gender']}, 情绪={result['emotion']}")

    # 多脸模式
    results = manager.detect_all_faces()
    for i, face in enumerate(results):
        print(f"Face {i+1}: {face['age']}岁 {face['gender']} {face['emotion']}")
"""

import os
import time
import json
import threading
import socket
import cv2
import numpy as np
from typing import Optional, Dict, List

# ==================== 配置（必须在 import DeepFace 之前设置）===================
# 机器人 IP 统一从 config 读取，不再本地定义
from config import ROBOT_IP

# DeepFace 模型路径（已复制到项目 .deepface 目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPFACE_MODELS_PATH = os.path.join(PROJECT_ROOT, ".deepface", "weights")

# 必须在 import DeepFace 之前设置环境变量
# 注意：DeepFace 会在 DEEPFACE_HOME 下创建 .deepface 子目录存放模型
# 所以 DEEPFACE_HOME 应该指向 .deepface 的父目录，这样创建的路径正好是 .deepface/weights
if os.path.exists(DEEPFACE_MODELS_PATH):
    os.environ['DEEPFACE_HOME'] = PROJECT_ROOT  # 设置为项目根目录，DeepFace会创建 项目根目录/.deepface/weights
    print(f"  [DeepFace] 模型目录: {os.environ.get('DEEPFACE_HOME')}/.deepface/weights")
    print(f"  [DeepFace] 检查模型文件...")

    # 检查关键模型是否存在
    import glob
    expected_models = [
        "age_model_weights.h5",
        "gender_model_weights.h5",
        "facial_expression_model_weights.h5",
    ]
    found_count = 0
    for model in expected_models:
        model_path = os.path.join(DEEPFACE_MODELS_PATH, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            print(f"    ✓ {model} ({size_mb:.1f}MB)")
            found_count += 1
        else:
            print(f"    ✗ {model} (未找到)")
    print(f"  [DeepFace] 模型文件: {found_count}/{len(expected_models)} 已就绪")

# 尝试导入 DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("  [警告] DeepFace 未安装，将使用 YanAPI 模式")

# Haar Cascade 路径（避免中文路径导致 OpenCV FileStorage C++ 后端加载失败）
_HAARCASCADE_FILE = "haarcascade_frontalface_default.xml"
_CASCADE_DIR = os.path.join(os.path.expanduser("~"), ".yanshee_cascade")
if not os.path.exists(_CASCADE_DIR):
    os.makedirs(_CASCADE_DIR, exist_ok=True)
_LOCAL_CASCADE_PATH = os.path.join(_CASCADE_DIR, _HAARCASCADE_FILE)

# 首次启动时从 cv2 data 目录复制 cascade 文件到安全路径（无中文字符）
if not os.path.exists(_LOCAL_CASCADE_PATH):
    import shutil
    _src = os.path.join(cv2.data.haarcascades, _HAARCASCADE_FILE)
    if os.path.exists(_src):
        shutil.copy2(_src, _LOCAL_CASCADE_PATH)
        print(f"  [Cascade] 已复制到 {_LOCAL_CASCADE_PATH}")

def _load_face_cascade():
    """加载 Haar 级联分类器，优先用本地非中文路径副本"""
    if os.path.exists(_LOCAL_CASCADE_PATH) and os.path.getsize(_LOCAL_CASCADE_PATH) > 0:
        cascade = cv2.CascadeClassifier(_LOCAL_CASCADE_PATH)
        if not cascade.empty():
            return cascade
    # fallback: 原始路径
    orig = os.path.join(cv2.data.haarcascades, _HAARCASCADE_FILE)
    if os.path.exists(orig):
        cascade = cv2.CascadeClassifier(orig)
        if not cascade.empty():
            return cascade
    return None

# MJPEG 流地址模式（Yanshee 机器人官方视频流）
MJPEG_PATTERNS = [
    "http://{ip}:8080/stream",   # 自部署 MJPEG 服务器（优先）
    "http://{ip}:8080/",         # 备选路径
]

# RTSP 流地址模式（Yanshee 机器人常用地址）
RTSP_PATTERNS = [
    "rtsp://{ip}:8554/stream0",
    "rtsp://{ip}:8554/live",
    "rtsp://{ip}:8554/stream",
    "rtsp://{ip}:554/live",
    "rtsp://{ip}:554/stream",
]

# HTTP 拍照轮询模式配置
HTTP_POLL_CONFIG = {
    "base_url": "http://{ip}:9090/v1",
    "interval": 0.5,  # 拍照间隔（秒）
    "resolution": "640x480",
}

# 情绪映射：DeepFace 情绪 → 项目情绪体系
DEEPFACE_TO_EMOTION_MAP = {
    "happy": "开心",
    "sad": "悲伤",
    "angry": "愤怒",
    "fear": "恐惧",
    "surprise": "惊讶",
    "disgust": "厌恶",
    "neutral": "平静",
}

# 摄像头情绪 → 强度等级映射（确保5个等级都有对应）
# high级别（绝望/崩溃等）需要关键词识别，摄像头只负责检测可识别的情绪
EMOTION_TO_LEVEL_MAP = {
    # mid_high (7-8): 强烈负面情绪
    "悲伤": "mid_high",
    "愤怒": "mid_high",
    "恐惧": "mid_high",
    # mid (5-7): 中等强度
    "厌恶": "mid",
    "惊讶": "mid",  # 惊讶可能是负面（震惊）也可能是惊喜
    # low (3-5): 轻度负面
    "心烦": "low",
    "无奈": "low",
    "孤单": "low",
    "麻木": "low",
    # positive (2-3): 正面情绪
    "开心": "positive",
    "平静": "positive",
    "满足": "positive",
    "期待": "positive",
    "放松": "positive",
}

# high级别关键词（摄像头识别不了的极度负面情绪）
HIGH_LEVEL_KEYWORDS = {
    "绝望", "崩溃", "心碎", "痛苦", "彻底完了", "活不下去了",
    "不想活了", "太难受了", "受不了了", "彻底垮了"
}

def get_emotion_level_from_face(emotion: str) -> str:
    """根据摄像头识别的情绪获取强度等级

    Returns:
        "high" / "mid_high" / "mid" / "low" / "positive"
    """
    return EMOTION_TO_LEVEL_MAP.get(emotion, "mid")

def detect_high_level_keywords(text: str) -> bool:
    """检测文本中是否包含high级别情绪关键词

    Args:
        text: 用户输入文本

    Returns:
        True if high-level emotion keywords detected
    """
    if not text:
        return False
    text_lower = text.lower()
    for keyword in HIGH_LEVEL_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

# 性别映射
GENDER_MAP = {
    "Man": "男",
    "Male": "男",
    "man": "男",
    "male": "男",
    "Woman": "女",
    "Female": "女",
    "woman": "女",
    "female": "女",
}

def _http_port_check(url, timeout=2.0):
    """快速检测 HTTP 端口是否可达（纯 socket，避免 VideoCapture 无限阻塞）"""
    import re
    m = re.match(r'https?://([^/:]+):(\d+)', url)
    if not m:
        return False
    host, port = m.group(1), int(m.group(2))
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _try_video_capture(url, timeout=5.0):
    """带超时的 VideoCapture 连接，避免 MJPEG/RTSP 不可达时卡死"""
    result = {"cap": None, "ok": False}

    def _connect():
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    result["cap"] = cap
                    result["ok"] = True
                    return
        if cap:
            cap.release()

    t = threading.Thread(target=_connect, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result["cap"] if result["ok"] else None


# MJPEG 摄像头类
class MJPEGCamera:
    """MJPEG 流摄像头（通过 HTTP 获取视频流，带超时保护）"""

    def __init__(self, ip: str, custom_url: str = None):
        self.ip = ip
        self.cap = None
        self.stream_url = None
        self.face_cascade = None

        # 加载 Haar 级联分类器（使用非中文路径副本）
        self.face_cascade = _load_face_cascade()

        # 尝试连接
        urls = [custom_url] if custom_url else [
            p.format(ip=ip) for p in MJPEG_PATTERNS
        ]

        for url in urls:
            print(f"  [MJPEG] 尝试连接: {url}")

            # 预检：HTTP 端点是否可达（2秒快速判断）
            if not _http_port_check(url, timeout=2.0):
                print(f"  [MJPEG] 端点不可达，跳过")
                continue

            # 带超时的 VideoCapture 连接
            self.cap = _try_video_capture(url, timeout=5.0)
            if self.cap is not None:
                print(f"  [MJPEG] ✓ 连接成功: {url}")
                self.stream_url = url
                return

        self.cap = None
        print(f"  [MJPEG] ✗ 无法连接到 MJPEG 流")

    def is_connected(self) -> bool:
        if self.cap is None:
            return False
        # 实时检查：尝试读一帧验证
        if not self.cap.isOpened():
            return False
        ret, _ = self.cap.read()
        if not ret:
            return False
        # 重新 seek 到最新帧（解决 MJPEG buffer 延迟问题）
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        return True

    def reconnect(self):
        """重新连接 MJPEG 流"""
        if self.cap:
            self.cap.release()
        self.cap = None
        self.stream_url = None
        urls = [p.format(ip=self.ip) for p in MJPEG_PATTERNS]
        for url in urls:
            print(f"  [MJPEG] 重新连接: {url}")
            self.cap = cv2.VideoCapture(url)
            if self.cap.isOpened():
                for _ in range(15):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"  [MJPEG] ✓ 重连成功: {url}")
                        self.stream_url = url
                        return True
                self.cap.release()
        self.cap = None
        print(f"  [MJPEG] ✗ 重连失败")
        return False

    def sync_to_latest(self):
        """同步到最新帧（解决 MJPEG buffer 延迟问题）"""
        if self.cap and self.cap.isOpened():
            try:
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_frames > 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            except Exception:
                pass

    def read_frame(self):
        """读取一帧（自动同步到最新帧，避免使用缓存的旧帧）"""
        if self.cap:
# 关键修复：每次读取前同步到最新帧
            self.sync_to_latest()
            return self.cap.read()
        return False, None

    def detect_faces(self, frame):
        """检测人脸"""
        if self.face_cascade is None or frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    def select_primary_face(self, faces) -> tuple:
        """选择主要人脸（最大的人脸）"""
        if len(faces) == 0:
            return None
        if len(faces) == 1:
            return tuple(faces[0])
        areas = [(x * y) for (x, y, w, h) in faces]
        max_idx = areas.index(max(areas))
        return tuple(faces[max_idx])

# HTTP 拍照轮询摄像头类
class HTTPCamera:
    """HTTP 拍照轮询摄像头（通过机器人拍照 API 获取帧）"""

    def __init__(self, ip: str):
        self.ip = ip
        self.base_url = f"http://{ip}:9090/v1"
        self._last_frame = None
        self._last_capture_time = 0
        self.capture_interval = 0.3  # 拍照间隔（秒）
        self.face_cascade = None
        self._available = True

        # 加载 Haar 级联分类器（使用非中文路径副本）
        self.face_cascade = _load_face_cascade()

        # 测试连接
        import requests
        try:
            resp = requests.get(f"{self.base_url}/sensors", timeout=3)
            if resp.status_code == 200:
                print(f"  [HTTP摄像头] ✓ 机器人 API 连接正常")
            else:
                print(f"  [HTTP摄像头] ⚠ API 返回状态码: {resp.status_code}")
        except Exception as e:
            print(f"  [HTTP摄像头] ✗ 无法连接到机器人 API: {e}")
            self._available = False

    def is_connected(self) -> bool:
        """检查 API 是否可用（实时验证）"""
        if not self._available:
            return False
        import requests
        try:
            resp = requests.get(f"{self.base_url}/sensors", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _capture_frame(self):
        """从机器人拍照并返回图像"""
        import requests
        import numpy as np

        try:
            # 拍照
            resp = requests.post(
                f"{self.base_url}/visions/photos",
                json={"resolution": HTTP_POLL_CONFIG["resolution"]},
                timeout=5
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            if data.get("code") != 0:
                return None

            photo_name = data.get("data", {}).get("name")
            if not photo_name:
                return None

            # 下载照片
            img_resp = requests.get(
                f"{self.base_url}/visions/photos?body={photo_name}",
                timeout=5
            )
            if img_resp.status_code != 200:
                return None

            # 解码图像
            nparr = np.frombuffer(img_resp.content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame

        except Exception as e:
            print(f"  [HTTP摄像头] 拍照失败: {e}")
            return None

    def read_frame(self, force_capture=False):
        """读取一帧（带缓存，force_capture=True 时强制重新拍照）"""
        import time
        current_time = time.time()

# 关键修复：force_capture=True 时强制重新拍照，避免使用缓存帧
        cache_valid = (self._last_frame is not None and 
                       (current_time - self._last_capture_time) < self.capture_interval and 
                           not force_capture)
        
        if not cache_valid:
            frame = self._capture_frame()
            if frame is not None:
                self._last_frame = frame
                self._last_capture_time = current_time

        if self._last_frame is not None:
            return True, self._last_frame.copy()  # 返回副本避免共享引用
        return False, None

    def clear_frame_cache(self):
        """清除帧缓存（强制下一次读取时重新拍照）"""
        self._last_frame = None
        self._last_capture_time = 0

    def detect_faces(self, frame):
        """检测人脸，返回所有人脸矩形列表"""
        if self.face_cascade is None or frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    def select_primary_face(self, faces) -> tuple:
        """选择主要人脸（最大的人脸）"""
        if len(faces) == 0:
            return None
        if len(faces) == 1:
            return tuple(faces[0])

        areas = [(x * y) for (x, y, w, h) in faces]
        max_idx = areas.index(max(areas))
        return tuple(faces[max_idx])

# RTSP 摄像头类
class RTSPCamera:
    """RTSP 视频流读取器（带超时保护）"""

    def __init__(self, ip: str, custom_url: str = None):
        self.ip = ip
        self.cap = None
        self.stream_url = None
        self.face_cascade = None

        # 加载 Haar 级联分类器（使用非中文路径副本）
        self.face_cascade = _load_face_cascade()

        # 预检 RTSP 端口可达性（快速跳过不可达地址）
        rtsp_ports = {554, 8554}
        port_reachable = False
        for port in rtsp_ports:
            try:
                sock = socket.create_connection((ip, port), timeout=2.0)
                sock.close()
                port_reachable = True
                break
            except (socket.timeout, ConnectionRefusedError, OSError):
                pass

        if not port_reachable:
            print(f"  [RTSP] 端口 554/8554 不可达，跳过 RTSP")
            self.cap = None
            return

        # 尝试连接
        urls = [custom_url] if custom_url else [
            p.format(ip=ip) for p in RTSP_PATTERNS
        ]

        for url in urls:
            print(f"  [RTSP] 尝试连接: {url}")
            self.cap = _try_video_capture(url, timeout=6.0)
            if self.cap is not None:
                print(f"  [RTSP] ✓ 连接成功: {url}")
                self.stream_url = url
                return

        self.cap = None
        print(f"  [RTSP] ✗ 无法连接到机器人摄像头")

    def is_connected(self) -> bool:
        if self.cap is None:
            return False
        if not self.cap.isOpened():
            return False
        ret, _ = self.cap.read()
        if not ret:
            return False
        # 重新 seek 到最新帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        return True

    def reconnect(self):
        """重新连接 RTSP 流"""
        if self.cap:
            self.cap.release()
        self.cap = None
        self.stream_url = None
        urls = [p.format(ip=self.ip) for p in RTSP_PATTERNS]
        for url in urls:
            print(f"  [RTSP] 重新连接: {url}")
            self.cap = cv2.VideoCapture(url)
            if self.cap.isOpened():
                for _ in range(10):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"  [RTSP] ✓ 重连成功: {url}")
                        self.stream_url = url
                        return True
                self.cap.release()
        self.cap = None
        print(f"  [RTSP] ✗ 重连失败")
        return False

    def sync_to_latest(self):
        """同步到最新帧（解决 RTSP buffer 延迟问题）"""
        if self.cap and self.cap.isOpened():
            try:
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_frames > 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            except Exception:
                pass

    def read_frame(self):
        """读取一帧（自动同步到最新帧，避免使用缓存的旧帧）"""
        if self.cap:
# 关键修复：每次读取前同步到最新帧
            self.sync_to_latest()
            return self.cap.read()
        return False, None

    def detect_faces(self, frame):
        """检测人脸，返回所有人脸矩形列表"""
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    def select_primary_face(self, faces) -> tuple:
        """选择主要人脸（最大的人脸）

        当有多个人脸时，选择面积最大的作为主要识别目标。
        原因：通常最大的人脸离摄像头最近，最可能是互动对象。

        Args:
            faces: 人脸矩形列表 [(x, y, w, h), ...]

        Returns:
            选中的人脸矩形 (x, y, w, h)，如果没有则返回 None
        """
        if len(faces) == 0:
            return None
        if len(faces) == 1:
            return tuple(faces[0])

        # 计算每个人脸的面积，选择最大的
        areas = [(x * y) for (x, y, w, h) in faces]
        max_idx = areas.index(max(areas))
        return tuple(faces[max_idx])

# DeepFace 分析
def analyze_face_deepface(face_roi) -> dict:
    """使用 DeepFace 分析人脸

    Args:
        face_roi: 人脸区域图像 (numpy array)

    Returns:
        {
            'success': bool,
            'age': int,
            'gender': str,       # "男" / "女"
            'emotion': str,       # 项目情绪体系
            'emotion_raw': str,   # DeepFace 原始情绪
            'all_emotions': dict,
            'confidence': float, # 识别置信度
        }
    """
    if not DEEPFACE_AVAILABLE:
        return {'success': False}

# 人脸区域基础检查：图像太小或太大都可能是误检
    if face_roi is None or face_roi.size == 0:
        return {'success': False}

    h, w = face_roi.shape[:2]
    # 人脸区域太小（小于40像素）或太大（超过原图90%）都视为异常
    if h < 40 or w < 40:
        print(f"  [DeepFace] 人脸区域过小({w}x{h})，跳过")
        return {'success': False}

    try:
        # 检测年龄、性别、情绪
        result = DeepFace.analyze(
            img_path=face_roi,
            actions=['age', 'gender', 'emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list) and result:
            data = result[0]

            # 映射性别
            gender_raw = data.get('dominant_gender', 'Unknown')
            gender = GENDER_MAP.get(gender_raw, "未知")

            # 映射情绪
            emotion_raw = data.get('dominant_emotion', 'neutral')
            emotion = DEEPFACE_TO_EMOTION_MAP.get(emotion_raw, "平静")

# 获取识别置信度：低于阈值认为是误检
            emotion_confidence = data.get('emotion', {}).get(emotion_raw, 0)
            gender_confidence = data.get('gender', {}).get(gender_raw, 0)
            min_confidence = 0.4  # 最低置信度阈值

            if emotion_confidence < min_confidence and gender_confidence < min_confidence:
                print(f"  [DeepFace] 置信度过低(emotion={emotion_confidence:.2f}, gender={gender_confidence:.2f})，跳过")
                return {'success': False}

            return {
                'success': True,
                'age': int(data.get('age', 25)),
                'gender': gender,
                'emotion': emotion,
                'emotion_raw': emotion_raw,
                'all_emotions': data.get('emotion', {}),
                'confidence': emotion_confidence,
            }
        elif isinstance(result, dict):
            # 单个结果的情况
            gender_raw = result.get('dominant_gender', 'Unknown')
            gender = GENDER_MAP.get(gender_raw, "未知")
            emotion_raw = result.get('dominant_emotion', 'neutral')
            emotion = DEEPFACE_TO_EMOTION_MAP.get(emotion_raw, "平静")

            emotion_confidence = result.get('emotion', {}).get(emotion_raw, 0)
            gender_confidence = result.get('gender', {}).get(gender_raw, 0)
            min_confidence = 0.4

            if emotion_confidence < min_confidence and gender_confidence < min_confidence:
                print(f"  [DeepFace] 置信度过低(emotion={emotion_confidence:.2f}, gender={gender_confidence:.2f})，跳过")
                return {'success': False}

            return {
                'success': True,
                'age': int(result.get('age', 25)),
                'gender': gender,
                'emotion': emotion,
                'emotion_raw': emotion_raw,
                'all_emotions': result.get('emotion', {}),
                'confidence': emotion_confidence,
            }
    except Exception as e:
        print(f"  [DeepFace] 分析失败: {e}")

    return {'success': False}

# 人脸识别器
class FaceRecognizer:
    """Yanshee 人脸识别器（DeepFace + RTSP/HTTP 双模式）"""

    def __init__(self, robot_ip: str = None, use_http_fallback: bool = True):
        self.ip = robot_ip or ROBOT_IP
        self.mjpeg_camera: Optional[MJPEGCamera] = None
        self.camera: Optional[RTSPCamera] = None
        self.http_camera: Optional[HTTPCamera] = None
        self._camera_mode = None  # "mjpeg" / "rtsp" / "http"
        self._last_result: Optional[Dict] = None
        self._last_detect_time: float = 0
        self._cache_duration: float = 3.0  # 缓存3秒
        self._use_http_fallback = use_http_fallback
# 新增：用于存储"拍摄"的当前帧，检测到人脸时捕获，情绪跳转后清除
        self._captured_frame: Optional[np.ndarray] = None
        self._captured_frame_time: float = 0

# 连续帧验证：只有连续多帧检测结果一致才认为有效
        self._frame_detection_history: List[Dict] = []  # 连续帧检测结果历史
        self._consecutive_frames_required = 3  # 需要连续多少帧一致才认为有效
        self._frame_validity_window = 3.0  # 帧的有效时间窗口（秒），超过则历史失效

        # 连接摄像头
        self._connect()

    def _connect(self):
        """连接摄像头（优先 MJPEG，其次 RTSP，最后 HTTP 轮询）"""
        if not DEEPFACE_AVAILABLE:
            print("  [人脸识别] ✗ DeepFace 不可用，请安装: pip install deepface")
            return

        print("\n  [人脸识别] 连接机器人摄像头 (IP: {})".format(self.ip))

        # 1. 优先尝试 MJPEG 模式（我们部署的 MJPEG 服务器）
        print("  [人脸识别] 尝试 MJPEG 模式...")
        self.mjpeg_camera = MJPEGCamera(ip=self.ip)

        if self.mjpeg_camera.is_connected():
            print("  [人脸识别] ✓ MJPEG 连接成功")
            self._camera_mode = "mjpeg"
            return

        print("  [人脸识别] ✗ MJPEG 连接失败")

        # 2. 尝试 RTSP 模式
        print("  [人脸识别] 尝试 RTSP 模式...")
        self.camera = RTSPCamera(ip=self.ip)

        if self.camera.is_connected():
            print("  [人脸识别] ✓ RTSP 连接成功")
            self._camera_mode = "rtsp"
            return

        print("  [人脸识别] ✗ RTSP 连接失败")

        # 3. 降级到 HTTP 轮询模式
        if self._use_http_fallback:
            print("  [人脸识别] 尝试 HTTP 轮询模式（通过拍照 API）...")
            self.http_camera = HTTPCamera(ip=self.ip)

            if self.http_camera.is_connected():
                print("  [人脸识别] ✓ HTTP 轮询模式已启用")
                self._camera_mode = "http"
                return

            print("  [人脸识别] ✗ HTTP 模式也无法连接")

        print("  [人脸识别] ✗ 无法连接任何摄像头模式")

    def is_available(self) -> bool:
        """检查识别器是否可用"""
        if not DEEPFACE_AVAILABLE:
            return False
        if self._camera_mode == "mjpeg":
            return self.mjpeg_camera is not None and self.mjpeg_camera.is_connected()
        elif self._camera_mode == "rtsp":
            return self.camera is not None and self.camera.is_connected()
        elif self._camera_mode == "http":
            return self.http_camera is not None and self.http_camera.is_connected()
        return False

    def _get_camera(self):
        """获取当前活动的摄像头"""
        if self._camera_mode == "mjpeg":
            return self.mjpeg_camera
        elif self._camera_mode == "rtsp":
            return self.camera
        elif self._camera_mode == "http":
            return self.http_camera
        return None

    def detect(self, use_cache: bool = True, force_new: bool = False) -> Dict:
        """检测人脸信息（单脸模式，优先选择最大人脸）

        Args:
            use_cache: 是否使用缓存（默认True）
            force_new: 是否强制全新检测（忽略缓存，默认False）

        Returns:
            {
                "success": bool,
                "age": int,
                "gender": str,
                "emotion": str,
                "emotion_raw": str,
                "all_emotions": dict,
                "face_count": int,
                "source": str,
            }
        """
        # 检查缓存
        current_time = time.time()
        if use_cache and not force_new and self._last_result and (current_time - self._last_detect_time) < self._cache_duration:
            result = self._last_result.copy()
            result["source"] = "cache"
            return result

        # 检查连接
        if not self.is_available():
            return self._empty_result()

        try:
            # 读取一帧
            camera = self._get_camera()
            if camera is None:
                return self._empty_result()

# 关键修复：force_new=True 时清除摄像头帧缓存，强制读取最新帧
            if force_new:
                if hasattr(camera, 'clear_frame_cache'):
                    camera.clear_frame_cache()
                elif hasattr(camera, 'sync_to_latest'):
                    camera.sync_to_latest()

            ret, frame = camera.read_frame()
            if not ret or frame is None:
                # 尝试重连
                print("  [人脸识别] 流读取失败，尝试重连...")
                if hasattr(camera, 'reconnect') and camera.reconnect():
                    ret, frame = camera.read_frame()
                    if not ret or frame is None:
                        print("  [人脸识别] 重连后仍无法读取帧")
                        return self._empty_result()
                else:
                    print("  [人脸识别] 无法读取视频帧")
                    return self._empty_result()

            # 检测人脸
            faces = camera.detect_faces(frame)
            face_count = len(faces)

            if face_count == 0:
                # Haar 没检测到人脸，清空历史，重新开始计数
                if self._frame_detection_history:
                    print(f"  [连续帧] Haar未检测到人脸，清空历史")
                    self._frame_detection_history = []
                return self._empty_result()

            # 选择主要人脸（最大的）
            primary_face = camera.select_primary_face(faces)
            if primary_face is None:
                return self._empty_result()

            x, y, w, h = primary_face
            
# 人脸面积检查：避免误检背景中的非人脸形状
            # 人脸面积小于 100x100（约 10000 像素）时认为是误检，忽略
            min_face_area = 100 * 100  # 最小有效人脸面积
            face_area = w * h
            if face_area < min_face_area:
                print(f"  [人脸识别] 人脸过小({w}x{h}={face_area}px)，跳过")
                # 面积过小，清空历史
                self._frame_detection_history = []
                return self._empty_result()

# 人脸长宽比检查：真人脸接近正方形，非人脸形状比例通常极端
            # 人脸宽高比应该在 0.7~1.4 之间（允许一定变形）
            aspect_ratio = w / h if h > 0 else 0
            if not (0.65 <= aspect_ratio <= 1.5):
                print(f"  [人脸识别] 人脸比例异常({w}/{h}={aspect_ratio:.2f})，跳过")
                # 比例异常，清空历史
                self._frame_detection_history = []
                return self._empty_result()

# 关键逻辑：检测到人脸时，立即"拍摄"当前帧
            # 这样确保用于分析的是检测到人脸那一刻的图像，而非后续可能变化的帧
            self._captured_frame = frame.copy()
            self._captured_frame_time = current_time
            print(f"  [帧捕获] ★ 检测到有效人脸，已拍摄当前帧（shape: {self._captured_frame.shape}, 人脸: {w}x{h}）")

# 使用已捕获的帧进行人脸区域提取和分析
            # 提取人脸区域（添加边距）
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(self._captured_frame.shape[1], x + w + margin)
            y2 = min(self._captured_frame.shape[0], y + h + margin)
            face_roi = self._captured_frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                # 人脸区域无效，清空历史
                self._frame_detection_history = []
                return self._empty_result()

            # DeepFace 分析
            analysis = analyze_face_deepface(face_roi)

            if not analysis['success']:
                # DeepFace 分析失败，清空历史
                self._frame_detection_history = []
                return self._empty_result()

            # 构建初步结果（包含人脸位置用于连续帧比较）
            raw_result = {
                "success": True,
                "age": analysis['age'],
                "gender": analysis['gender'],
                "emotion": analysis['emotion'],
                "emotion_raw": analysis['emotion_raw'],
                "all_emotions": analysis['all_emotions'],
                "face_count": face_count,
                "source": "deepface",
                "face_rect": (x, y, w, h),  # 人脸位置用于连续帧比较
            }

# 连续帧验证：只有连续5帧一致才认为有效
            # 更新摄像头帧缓存
            if hasattr(camera, 'clear_frame_cache'):
                camera.clear_frame_cache()

            stable_result = self._get_stable_result(raw_result)

            if stable_result is None:
                # 未达到连续帧要求，返回空结果（但不更新缓存，让下次检测继续计数）
                return self._empty_result()

            # 达到稳定，更新缓存并返回结果
            self._last_result = stable_result
            self._last_detect_time = current_time

            face_info = f"age={stable_result['age']}, gender={stable_result['gender']}, emotion={stable_result['emotion']}"
            if face_count > 1:
                print(f"  [人脸识别] ✓ 检测成功({face_count}人中选最大): {face_info}")
            else:
                print(f"  [人脸识别] ✓ 检测成功: {face_info}")

            return stable_result

        except Exception as e:
            print(f"  [人脸识别] ✗ 检测异常: {e}")
            return self._empty_result()

    def detect_all_faces(self) -> List[Dict]:
        """检测所有人脸信息（多脸模式）

        当画面中有多个用户时，识别每一个人脸。

        Returns:
            [
                {
                    "success": bool,
                    "face_index": int,  # 人脸索引（0=最大的, 1=第二大的, ...）
                    "age": int,
                    "gender": str,
                    "emotion": str,
                    "emotion_raw": str,
                    "all_emotions": dict,
                    "face_area": int,  # 人脸面积（用于排序）
                },
                ...
            ]
        """
        if not self.is_available():
            return []

        try:
            # 读取一帧
            camera = self._get_camera()
            if camera is None:
                return []

            ret, frame = camera.read_frame()
            if not ret or frame is None:
                # 尝试重连
                if hasattr(camera, 'reconnect') and camera.reconnect():
                    ret, frame = camera.read_frame()
                    if not ret or frame is None:
                        return []
                else:
                    return []

            # 检测人脸
            faces = camera.detect_faces(frame)
            if len(faces) == 0:
                return []

            # 按面积排序（从大到小）
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            results = []
            for idx, (x, y, w, h) in enumerate(faces_sorted):
# 过滤检查：面积和长宽比
                min_face_area = 100 * 100  # 最小有效人脸面积
                face_area = w * h
                if face_area < min_face_area:
                    continue  # 跳过面积过小的人脸

                aspect_ratio = w / h if h > 0 else 0
                if not (0.65 <= aspect_ratio <= 1.5):
                    continue  # 跳过比例异常的人脸

                # 提取人脸区域
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size == 0:
                    continue

                # DeepFace 分析
                analysis = analyze_face_deepface(face_roi)

                if analysis['success']:
                    results.append({
                        "success": True,
                        "face_index": idx,
                        "age": analysis['age'],
                        "gender": analysis['gender'],
                        "emotion": analysis['emotion'],
                        "emotion_raw": analysis['emotion_raw'],
                        "all_emotions": analysis['all_emotions'],
                        "face_area": w * h,
                    })

            if len(results) > 0:
                print(f"  [人脸识别] ✓ 检测到 {len(results)} 个人脸")

            return results

        except Exception as e:
            print(f"  [人脸识别] ✗ 多脸检测异常: {e}")
            return []

    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            "success": False,
            "age": None,
            "gender": None,
            "emotion": None,
            "emotion_raw": None,
            "all_emotions": {},
            "face_count": 0,
            "source": "none",
        }

    def _is_similar_result(self, r1: Dict, r2: Dict) -> bool:
        """检查两次检测结果是否相似（可能是同一个人）

        判断标准：
        - 性别必须相同
        - 年龄差距不超过10岁（允许检测误差）
        - 人脸位置差距不超过30%（允许轻微移动）
        """
        if not r1 or not r2:
            return False

        # 性别必须相同
        g1 = r1.get('gender', '未知')
        g2 = r2.get('gender', '未知')
        if g1 != g2 or g1 == '未知' or g2 == '未知':
            return False

        # 年龄差距不超过10岁
        age1 = r1.get('age', 0)
        age2 = r2.get('age', 0)
        if age1 > 0 and age2 > 0 and abs(age1 - age2) > 10:
            return False

        # 人脸位置差距不超过30%（检查中心点偏移）
        face1 = r1.get('face_rect')
        face2 = r2.get('face_rect')
        if face1 and face2:
            x1, y1, w1, h1 = face1
            x2, y2, w2, h2 = face2
            # 计算中心点
            cx1, cy1 = x1 + w1/2, y1 + h1/2
            cx2, cy2 = x2 + w2/2, y2 + h2/2
            # 归一化偏移量
            avg_w = (w1 + w2) / 2
            avg_h = (h1 + h2) / 2
            if avg_w > 0 and avg_h > 0:
                dx = abs(cx1 - cx2) / avg_w
                dy = abs(cy1 - cy2) / avg_h
                # 水平或垂直偏移超过30%认为不是同一个人
                if dx > 0.3 or dy > 0.3:
                    return False

        return True

    def _get_stable_result(self, new_result: Dict) -> Optional[Dict]:
        """获取稳定结果：只有连续多帧检测一致才返回有效结果

        逻辑：
        1. 如果新结果与历史中最后一帧不相似，清空历史，重新开始计数
        2. 将新结果加入历史
        3. 如果连续5帧都相似，返回稳定结果
        4. 否则返回None（表示还未达到稳定状态）
        """
        current_time = time.time()

        # 检查历史是否过期（超过时间窗口则清空）
        if self._frame_detection_history:
            first_frame_time = self._frame_detection_history[0].get('_timestamp', 0)
            if current_time - first_frame_time > self._frame_validity_window:
                print(f"  [连续帧] 历史过期（{current_time - first_frame_time:.1f}秒），重新计数")
                self._frame_detection_history = []

        # 检查新结果是否与历史最后一帧相似
        if self._frame_detection_history:
            last_result = self._frame_detection_history[-1]
            if not self._is_similar_result(last_result, new_result):
                # 不相似，清空历史，重新开始计数
                print(f"  [连续帧] 检测结果变化，清空历史重新计数")
                self._frame_detection_history = []

        # 将新结果加入历史
        new_result['_timestamp'] = current_time
        self._frame_detection_history.append(new_result)

        # 保持历史长度不超过所需帧数
        while len(self._frame_detection_history) > self._consecutive_frames_required:
            self._frame_detection_history.pop(0)

        # 检查是否达到稳定
        if len(self._frame_detection_history) >= self._consecutive_frames_required:
            # 取最后一帧作为稳定结果
            stable = self._frame_detection_history[-1].copy()
            stable.pop('_timestamp', None)
            print(f"  [连续帧] ★★★ 连续{self._consecutive_frames_required}帧一致，判定有效！")
            return stable

        # 未达到稳定状态
        print(f"  [连续帧] 累计{len(self._frame_detection_history)}/{self._consecutive_frames_required}帧...")
        return None

    def get_cached_result(self) -> Optional[Dict]:
        """获取缓存的识别结果"""
        return self._last_result if self._last_result else None

    def clear_cache(self):
        """清除缓存的识别结果（用于人脸情绪跳转后避免重复触发）
        
        同时清除：
        1. FaceRecognizer 的结果缓存
        2. 摄像头的帧缓存
        3. ★★★ 已捕获的当前帧（情绪跳转后必须清除，确保下次检测重新拍摄）★★★
        """
        # 清除识别结果缓存
        self._last_result = None
        self._last_detect_time = 0
        
# 关键修复：清除已捕获的当前帧
        # 情绪跳转后必须清除已拍摄的帧，确保下次检测重新拍摄新图像
        if self._captured_frame is not None:
            print(f"  [帧清除] ★ 清除已捕获的帧（shape: {self._captured_frame.shape}）")
        self._captured_frame = None
        self._captured_frame_time = 0

# 清除连续帧验证历史
        if self._frame_detection_history:
            print(f"  [帧清除] ★ 清除连续帧历史（{len(self._frame_detection_history)}帧）")
        self._frame_detection_history = []
        
        # 同时清除摄像头的帧缓存，避免读取到旧帧
        camera = self._get_camera()
        if camera:
            if hasattr(camera, 'clear_frame_cache'):
                camera.clear_frame_cache()
            elif hasattr(camera, 'sync_to_latest'):
                camera.sync_to_latest()
            elif hasattr(camera, 'cap') and camera.cap:
                # 对于 MJPEG 流，强制 seek 到最新帧
                try:
                    total_frames = camera.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if total_frames > 1:
                        camera.cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                except Exception:
                    pass
        self._last_detect_time = 0

    def release(self):
        """释放资源"""
        if self.mjpeg_camera and self.mjpeg_camera.cap:
            self.mjpeg_camera.cap.release()
        if self.camera and self.camera.cap:
            self.camera.cap.release()
        if self.http_camera:
            self.http_camera = None

# 人脸识别管理器
class FaceRecognitionManager:
    """人脸识别管理器（支持并行启动）

    多脸识别策略：
    - 单脸模式 (detect): 检测所有人脸，选择最大的一个进行情绪识别
    - 多脸模式 (detect_all_faces): 识别画面中所有的人脸

    适用场景：
    - 单人对话：使用 detect() 获得主要用户的情绪
    - 多人场景：使用 detect_all_faces() 获取所有人的情绪，
      然后根据业务逻辑选择（比如悲伤程度最高的）
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, robot_ip: str = None):
        if not hasattr(self, "_initialized"):
            self._recognizer: Optional[FaceRecognizer] = None
            self._robot_ip = robot_ip
            self._started = False
            self._initialized = True

    def start(self, robot_ip: str = None):
        """启动人脸识别（开机时调用）"""
        if self._started:
            return

        ip = robot_ip or self._robot_ip or ROBOT_IP

        print(f"\n  [人脸识别] 启动摄像头... (IP: {ip})")
        self._recognizer = FaceRecognizer(robot_ip=ip)

        if self._recognizer.is_available():
            print("  [人脸识别] ✓ 摄像头已就绪 (DeepFace + RTSP)")
        else:
            print("  [人脸识别] ⚠ 摄像头不可用，将使用原有识别方式")

        self._started = True

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._started and self._recognizer is not None and self._recognizer.is_available()

    def detect(self, force_new: bool = False) -> Dict:
        """检测人脸信息（单脸模式）
        
        Args:
            force_new: 是否强制全新检测（忽略缓存，默认False）
        """
        if not self.is_available():
            return self._empty_result()
        return self._recognizer.detect(force_new=force_new)

    def detect_all_faces(self) -> List[Dict]:
        """检测所有人脸（多脸模式）"""
        if not self.is_available():
            return []
        return self._recognizer.detect_all_faces()

    def detect_with_priority(self, keyword_emotion: str = None,
                            keyword_confidence: float = 0.0,
                            content_emotion: str = None,
                            user_text: str = None) -> Dict:
        """带优先级的识别

        优先级规则:
        1. 文本含high关键词 → high级别
        2. 人脸识别成功 → 使用其结果
        3. 人脸失败 + 关键词成功 → 使用关键词结果
        4. 都失败 → 使用内容分析结果
        5. 都没有 → 使用默认（称呼"朋友"）

        Returns:
            包含 emotion_level 字段: "high" / "mid_high" / "mid" / "low" / "positive"
        """
        result = {
            "success": False,
            "emotion": None,
            "emotion_level": "mid",  # 默认中等级别
            "gender": None,
            "age": None,
            "address": "朋友",
            "voice": "x4_yezi",
            "source": "default",
            "face_result": None,
        }

        # 第0优先级: 检测high级别关键词（摄像头识别不了的极度负面情绪）
        if user_text and detect_high_level_keywords(user_text):
            result["success"] = True
            result["emotion"] = "绝望"  # 用"绝望"代表high级别
            result["emotion_level"] = "high"
            result["source"] = "keyword_high"
            print(f"  [人脸识别] ✓ 检测到high级别关键词: 绝望")
            return result

        # 第1优先级: 人脸识别
        if self.is_available():
            face_result = self._recognizer.detect()
            if face_result.get("success"):
                result["success"] = True
                result["emotion"] = face_result.get("emotion")
                result["gender"] = face_result.get("gender")
                result["age"] = face_result.get("age")
                result["face_result"] = face_result
                result["source"] = "face"
                # 根据摄像头识别的情绪获取对应强度等级
                result["emotion_level"] = get_emotion_level_from_face(result["emotion"])
                result["address"] = self._generate_address(face_result.get("gender"), face_result.get("age"))
                result["voice"] = self._select_voice(face_result.get("gender"), face_result.get("age"))
                return result

        # 第2优先级: 关键词识别
        if keyword_emotion and keyword_confidence >= 0.6:
            result["success"] = True
            result["emotion"] = keyword_emotion
            result["source"] = "keyword"
            # 从关键词情绪获取对应强度等级
            from config import get_emotion_level
            result["emotion_level"] = get_emotion_level(keyword_emotion)
            return result

        # 第3优先级: 内容分析
        if content_emotion:
            result["success"] = True
            result["emotion"] = content_emotion
            result["source"] = "content"
            from config import get_emotion_level
            result["emotion_level"] = get_emotion_level(content_emotion)
            return result

        return result

    def detect_all_with_priority(self, keyword_emotion: str = None,
                                 keyword_confidence: float = 0.0,
                                 content_emotion: str = None,
                                 user_text: str = None) -> Dict:
        """多脸模式 + 优先级识别

        当检测到多个人脸时，选择"最需要安慰"的一个：
        1. 优先选择有负面情绪的人脸（悲伤 > 愤怒 > 恐惧 > 厌恶 > 平静）
        2. 负面情绪程度相同时，选择最大的那个人脸

        强度等级（5级）:
        - high: 绝望/崩溃等（需要关键词识别）
        - mid_high: 悲伤/愤怒/恐惧
        - mid: 厌恶/惊讶
        - low: 心烦/无奈/孤单/麻木
        - positive: 开心/平静

        Returns:
            同 detect_with_priority，但 source 可能为 "face_primary" / "face_worst" / "keyword_high"
        """
        result = {
            "success": False,
            "emotion": None,
            "emotion_level": "mid",
            "gender": None,
            "age": None,
            "address": "朋友",
            "voice": "x4_yezi",
            "source": "default",
            "face_result": None,
            "all_faces": [],
        }

        # 第0优先级: 检测high级别关键词
        if user_text and detect_high_level_keywords(user_text):
            result["success"] = True
            result["emotion"] = "绝望"
            result["emotion_level"] = "high"
            result["source"] = "keyword_high"
            print(f"  [人脸识别] ✓ 检测到high级别关键词: 绝望")
            return result

        if not self.is_available():
            # 回退到关键词/内容分析
            if keyword_emotion and keyword_confidence >= 0.6:
                result["success"] = True
                result["emotion"] = keyword_emotion
                result["source"] = "keyword"
                from config import get_emotion_level
                result["emotion_level"] = get_emotion_level(keyword_emotion)
            elif content_emotion:
                result["success"] = True
                result["emotion"] = content_emotion
                result["source"] = "content"
                from config import get_emotion_level
                result["emotion_level"] = get_emotion_level(content_emotion)
            return result

        # 检测所有人脸
        all_faces = self._recognizer.detect_all_faces()
        result["all_faces"] = all_faces

        if len(all_faces) == 0:
            # 无人脸，回退到其他识别方式
            if keyword_emotion and keyword_confidence >= 0.6:
                result["success"] = True
                result["emotion"] = keyword_emotion
                result["source"] = "keyword"
                from config import get_emotion_level
                result["emotion_level"] = get_emotion_level(keyword_emotion)
            elif content_emotion:
                result["success"] = True
                result["emotion"] = content_emotion
                result["source"] = "content"
                from config import get_emotion_level
                result["emotion_level"] = get_emotion_level(content_emotion)
            return result

        # 多脸识别：选择"最需要安慰"的人脸
        # 负面情绪等级（越高越需要安慰）
        negative_levels = {
            "悲伤": 5,
            "愤怒": 4,
            "恐惧": 3,
            "厌恶": 2,
            "平静": 1,
            "开心": 0,
            "惊讶": 0,
        }

        # 计算每张脸的"需要安慰指数"
        def calc_comfort_need(face):
            emotion = face.get("emotion", "平静")
            level = negative_levels.get(emotion, 0)
            area = face.get("face_area", 0)
            # 公式: 情绪等级 * 100000 + 面积
            # 确保负面情绪优先，面积作为次要排序依据
            return level * 100000 + area

        # 找出最需要安慰的人脸
        all_faces_sorted = sorted(all_faces, key=calc_comfort_need, reverse=True)
        primary_face = all_faces_sorted[0]

        result["success"] = True
        result["emotion"] = primary_face.get("emotion")
        result["gender"] = primary_face.get("gender")
        result["age"] = primary_face.get("age")
        result["face_result"] = primary_face
        result["source"] = "face_worst" if len(all_faces) > 1 else "face_primary"
        # 根据摄像头识别的情绪获取对应强度等级
        result["emotion_level"] = get_emotion_level_from_face(result["emotion"])
        result["address"] = self._generate_address(primary_face.get("gender"), primary_face.get("age"))
        result["voice"] = self._select_voice(primary_face.get("gender"), primary_face.get("age"))

        # 打印多脸信息
        if len(all_faces) > 1:
            faces_summary = ", ".join([
                f"#{f['face_index']+1}({f['emotion']})" for f in all_faces_sorted[:3]
            ])
            print(f"  [人脸识别] {len(all_faces)}人: {faces_summary}" + ("..." if len(all_faces) > 3 else ""))

        return result

    @staticmethod
    def _generate_address(gender: str, age: int) -> str:
        """根据性别和年龄生成称呼"""
        if not gender or not age:
            return "朋友"
        if gender == "未知" or age is None or age <= 0:
            return "朋友"

        ROBOT_AGE = 18
        if age < ROBOT_AGE:
            return "小弟弟" if gender == "男" else "小妹妹"
        elif age < ROBOT_AGE + 18:  # 36岁以下：小哥哥/小姐姐
            return "小哥哥" if gender == "男" else "小姐姐"
        elif age < 60:  # 36-59岁：叔叔/阿姨
            return "叔叔" if gender == "男" else "阿姨"
        else:
            return "爷爷" if gender == "男" else "奶奶"

    @staticmethod
    def _select_voice(gender: str, age: int) -> str:
        """统一使用椰子音色（温柔女声）"""
        return "x4_yezi"

    def _empty_result(self) -> Dict:
        return {
            "success": False,
            "emotion": None,
            "emotion_level": "mid",
            "gender": None,
            "age": None,
            "address": "朋友",
            "voice": "x4_yezi",
            "source": "none",
            "face_result": None,
            "all_faces": [],
        }

    def clear_cache(self):
        """清除缓存的识别结果（避免重复触发）"""
        if self._recognizer:
            self._recognizer.clear_cache()

    def release(self):
        """释放资源"""
        if self._recognizer:
            self._recognizer.release()

# 单例访问函数
_face_manager: Optional[FaceRecognitionManager] = None

def get_face_manager() -> FaceRecognitionManager:
    """获取人脸识别管理器单例"""
    global _face_manager
    if _face_manager is None:
        _face_manager = FaceRecognitionManager()
    return _face_manager

def init_face_recognition(robot_ip: str = None):
    """初始化人脸识别（开机时调用）"""
    manager = get_face_manager()
    manager.start(robot_ip)
    return manager

def detect_with_priority(keyword_emotion: str = None,
                        keyword_confidence: float = 0.0,
                        content_emotion: str = None) -> Dict:
    """带优先级的识别（快捷函数）"""
    manager = get_face_manager()
    return manager.detect_with_priority(keyword_emotion, keyword_confidence, content_emotion)
