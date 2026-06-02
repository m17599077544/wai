#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot类：Yanshee机器人HTTP/传感器控制、动作执行、摔倒检测线程"""
import os, time, threading, random, wave, requests,json,urllib,socket
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from config import (
    BASE_URL, DANCE_TIMEOUT, DANCE_STOP_TIMEOUT,
    MOTION_TIME_TABLE, MOTION_TIME_DEFAULT,
    RECOVERY_FRONT, RECOVERY_REAR, RECOVERY_MAX_ATTEMPTS,
    FRONT_GETUP_ASSIST_ANGLES, FRONT_GETUP_ASSIST_RUNTIME, FRONT_GETUP_ASSIST_DELAY,
    FALL_ACCEL_Y_FAST, FALL_EULER_DEVIATION_FAST, FALL_ACCEL_TOTAL_MIN,
    FALL_SEVERE_EULER_DEVIATION, FALL_SEVERE_ACCEL_Y,
    FALL_IMPACT_DELTA, FALL_CONFIRM_FRAMES, FALL_CHECK_INTERVAL,
    FALL_DANCE_WIN_SIZE, FALL_DANCE_VAR_TH, FALL_DANCE_GYRO_MEAN_MAX,
    CF_GYRO_WEIGHT, CF_ACCEL_LPF, CF_STAND_FRAMES, CF_STAND_MIN, CF_STAND_MAX,
    FRONT_FALL_EULER_MAX, REAR_FALL_EULER_MIN, REAR_FALL_EULER_POSITIVE,
    STAND_CONFIRM_FRAMES, STAND_ACCEL_Y_MIN, STAND_EULER_X_MIN, STAND_EULER_X_MAX,
    INIT_MODE_ENABLED, INIT_STAND_ACCEL_Y, INIT_STAND_EULER_MIN, INIT_STAND_EULER_MAX,
    INIT_STAND_FRAMES, INIT_ACCEL_Y_THRESHOLD, INIT_ACCEL_X_THRESHOLD,
    INIT_ACCEL_TOTAL_MIN, INIT_EULER_DEVIATION, INIT_CONFIRM_FRAMES,
    DANCE_STOP_WIN_SIZE, DANCE_STOP_GYRO_VAR_MAX, DANCE_STOP_GYRO_MEAN_MAX,
    DANCE_STOP_EULER_MIN, DANCE_STOP_EULER_MAX, DANCE_STOP_CONFIRM_FRAMES,
    DANCE_STOP_CHECK_INTERVAL,
)
from config import (
    ROBOT_IP, ROBOT_PORT, MOTION_CN_MAP, EMOTION_VOLUME,
    DANCE_MOTION_POOL, VOICE_DESCRIPTIONS, get_voice,
)
from config.tts_config import COMFORT_PITCH, COMFORT_SPEED, USE_XUNFEI_TTS
from voice import XunfeiTTS

# Robot 本地常量
STAND_EULER_MIN = 75.0
STAND_EULER_MAX = 115.0
MIN_DANCE_TIME = 5.0          # 启动保护：前5秒不判定静止
API_CHECK_EVERY = 10          # 每10次传感器读取检查1次API
API_IDLE_CONFIRM = 3          # API连续idle+motion_name空 确认次数
STATE_CHECK_INTERVAL = 0.5
IDLE_CONFIRM_NEEDED = 2
STAND_UP_BY_LEVEL = {
    "high": ["Reset"],
    "mid_high": ["Reset"],
    "mid": ["Reset", "reset_without_head"],
    "low": ["reset_without_head"],
    "positive": ["Reset"],
}
BOW_ANGLES = {
    "LeftHipFB": 106, "LeftHipLR": 90, "LeftKneeFlex": 53,
    "LeftAnkleFB": 90, "LeftAnkleUD": 90,
    "RightHipFB": 74, "RightHipLR": 90, "RightKneeFlex": 127,
    "RightAnkleFB": 90, "RightAnkleUD": 90,
    "LeftShoulderFlex": 0, "LeftShoulderRoll": 90, "LeftElbowFlex": 90,
    "RightShoulderFlex": 170, "RightShoulderRoll": 0, "RightElbowFlex": 170,
    "NeckLR": 90, "NeckUD": 90,
}

# Robot 类（含动作控制）
_tts_engine = None
_tts_cache_dir = os.path.join(SCRIPT_DIR, "tts_cache")
os.makedirs(_tts_cache_dir, exist_ok=True)

def _get_tts():
    global _tts_engine
    if _tts_engine is None: _tts_engine = XunfeiTTS()
    return _tts_engine

class Robot:
    def __init__(self):
        self.base = BASE_URL
        self.ip = ROBOT_IP
        self.port = ROBOT_PORT
        self.is_tts = False
        self.last_tts = 0

    def req(self, method, ep, data=None, params=None, timeout=10):
        url = f"{self.base}{ep}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        body = json.dumps(data).encode("utf-8") if data else None
        last_err = None
        for attempt in range(3):
            # 每次重试创建新的Request对象，避免超时后socket状态污染
            r = urllib.request.Request(url, data=body, method=method)
            if body:
                r.add_header("Content-Type", "application/json; charset=utf-8")
            try:
                with urllib.request.urlopen(r, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="ignore")
                try:
                    return json.loads(detail)
                except Exception:
                    return {"code": e.code, "msg": "HTTP Error"}
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(0.3)
                continue
        return {"code": -1, "msg": str(last_err)}

    def set_volume(self, volume):
        for ep, data in [("/media/volume", {"volume": volume}), ("/voice/volume", {"volume": volume})]:
            try:
                if self.req("PUT", ep, data, timeout=5).get("code") == 0: return True
            except Exception:
                pass
        return False

    def speak(self, text, emotion="", speed=None, _bot=None, voice=None):
        """TTS播报，返回实际时长。_bot传入ComfortBot实例时使用可中断等待。voice可指定音色覆盖默认。"""
        self.is_tts = True
        voice = voice or get_voice(emotion)
        # v110：音量范围68~85
        volume = max(68, min(85, EMOTION_VOLUME.get(emotion, 80)))
        self.set_volume(volume)
        tts_speed = speed if speed is not None else COMFORT_SPEED
        print(f"  [TTS] 情绪:{emotion} 音色:{VOICE_DESCRIPTIONS.get(voice, voice)} 音量:{volume}")
        print(f'  [TTS] "{text[:60]}"')
        if USE_XUNFEI_TTS:
            wav = _get_tts().synthesize(text, voice=voice, speed=tts_speed, volume=volume, pitch=COMFORT_PITCH)
            if wav:
                tmp = os.path.join(_tts_cache_dir, f"tts_{int(time.time())}.wav")
                with open(tmp, "wb") as f:
                    f.write(wav)
                upload_result = self.upload(tmp)
                if upload_result.get("code") != 0:
                    print(f"  [TTS] 上传失败，降级内置TTS播报")
                    self.is_tts = False
                    self.last_tts = time.time()
                    self.req("PUT", "/voice/tts", {"tts": text, "interrupt": True}, timeout=10)
                    dur = max(len(text) / 3.5, 2.0)
                    if _bot and hasattr(_bot, '_interruptible_sleep'):
                        _bot._interruptible_sleep(dur)
                    else:
                        time.sleep(dur)
                    self.is_tts = False
                    self.last_tts = time.time()
                    return dur
                self.play(os.path.basename(tmp))
                try:
                    with wave.open(tmp, 'rb') as w:
                        dur = w.getnframes() / float(w.getframerate())
                except Exception:
                    dur = max(len(text) / 3.5, 2.0)
                # 可中断等待TTS播放（摔倒时自动处理）
                if _bot and hasattr(_bot, '_interruptible_sleep'):
                    _bot._interruptible_sleep(dur + 0.1)
                else:
                    time.sleep(dur + 0.1)
                self.stop()
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                self.is_tts = False;
                self.last_tts = time.time();
                return dur
            print("  [TTS] 讯飞失败，降级内置TTS")
        self.req("PUT", "/voice/tts", {"tts": text, "interrupt": True}, timeout=10)
        dur = max(len(text) / 3.5, 2.0);
        if _bot and hasattr(_bot, '_interruptible_sleep'):
            _bot._interruptible_sleep(dur)
        else:
            time.sleep(dur)
        self.is_tts = False;
        self.last_tts = time.time();
        return dur

    def upload(self, path):
        ext = os.path.splitext(path)[1].lower()
        mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
        last_err = None
        for attempt in range(3):
            try:
                with open(path, "rb") as f:
                    return requests.post(f"{self.base}/media/music",
                                         files={"file": (os.path.basename(path), f, mime)},
                                         timeout=30).json()
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
                print(f"  [上传] 第{attempt+1}次失败（{type(e).__name__}），{'重试...' if attempt < 2 else '放弃'}")
                if attempt < 2:
                    time.sleep(1.0)
        print(f"  [上传] 上传失败已跳过: {last_err}")
        return {"code": -1, "msg": str(last_err)}

    def play(self, name):
        r = self.req("PUT", "/media/music", {"operation": "start", "name": name})
        if r.get("code") == 0: return True
        # 失败重试一次
        time.sleep(0.3)
        return self.req("PUT", "/media/music", {"operation": "start", "name": name}).get("code") == 0

    def stop(self):
        self.req("PUT", "/media/music", {"name": "", "operation": "stop"})

    def check_connection(self, timeout=3):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            ok = sock.connect_ex((self.ip, self.port)) == 0;
            sock.close();
            return ok
        except Exception:
            return False

    def get_gyro(self):
        """读取九轴陀螺仪数据"""
        r = self.req("GET", "/sensors/gyro", timeout=5)
        if r.get("code") == 0:
            gyro_list = r.get("data", {}).get("gyro", [])
            if gyro_list:
                return gyro_list[0]
        return None

# 动作控制（v109新增，自动区分v1/v2）
    # v2 Layers 动作前缀列表（用于自动区分 version）
    _V2_PREFIXES = ("H_", "F_", "Hd_")

    def _get_motion_version(self, motion_name):
        """根据动作名前缀自动判断 version：H_/F_/Hd_ → v2，其余 → v1"""
        if motion_name.startswith(self._V2_PREFIXES):
            return "v2"
        return "v1"

    def play_motion(self, motion_name, wait=True, scene="讲话"):
        """播放动作（自动区分v1 HTS / v2 Layers），立即返回（不定时等待）。
        动作会在后台播放，配合讲话同时进行，讲完再复位。
        scene: '讲话'|'唱歌'|'跳舞'，用于日志提示。
        version 字段放在顶层（非 motion 内部），符合 YanAPI 2.0 spec。"""
        cn = MOTION_CN_MAP.get(motion_name, motion_name)
        version = self._get_motion_version(motion_name)
        print(f"  [动作] 执行: {motion_name}（{cn}）version={version}")
        result = self.req("PUT", "/motions", {
            "operation": "start", "version": version,
            "motion": {"name": motion_name, "repeat": 1, "speed": "normal"}
        }, timeout=10)
        if result.get("code") == 0:
            scene_hint = {
                "讲话": "开始讲话（讲完再复位）",
                "唱歌": "开始唱歌（唱完再复位）",
                "跳舞": "开始跳舞（跳完再复位）",
            }.get(scene, f"开始{scene}（完再复位）")
            print(f"  [动作] ✓ {cn} 已触发（{version}），{scene_hint}")
            return True
        else:
            print(f"  [动作] ✗ {cn} 失败: {result}")
            return False

    def wait_motion_done(self, timeout=60, _bot=None, motion_name="", min_run_time=0,
                         sensor_verify_stand=False, sensor_verify_retries=3,
                         check_fall_event=True):
        """自适应退避轮询等待动作执行完毕。
        算法：指数退避 + 状态缓存 + 提前退出 + 最小运行时间保护 + 传感器姿态确认。
        比固定0.5s轮询减少40-60%请求次数。

        v122 优化：
        - min_run_time 自适应：若传入0，从 MOTION_TIME_TABLE 自动查表
        - 传感器姿态确认：API idle 后可选地用传感器二次验证（防止 API 假阴性）
        - sensor_verify_stand: True=API idle后用传感器确认站起姿态（爬起动作用）
        - sensor_verify_retries: 传感器验证重试次数（每秒1次）
        - check_fall_event: 是否在轮询中检查 fall_event。爬起动作期间应传 False，
          因为此时 fall_event 可能被 FallWatcher 反复 set（机器人还在地上），
          传 True 会导致立即返回 False，爬起动作永远等不完。

        返回 True=动作确实完成, False=超时或错误。
        _bot: ComfortBot实例（支持摔倒中断+传感器验证）。
        motion_name: 动作名称（用于日志+查表）。
        timeout: 最大等待秒数。
        min_run_time: 最小运行时间(秒)。0=自动从MOTION_TIME_TABLE查表，查不到用默认3s。
                     在前N秒内即使state返回idle也忽略，防止动作启动窗口内误判为已完成。
                     对爬起等慢启动动作特别重要——机器人从摔倒姿态到开始爬起有物理延迟。"""
# v122: min_run_time 自适应查表
        if min_run_time == 0 and motion_name:
            table_entry = MOTION_TIME_TABLE.get(motion_name, MOTION_TIME_DEFAULT)
            min_run_time = table_entry[1]  # 取最小运行保护时间
            if motion_name in MOTION_TIME_TABLE:
                print(f"  [动作] min_run_time 查表: {motion_name} → {min_run_time}s")
        elif min_run_time == 0:
            min_run_time = MOTION_TIME_DEFAULT[1]  # 默认3s

        start = time.time()
        poll_interval = 0.2  # 初始轮询间隔（快速检测动作启动）
        min_interval = 0.2  # 最小间隔
        max_interval = 1.5  # 最大间隔（稳定后降低请求频率）
        stable_count = 0  # 连续"执行中"计数（用于判断稳定性）
        stable_threshold = 3  # 连续N次stable后才放大间隔
        last_state_playing = False  # 上一轮状态缓存

        while time.time() - start < timeout:
            elapsed = time.time() - start

            # 摔倒中断检查（v122: check_fall_event=False 时跳过，用于爬起动作）
            if check_fall_event and _bot and hasattr(_bot, 'fall_event') and _bot.fall_event.is_set():
                print(f"  [动作] ⚠️ 摔倒中断{motion_name}等待")
                return False

            try:
                r = self.req("GET", "/motions", timeout=2)
                raw_state = r.get("data", {}).get("state")
                state_str = str(raw_state).strip().lower() if raw_state is not None else ""

                # 完成判定：0 / "0" / "idle" / "" (空/缺失视为空闲)
                if state_str in ("0", "idle", ""):
# 最小运行时间保护：如果还没跑够min_run_time，不判定完成
                    # 原因：某些动作（特别是爬起）启动时API可能短暂返回idle，
                    # 或者机器人还在从上一个状态切换通道，此时idle是假阴性
                    if min_run_time > 0 and elapsed < min_run_time:
                        hint = f"（{motion_name}）" if motion_name else ""
                        print(f"  [动作] ⏳ 启动保护中{hint}（{elapsed:.1f}s/{min_run_time}s，state=idle但忽略）")
                        last_state_playing = False
                        stable_count = 0
                        poll_interval = min_interval
                        time.sleep(poll_interval)
                        continue

                    # ===== v122: 传感器姿态确认（爬起动作用）=====
                    # 注意：_bot 在爬起场景必须传入（用于读取传感器），
                    # 但 check_fall_event=False 已防止 fall_event 中断
                    if sensor_verify_stand and _bot:
                        stand_ok = self._sensor_verify_standing(_bot, retries=sensor_verify_retries)
                        if not stand_ok:
                            hint = f"（{motion_name}）" if motion_name else ""
                            print(f"  [动作] ⚠️ API idle 但传感器确认未站起{hint}，继续等待...")
                            # 传感器确认失败，可能是 API 过早返回 idle
                            # 重置状态，继续轮询
                            last_state_playing = False
                            stable_count = 0
                            poll_interval = min_interval
                            time.sleep(1.0)  # 等待1秒再查
                            continue
                        else:
                            hint = f"（{motion_name}）" if motion_name else ""
                            print(f"  [动作] ✓ 传感器确认已站起{hint}")

                    hint = f"（{motion_name}）" if motion_name else ""
                    print(f"  [动作] ✓ 动作执行完毕{hint}（耗时 {elapsed:.1f}s）")
                    return True

                # 执行中判定：1 / "1" / "playing" / 其他非空值
                is_playing = state_str in ("1", "playing") or (state_str and state_str not in ("0", "idle"))

                if is_playing:
                    if last_state_playing:
                        stable_count += 1
                        if stable_count >= stable_threshold:
                            # 连续稳定执行中 → 放大轮询间隔减少请求
                            poll_interval = min(poll_interval * 1.3, max_interval)
                    else:
                        # 刚进入playing状态或状态恢复
                        stable_count = 1
                        poll_interval = min_interval
                    last_state_playing = True
                else:
                    # 非预期状态（如网络抖动导致的临时异常值）
                    stable_count = 0
                    poll_interval = min_interval
                    last_state_playing = False

                time.sleep(poll_interval)

            except Exception as e:
                # 网络异常时不放大间隔，快速重试
                stable_count = 0
                poll_interval = min_interval
                time.sleep(min_interval)

        elapsed = time.time() - start
        hint = f"（{motion_name}）" if motion_name else ""
        print(f"  [动作] ⚠️ 等待动作超时{hint}（{elapsed:.1f}s/{timeout}s）")
        return False

    def _sensor_verify_standing(self, _bot, retries=3):
        """传感器验证机器人是否真正站起。
        通过连续读取传感器 euler-x 判断姿态是否在正常站立范围（75°~115°）。

        v122 新增：解决爬起动作 API 返回 idle 但机器人实际未站起的问题。
        原因：机器人固件在某些情况下会在动作完成前就上报 idle，
        特别是在从摔倒姿态爬起时，固件可能将"动作序列执行完"误报为 idle，
        但物理上机器人可能还在倾斜状态。

        参数:
            _bot: ComfortBot 实例（用于读取传感器）
            retries: 验证重试次数（每秒1次）
        返回:
            True: 传感器确认站起（euler-x 在 75°~115°）
            False: 传感器确认未站起或无法读取
        """
        STAND_EULER_MIN = 75.0
        STAND_EULER_MAX = 115.0
        STAND_CONFIRM_FRAMES = 2  # 连续2帧确认

        confirm_count = 0
        for attempt in range(retries):
            gyro_data = self.get_gyro()
            if gyro_data is None:
                print(f"  [传感器验证] 第{attempt + 1}次：无法读取传感器")
                time.sleep(1.0)
                continue

            euler_x = gyro_data.get("euler-x", 0)
            if STAND_EULER_MIN <= euler_x <= STAND_EULER_MAX:
                confirm_count += 1
                if confirm_count >= STAND_CONFIRM_FRAMES:
                    print(f"  [传感器验证] ✓ 站立确认（euler-x={euler_x:.1f}°，连续{confirm_count}帧）")
                    return True
            else:
                confirm_count = 0  # 不在站立范围，重置
                print(
                    f"  [传感器验证] 第{attempt + 1}次：euler-x={euler_x:.1f}°（不在站立范围{STAND_EULER_MIN}~{STAND_EULER_MAX}°）")

            time.sleep(1.0)

        # 所有重试耗尽，传感器未确认站立
        print(f"  [传感器验证] ✗ {retries}次验证均未确认站立")
        return False

    def wait_dance_done(self, _bot, timeout=DANCE_TIMEOUT):
        """舞蹈专用完成检测（v123新增，不影响已有 wait_motion_done）。
        
        核心算法：传感器检测为主 + API状态轮询为辅 + 摔倒中断支持。
        
        为什么舞蹈不能用 wait_motion_done：
          Yanshee API 在舞蹈期间一直返回 state=idle，导致 wait_motion_done
          在 min_run_time 保护期后立即判定完成（实际舞蹈远未结束）。
          
        算法流程：
          1. 传感器主检测：短窗口gyro方差/均值 + euler姿态 → 静止确认
             （舞蹈中 gyro 活跃，停止后 gyro 归零 → 检测从"动"到"静"的转换）
          2. API 辅助检测：同时检查 motion_name 是否清空（舞蹈结束的可靠信号）
          3. 启动保护：前5秒只收集数据不判定静止（舞蹈启动时短暂静止）
          4. 摔倒中断：fall_event 触发时立即返回 False
          
        返回 True=舞蹈完成, False=超时/摔倒中断。
        """
        gyro_buf = []          # 短窗口 gyro |x| 缓冲
        still_counter = 0      # 静止帧计数器
        start = time.time()
        MIN_DANCE_TIME = 5.0   # 启动保护：前5秒不判定静止
        api_check_counter = 0  # API检查计数器（每10次传感器读取检查1次API）
        api_idle_count = 0     # API连续idle计数
        API_CHECK_EVERY = 10   # 每10次传感器读取检查1次API
        API_IDLE_CONFIRM = 3   # API连续idle+motion_name空 确认次数
        
        print(f"  [跳舞] 传感器检测中，等待舞蹈停止（超时{timeout}s）...")
        
        while time.time() - start < timeout:
            elapsed = time.time() - start
            
            # 摔倒中断检测
            if _bot and hasattr(_bot, 'fall_event') and _bot.fall_event.is_set():
                print(f"  [跳舞] ⚠️ 摔倒中断")
                return False
            
            # ====== API 辅助检测（低频，每次传感器循环的第10次）======
            # 检查 motion_name 是否已清空——这是舞蹈结束的可靠信号
            api_check_counter += 1
            if api_check_counter >= API_CHECK_EVERY and elapsed > MIN_DANCE_TIME:
                api_check_counter = 0
                try:
                    r = self.req("GET", "/motions", timeout=2)
                    data = r.get("data", {})
                    motion_name = data.get("name", "")
                    status = data.get("status", data.get("state", "unknown"))
                    # motion_name 清空 = 机器人已无动作在执行，这是最可靠的结束信号
                    if motion_name == "" and (str(status).strip().lower() in ("0", "idle", "")):
                        api_idle_count += 1
                        if api_idle_count >= API_IDLE_CONFIRM:
                            print(f"  [跳舞] ✓ API确认舞蹈完成（motion_name清空，连续{api_idle_count}次，耗时{elapsed:.1f}s）")
                            return True
                    else:
                        api_idle_count = 0
                except Exception:
                    pass
            
# 传感器主检测
            # wait_dance_done 是 RobotClient 方法，self 就是 RobotClient
            # 直接调 self.get_gyro()（与 _sensor_verify_standing 一致）
            gyro_data = self.get_gyro()
            if gyro_data is None:
                time.sleep(DANCE_STOP_CHECK_INTERVAL)
                continue
            
            gyro_x = abs(gyro_data.get("gyro-x", 0))
            euler_x = gyro_data.get("euler-x", 92.0)
            
            # 短窗口缓冲
            gyro_buf.append(gyro_x)
            if len(gyro_buf) > DANCE_STOP_WIN_SIZE:
                gyro_buf.pop(0)
            
            # 启动保护期内：只收集数据，不判定静止
            if elapsed < MIN_DANCE_TIME:
                time.sleep(DANCE_STOP_CHECK_INTERVAL)
                continue
            
            # 需要至少3帧数据才判断
            if len(gyro_buf) < 3:
                time.sleep(DANCE_STOP_CHECK_INTERVAL)
                continue
            
            # 计算短窗口 gyro 方差和均值
            g_mean = sum(gyro_buf) / len(gyro_buf)
            g_var = sum((g - g_mean) ** 2 for g in gyro_buf) / len(gyro_buf)
            
            # 静止条件：gyro 方差低 + 均值低 + 姿态正常
            is_still = (
                g_var < DANCE_STOP_GYRO_VAR_MAX and
                g_mean < DANCE_STOP_GYRO_MEAN_MAX and
                DANCE_STOP_EULER_MIN <= euler_x <= DANCE_STOP_EULER_MAX
            )
            
            if is_still:
                still_counter += 1
                if still_counter >= DANCE_STOP_CONFIRM_FRAMES:
                    elapsed = time.time() - start
                    print(f"  [跳舞] ✓ 传感器检测到舞蹈停止（静止{still_counter}帧，耗时{elapsed:.1f}s）"
                          f" gyro_var={g_var:.1f} gyro_mean={g_mean:.1f} euler={euler_x:.1f}°")
                    return True
            else:
                if still_counter > 0:
                    still_counter = 0  # 静止被打破（舞蹈又动了），重置计数
            
            time.sleep(DANCE_STOP_CHECK_INTERVAL)
        
        # 超时兜底
        elapsed = time.time() - start
        print(f"  [跳舞] ⏰ 超时（{elapsed:.1f}s/{timeout}s），强制结束")
        return False

    def _wait_dance_api_done(self, _bot=None, timeout=DANCE_TIMEOUT):
        """舞蹈专用完成检测（v123新增）：纯 API 轮询，不使用传感器。
        
        与 wait_dance_done（传感器检测）不同，此方法只通过 API 状态轮询判断：
          - 每0.5秒查 /motions 接口
          - motion_name 清空 + status=idle → 连续2次确认完成
          - 支持摔倒中断（fall_event，通过 _bot 传入）
          
        适用场景：舞蹈动作（API在舞蹈期间不会一直返回idle，
        舞蹈结束后 motion_name 会清空）。
        
        返回 True=舞蹈完成, False=超时/摔倒中断。
        """
        start_time = time.time()
        last_state_check = 0
        idle_confirm_count = 0
        STATE_CHECK_INTERVAL = 0.5
        IDLE_CONFIRM_NEEDED = 2
        
        while time.time() - start_time < timeout:
            # 摔倒中断检测（fall_event 在 _bot 上）
            if _bot and hasattr(_bot, 'fall_event') and _bot.fall_event.is_set():
                print(f"  [跳舞API] ⚠️ 摔倒中断")
                return False
            
            # 每0.5秒检查一次动作状态
            if time.time() - last_state_check >= STATE_CHECK_INTERVAL:
                last_state_check = time.time()
                try:
                    r = self.req("GET", "/motions", timeout=5)
                    data = r.get("data", {})
                    status = data.get("status", data.get("state", "unknown"))
                    motion_name = data.get("name", "")
                    
                    # 判定完成：motion_name 为空 或 status 为 idle
                    # 判定完成：motion_name 为空 且 status 为 idle（v124修复：or→and，防止单条件误判）
                    if (motion_name == "" or motion_name is None) and (status == "idle" or status == 0 or status == "0"):
                        idle_confirm_count += 1
                        if idle_confirm_count >= IDLE_CONFIRM_NEEDED:
                            elapsed = time.time() - start_time
                            print(f"  [跳舞API] ✓ 舞蹈执行完毕（耗时 {elapsed:.1f}s）")
                            return True
                    else:
                        idle_confirm_count = 0
                except Exception:
                    pass
            
            time.sleep(STATE_CHECK_INTERVAL)
        
        # 超时
        elapsed = time.time() - start_time
        print(f"  [跳舞API] ⏰ 超时（{elapsed:.1f}s/{timeout}s）")
        return False

    def do_dance(self):
        """从动作池随机选一个动作表演（跳舞），等待舞蹈完成再返回。"""
        motion = random.choice(DANCE_MOTION_POOL)
        cn = MOTION_CN_MAP.get(motion, motion)
        print(f"  [跳舞] 随机选择: {motion}（{cn}）")
        started = self.play_motion(motion, wait=False, scene="跳舞")
        if started:
            # 等待舞蹈完成，兜底60秒
            self.wait_motion_done(timeout=60)
        print(f"  [跳舞] 舞蹈结束")
        return motion

    def reset_pose(self):
        """复位到站立姿势（v1 Reset，立即返回，不卡住）"""
        print(f"  [动作] 复位（Reset）")
        self.req("PUT", "/motions", {
            "operation": "start", "version": "v1",
            "motion": {"name": "Reset", "repeat": 1, "speed": "normal"}
        }, timeout=10)
        # 不等待，立即返回，不卡住

    STAND_UP_BY_LEVEL = {
        "high": ["Reset"],  # 标准复位——稳重有力
        "mid_high": ["Reset"],  # 标准复位——稳定有力
        "mid": ["Reset", "reset_without_head"],  # 标准复位/轻柔复位——平稳过渡
        "low": ["reset_without_head"],  # 轻柔复位——温和低调
        "positive": ["Reset"],  # 标准复位——阳光自然
    }

    def stand_up(self, emotion_level=None):
        """恢复站立（鞠躬后起立）。触发后立即返回，不卡住。"""
        # 1. 根据情绪强度选起立动作
        if emotion_level and emotion_level in self.STAND_UP_BY_LEVEL:
            pool = self.STAND_UP_BY_LEVEL[emotion_level]
            motion = random.choice(pool)
        else:
            motion = "Reset"
        cn = MOTION_CN_MAP.get(motion, motion)
        version = self._get_motion_version(motion)
        print(f"  [动作] 起立（{motion}，{cn}）version={version}")
        self.req("PUT", "/motions", {
            "operation": "start", "version": version,
            "motion": {"name": motion, "repeat": 1, "speed": "normal"}
        }, timeout=10)
        # 不等待，立即返回，不卡住

        # 2. 起立后追加calibration校准姿势，不等待
        print(f"  [动作] 校准（calibration）")
        self.req("PUT", "/motions", {
            "operation": "start", "version": "v1",
            "motion": {"name": "calibration", "repeat": 1, "speed": "normal"}
        }, timeout=10)
        # 不等待，立即返回

    # 鞠躬舵机角度：低头(65) + 双肩前倾(105)
    BOW_ANGLES = {
        # 下半身
        "LeftHipFB": 106, "LeftHipLR": 90, "LeftKneeFlex": 53,
        "LeftAnkleFB": 90, "LeftAnkleUD": 90,
        "RightHipFB": 74, "RightHipLR": 90, "RightKneeFlex": 127,
        "RightAnkleFB": 90, "RightAnkleUD": 90,
        # 上半身
        "LeftShoulderFlex": 0, "LeftShoulderRoll": 90, "LeftElbowFlex": 90,
        "RightShoulderFlex": 170, "RightShoulderRoll": 0, "RightElbowFlex": 170,
        "NeckLR": 90, "NeckUD": 90,
    }

    def bow(self):
        """鞠躬：纯舵机角度方式（runtime=500ms执行 + 保持姿态0.7s）"""
        print(f"  [动作] 鞠躬（舵机角度）")
        self.req("PUT", "/servos/angles", {
            "angles": self.BOW_ANGLES, "runtime": 500
        }, timeout=5)
        time.sleep(1.2)  # 500ms执行 + 700ms保持（原2.5s过长）
        self.reset_pose()

