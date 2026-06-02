#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComfortBot：情绪陪伴主逻辑，5级强度循环、摔倒处理、断点恢复"""
import os, time, threading, random,requests,json,math
from config import (
    # 基础
    ROBOT_IP, ROBOT_PORT, BASE_URL,
    # 监听
    LISTEN_TIMEOUT, SILENCE_GAP, EXTRA,
    # 情绪
    ALL_EMOTIONS, POSITIVE_EMOTIONS, EMOTION_INTENSITY, EMOTION_LEVELS,
    EMOTION_MUSIC_KEYWORDS,
    # 动作/舞蹈
    DANCE_MOTION_POOL, COMFORT_MOTION_OPTIONS, MUSIC_POSE_OPTIONS,
    ANNOUNCE_MOTION_OPTIONS,
    DANCE_TIMEOUT, DANCE_STOP_TIMEOUT,
    MOTION_CN_MAP, MOTION_TIME_TABLE, MOTION_TIME_DEFAULT,
    # 爬起
    RECOVERY_FRONT, RECOVERY_REAR, RECOVERY_MAX_ATTEMPTS,
    FRONT_GETUP_ASSIST_ANGLES, FRONT_GETUP_ASSIST_RUNTIME, FRONT_GETUP_ASSIST_DELAY,
    # 摔倒检测 - 快速通道
    FALL_ACCEL_Y_FAST, FALL_EULER_DEVIATION_FAST, FALL_ACCEL_TOTAL_MIN,
    FALL_SEVERE_EULER_DEVIATION, FALL_SEVERE_ACCEL_Y, FALL_IMPACT_DELTA,
    FALL_CONFIRM_FRAMES, FALL_CHECK_INTERVAL,
    # 摔倒检测 - 舞蹈窗口
    FALL_DANCE_WIN_SIZE, FALL_DANCE_VAR_TH, FALL_DANCE_GYRO_MEAN_MAX,
    # 互补滤波
    CF_GYRO_WEIGHT, CF_ACCEL_LPF, CF_STAND_FRAMES, CF_STAND_MIN, CF_STAND_MAX,
    # 姿态判断
    FRONT_FALL_EULER_MAX, REAR_FALL_EULER_MIN, REAR_FALL_EULER_POSITIVE,
    STAND_CONFIRM_FRAMES, STAND_ACCEL_Y_MIN, STAND_EULER_X_MIN, STAND_EULER_X_MAX,
    # 初始模式
    INIT_MODE_ENABLED, INIT_STAND_ACCEL_Y, INIT_STAND_EULER_MIN, INIT_STAND_EULER_MAX,
    INIT_STAND_FRAMES, INIT_ACCEL_Y_THRESHOLD, INIT_ACCEL_X_THRESHOLD,
    INIT_ACCEL_TOTAL_MIN, INIT_EULER_DEVIATION, INIT_CONFIRM_FRAMES,
    # 舞蹈停止检测
    DANCE_STOP_WIN_SIZE, DANCE_STOP_GYRO_VAR_MAX, DANCE_STOP_GYRO_MEAN_MAX,
    DANCE_STOP_EULER_MIN, DANCE_STOP_EULER_MAX,
    DANCE_STOP_CONFIRM_FRAMES, DANCE_STOP_CHECK_INTERVAL,
)
from config import get_emotion_level
from config import CBT_MAX_ROUNDS, CBT_MIN_CHALLENGE_ROUNDS, CBT_MAX_CHALLENGE_ROUNDS, CBT_LOOSENING_SIGNALS, CBT_SUITABLE_EMOTIONS, CBT_MOTION_OPTIONS
from robot import Robot
from core.deepseek import _call_deepseek
from core.emotion import _detect_emotion_keyword, _is_stop, detect_emotion, detect_intent, check_if_user_wants_to_stop, _detect_emotion_keyword_with_confidence
from core.face_recognition import init_face_recognition, get_face_manager, FaceRecognitionManager
from core.text_gen import (
    generate_comfort_with_motion, generate_comfort_response, generate_comfort_with_address,
    generate_timeout_response, generate_greeting, generate_goodbye,
    generate_retry_ask, generate_response_question,
    generate_music_intro, generate_dance_intro, generate_positive_show_intro,
    generate_cycle_closing, generate_post_show_greeting,
    generate_fall_exclamation, generate_repeat_comfort, generate_repeat_greeting,
    generate_standup_recovery, generate_pushup_intro,
    deepseek_select_comfort_motion, deepseek_select_greeting_motion,
    recommend_music,
    # ★ 新增：人脸识别版话术
    generate_checkin_with_address, generate_face_comfort_intro,
)
from music import download, truncate_audio, get_duration
from dialogue import _AGENT_MEMORY, dialogue_observe_user_text, generate_checkin_question
from core.crisis_detection import CrisisDetector, get_crisis_response
from core.cbt_engine import CBTEngine, CBTPhase, should_use_cbt, start_cbt_session, get_cbt_response, end_cbt_session, is_cbt_active, get_cbt_session
from core.cbt_text_gen import generate_cbt_intro, generate_cbt_comfort_with_motion, generate_cbt_closing_with_summary

class ComfortBot:
    def __init__(self):
        self.robot = Robot()
        self.played_songs = []
        self.previous_responses = []
        self.previous_closings = []  # 每轮cycle结尾的过渡鼓励语记录
        self.previous_checkins = []  # 每轮结束后的递进式询问话术记录
        self.round_num = 0
        self.is_first_greeting = True
        # 摔倒检测
        self.fall_event = threading.Event()  # 摔倒信号：set=摔倒，clear=已恢复
        self._handling_fall = False  # 互斥锁：防止摔倒恢复过程中再次触发摔倒处理（递归）
        self.fall_watcher = None  # FallWatcher 线程
        # 摔倒前活动状态记忆（用于恢复）
        self.pre_fall_activity = None  # {"type":"dance"/"music", "motion":"xxx", "song_kw":"xxx", "song_path":"xxx", "song_dur":0, "emotion":"xxx"}
        self.current_emotion = None  # 当前情绪（摔倒时用同音色说话）
        self.last_dance_motion = None  # 上一次跳的舞（用于连续跳舞时避免重复）
        self.last_perform_type = None  # 上一次表演类型："music"或"dance"（避免连续同类型）
        self.last_song_kw = None  # 上一次播放的歌曲关键词（避免连续播放同一首）
        self.consecutive_perform_count = 0  # v125: 连续表演请求计数（3次后允许检测结束词退出）
        # ★ 人脸识别
        self.face_manager = None  # 人脸识别管理器
        self._use_face_recognition = False  # 是否启用人脸识别
        self._current_address = "朋友"  # 当前称呼
        self._current_voice = "x4_yezi"  # 当前音色
        self._pending_face_result = None  # ★ wait_speech 期间检测到的人脸情绪结果
        self._face_jump_cooldown_until = 0  # ★ 人脸情绪跳转冷却时间（防止短时间内重复触发）
        self._face_jump_just_triggered = False  # ★ 标记刚触发过人脸情绪跳转（跳过cycle内的再次人脸检测）
        self._face_detection_paused = False  # ★ 标记人脸检测已暂停（情绪跳转后）
        self._face_detection_stop_event = threading.Event()  # ★ 用于立即停止人脸检测线程的事件
        
        # ★ 危机预警系统
        self.crisis_detector = CrisisDetector()
        self._crisis_mode = False  # 是否处于危机模式
        self._crisis_level = None  # 当前危机等级
        
        # ★ CBT认知行为疗法
        self._cbt_engine = CBTEngine()  # CBT引擎实例
        self._cbt_active = False  # 是否处于CBT模式
        self._cbt_round_count = 0  # CBT对话轮次计数

    # ===== ASR监听 =====
    def read_asr(self):
        try:
            r = requests.get(f"{self.robot.base}/voice/asr", timeout=1)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == 0:
                    raw = data.get("data", "").rstrip("\x00")
                    if raw:
                        return json.loads(raw).get("intent", {}).get("text", "").strip()
        except Exception:
            pass
        return None

    def wait_speech(self, timeout=LISTEN_TIMEOUT, gap=SILENCE_GAP):
        """等待用户说话。

        返回值：
        - str: 用户说的话
        - None: 超时无人说话
        - tuple: ("_face_emotion_", emotion, face_result) 检测到人脸情绪，立即跳转
        """
        # ★ 清除停止事件，准备启动新的检测线程
        self._face_detection_stop_event.clear()
        
        # ★ 每次询问开始时恢复人脸检测（全新的识别）
        if self._face_detection_paused:
            self._face_detection_paused = False
            print("  [人脸检测] ★ 恢复检测（全新识别周期）")
        
        # ★★★ 关键修复：恢复检测时，清除上一次的所有人脸识别结果 ★★★
        # 只保留 _current_address（称呼），其他全部重置为全新状态
        self._pending_face_result = None  # 清除残留的人脸结果
        force_new_detection = False  # ★ 标记是否强制全新检测
        if self.face_manager:
            self.face_manager.clear_cache()  # 清除缓存，确保全新识别
            force_new_detection = True  # ★ 清除缓存后，下次检测必须全新
        self._face_jump_cooldown_until = 0  # 清除冷却时间

        for _ in range(3):
            try:
                requests.get(f"{self.robot.base}/voice/asr", timeout=0.5)
            except Exception:
                pass
            time.sleep(0.05)
        try:
            self.robot.req("DELETE", "/voice/asr");
            time.sleep(0.05)
            self.robot.req("PUT", "/voice/asr", {"continues": True});
            time.sleep(0.15)
        except Exception:
            pass
        deadline = time.time() + timeout
        texts = []
        # ★ 人脸情绪检测结果
        face_result = [None]

        # ★ 启动并行人脸情绪检测（等待用户说话期间）
        # ★ 捕获 force_new_detection 的值，确保每次检测都是全新的
        _force_new = force_new_detection
        
        def _face_detect_loop():
            """持续检测人脸情绪，检测到情绪后存入 face_result[0]"""
            start = time.time()
            while time.time() - start < timeout:
                # ★ 检查停止事件（立即退出）
                if self._face_detection_stop_event.is_set():
                    return
                # ★ 检查暂停标志
                if self._face_detection_paused:
                    return
                if self.fall_event.is_set():
                    break
                try:
                    # ★★★ 关键：强制全新检测（忽略任何缓存）★★★
                    fr = self.face_manager.detect(force_new=_force_new)
                    if fr and fr.get("success"):
                        emotion = fr.get("emotion")
                        if emotion and emotion in ALL_EMOTIONS:
                            # ★ 冷却检查：避免短时间内重复触发人脸情绪跳转
                            if time.time() < self._face_jump_cooldown_until:
                                continue  # 在冷却期内，跳过本次检测，继续等待
                            # ★ 补充生成称呼和音色
                            gender = fr.get("gender")
                            age = fr.get("age")
                            fr["address"] = FaceRecognitionManager._generate_address(gender, age)
                            fr["voice"] = FaceRecognitionManager._select_voice(gender, age)
                            face_result[0] = fr
                            print(f"  [人脸情绪-检测] 情绪: {emotion}，称呼: {fr['address']}，年龄: {age}，性别: {gender}")
                            return  # 检测到情绪后立即退出线程
                except Exception:
                    pass
                time.sleep(0.3)  # 每0.3秒检测一次

        if self._use_face_recognition and self.face_manager:
            face_thread = threading.Thread(target=_face_detect_loop, daemon=True)
            face_thread.start()
        else:
            face_thread = None
        while time.time() < deadline:
            # ★ 优先检查人脸情绪（检测到情绪 → 立即返回，不等待用户说话）
            if face_result[0] is not None:
                fr = face_result[0]
                print(f"  [人脸情绪-跳转] 立即触发情绪跳转: {fr.get('emotion')}，称呼: {fr.get('address')}")
                return ("_face_emotion_", fr.get("emotion"), fr)  # 返回元组让调用方识别
            # 摔倒检测：等待说话时也要响应摔倒，处理完后重新启动监听
            if self.fall_event.is_set():
                self._handle_fall()
                # 重新启动 ASR 监听（清空缓冲、重新开始）
                try:
                    self.robot.req("DELETE", "/voice/asr")
                    time.sleep(0.05)
                    self.robot.req("PUT", "/voice/asr", {"continues": True})
                    time.sleep(0.15)
                except Exception:
                    pass
                texts = []
                deadline = time.time() + timeout  # 重置超时
                continue
            if self.robot.is_tts or time.time() - self.robot.last_tts < 0.15:
                time.sleep(0.05);
                continue
            t = self.read_asr()
            if t and t not in texts:
                texts.append(t)
                stable = time.time()
                fell_during_stable = False
                while time.time() - stable < gap and time.time() < deadline:
                    if self.fall_event.is_set():
                        self._handle_fall()
                        # 重新启动 ASR 监听
                        try:
                            self.robot.req("DELETE", "/voice/asr")
                            time.sleep(0.05)
                            self.robot.req("PUT", "/voice/asr", {"continues": True})
                            time.sleep(0.15)
                        except Exception:
                            pass
                        fell_during_stable = True
                        break  # 摔倒处理完，跳出内层稳定检测
                    if self.robot.is_tts: time.sleep(0.05); continue
                    t2 = self.read_asr()
                    if t2 and t2 not in texts: texts.append(t2); stable = time.time()
                    time.sleep(0.05)
                if fell_during_stable:
                    texts = []
                    deadline = time.time() + timeout  # 重置超时
                    continue
                return " ".join(texts)
            time.sleep(0.05)
        return None

    # ===== 动作融合：先做动作→保持→讲话→智能等待→复位（均支持摔倒中断与恢复）=====

    @staticmethod
    def _motion_tail_wait(text, base=0.2):
        """动态计算动作收尾等待时间（基于文本长度的自适应算法）v2。
        原理：speak()返回时TTS已播完（含dur+0.1s缓冲），但动作可能还在执行。
        大部分v2 Layers动作在 speak 完成时已接近同步完成（因为动作先触发、TTS有合成延迟），
        所以只需极短缓冲让舵机到位即可。长文本意味着TTS播放久、动作已提前结束很久。

        v2优化：
        - base 从 0.4→0.2（speak()已含0.1s缓冲，动作基本同步）
        - ≤12字 → base（短句TTS快，动作几乎同步）
        - 13-25字 → base+0.2s
        - 26-40字 → base+0.3s
        - >40字 → 封顶0.8s（原1.2s过长）
        平均每次节省 0.2-0.4s，一个完整cycle(~10次调用)省 2-4s。"""
        text_len = len(text)
        if text_len <= 12:
            return base
        elif text_len <= 25:
            return base + 0.2
        elif text_len <= 40:
            return base + 0.3
        else:
            return min(base + 0.5, 0.8)

    def _comfort_with_motion(self, text, emotion, pre_selected_motion=None):
        """安慰：先做动作→保持→讲话→智能等待→复位

        pre_selected_motion: (motion_id, motion_desc) 元组，由 generate_comfort_with_motion()
            预先返回。传入时可跳过 deepseek_select_comfort_motion() 的额外API调用，
            节省 1-5s 衔接时间。
        """
        if pre_selected_motion:
            motion_id, motion_desc = pre_selected_motion
            print(f"  [安慰动作] 预选（合并API）: {motion_id}（{motion_desc}）")
        else:
            motion_id, motion_desc = deepseek_select_comfort_motion(emotion)
            print(f"  [安慰动作] DeepSeek选择: {motion_id}（{motion_desc}）")
        cn = MOTION_CN_MAP.get(motion_id, motion_id)
        self.pre_fall_activity = {"type": "comfort", "text": text, "emotion": emotion, "motion_id": motion_id}
        self.robot.play_motion(motion_id, scene="讲话")
        self.robot.speak(text, emotion, voice=self._current_voice, _bot=self)
        if self.fall_event.is_set():
            return  # speak内部已处理摔倒
        wait_sec = self._motion_tail_wait(text, base=0.2)
        self._interruptible_sleep(wait_sec)
        self.robot.reset_pose()
        self.pre_fall_activity = None

    def _greet_with_motion(self, text, emotion=None):
        """问候：先做动作→保持→讲话→智能等待→复位"""
        motion_id, motion_desc = deepseek_select_greeting_motion(
            emotion=emotion, is_first=self.is_first_greeting)
        cn = MOTION_CN_MAP.get(motion_id, motion_id)
        print(f"  [问候动作] {motion_id}（{motion_desc}）")
        self.pre_fall_activity = {"type": "greet", "text": text, "emotion": emotion, "motion_id": motion_id}
        self.robot.play_motion(motion_id, scene="讲话")
        self.robot.speak(text, emotion or "", speed=55, voice=self._current_voice, _bot=self)
        if self.fall_event.is_set():
            return
        wait_sec = self._motion_tail_wait(text, base=0.2)
        self._interruptible_sleep(wait_sec)
        self.robot.reset_pose()
        self.is_first_greeting = False
        self.pre_fall_activity = None

    def _announce_with_motion(self, text, emotion):
        """告知语：用告知动作池的动作，先做动作→保持→讲话→智能等待→复位"""
        motion_id = random.choice(list(ANNOUNCE_MOTION_OPTIONS.keys()))
        motion_desc = ANNOUNCE_MOTION_OPTIONS[motion_id]
        cn = MOTION_CN_MAP.get(motion_id, motion_id)
        print(f"  [告知动作] 随机选择: {motion_id}（{motion_desc}）")
        self.pre_fall_activity = {"type": "announce", "text": text, "emotion": emotion, "motion_id": motion_id}
        self.robot.play_motion(motion_id, scene="讲话")
        self.robot.speak(text, emotion, speed=55, voice=self._current_voice, _bot=self)
        if self.fall_event.is_set():
            return
        wait_sec = self._motion_tail_wait(text, base=0.2)
        self._interruptible_sleep(wait_sec)
        self.robot.reset_pose()
        self.pre_fall_activity = None

    def _announce_no_reset(self, text, emotion):
        """告知语（不复位版）：说完过渡语后不复位，直接过渡到下一个pose。"""
        motion_id = random.choice(list(ANNOUNCE_MOTION_OPTIONS.keys()))
        motion_desc = ANNOUNCE_MOTION_OPTIONS[motion_id]
        cn = MOTION_CN_MAP.get(motion_id, motion_id)
        print(f"  [告知动作] 随机选择: {motion_id}（{motion_desc}，不复位）")
        self.pre_fall_activity = {"type": "announce", "text": text, "emotion": emotion, "motion_id": motion_id}
        self.robot.play_motion(motion_id, scene="讲话")
        self.robot.speak(text, emotion, speed=55, voice=self._current_voice, _bot=self)
        if self.fall_event.is_set():
            return
        self._interruptible_sleep(0.1)  # 不复位版只需极短缓冲（直接过渡到下一个pose）
        self.pre_fall_activity = None

    def _post_show_bow_greet(self, emotion):
        """表演后：鞠躬→起立→问候问新心情（不卡住）"""
        self.robot.bow()
        # 不等待，立即继续
        self.robot.stand_up(get_emotion_level(emotion))
        re_greet = generate_post_show_greeting(emotion)
        print(f"  [重新问候] {re_greet}")
        self._greet_with_motion(re_greet, emotion)

    def _goodbye_with_bow(self, text, emotion):
        """告别：鞠躬→讲话→再鞠躬→起立"""
        self.robot.bow()
        self.robot.speak(text, emotion, voice=self._current_voice, _bot=self)
        self.robot.bow()
        self.robot.stand_up(get_emotion_level(emotion))

    # ==================== 危机预警处理 ====================
    def _handle_crisis(self, emotion, crisis_result):
        """处理危机预警
        
        Args:
            emotion: 当前情绪
            crisis_result: 危机检测结果字典
        """
        level = crisis_result['level']
        risk_score = crisis_result['risk_score']
        
        print(f"\n  ⚠️ [危机预警] 等级: {level}, 风险分数: {risk_score}")
        print(f"  [危机预警] 成分: {crisis_result['components']}")
        
        # 获取危机响应话术和动作
        response = get_crisis_response(level)
        phrases = response['phrases']
        motions = response['motions']
        
        # 根据危机等级执行不同策略
        if level == 'critical':
            # 紧急情况：立即安抚
            phrase = random.choice(phrases)
            print(f"  [危机处理-紧急] {phrase}")
            
            # 使用紧急安抚动作
            motion = random.choice(motions)
            motion_cn = MOTION_CN_MAP.get(motion, motion)
            print(f"  [危机处理-动作] {motion}（{motion_cn}）")
            
            self._comfort_with_motion(phrase, emotion, pre_selected_motion=motion)
            
            # 持续关注：缩短询问超时时间
            self._crisis_mode = True
            self._crisis_level = 'critical'
            
            # 记录危机事件
            print(f"  [危机处理] 已记录危机事件，将持续关注")
            
        elif level == 'warning':
            # 警告情况：加强陪伴
            phrase = random.choice(phrases)
            print(f"  [危机处理-警告] {phrase}")
            
            motion = random.choice(motions)
            motion_cn = MOTION_CN_MAP.get(motion, motion)
            print(f"  [危机处理-动作] {motion}（{motion_cn}）")
            
            self._comfort_with_motion(phrase, emotion, pre_selected_motion=motion)
            self._crisis_mode = True
            self._crisis_level = 'warning'
            
        elif level == 'attention':
            # 需要关注：保持注意力
            phrase = random.choice(phrases)
            print(f"  [危机处理-关注] {phrase}")
            
            self._comfort_with_motion(phrase, emotion)
            self._crisis_mode = True
            self._crisis_level = 'attention'
        
        # 记录日志
        try:
            import json
            log_data = {
                'timestamp': time.time(),
                'level': level,
                'risk_score': risk_score,
                'emotion': emotion,
                'components': crisis_result['components']
            }
            log_path = "crisis_log.json"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as _e:
            print(f"  [危机日志] 写入失败: {_e}")
    
    
    # ===== 音乐播放（带保持动作）=====
    def _play_one_song(self, emotion, user_text=""):
        """推荐并播放一首歌。先上传音乐，再摆pose保持到播放结束，复位。"""
        kws = recommend_music(emotion, user_text, self.played_songs)
        if not kws:
            # 极端兜底（理论上不会到这里）
            kws = ["晴天 周杰伦", "夜空中最亮的星 逃跑计划", "平凡之路 朴树"]
            print(f"  [音乐] 极端兜底: {kws}")

        for kw in kws:
            print(f"  [音乐] 尝试: {kw}")
            # v123: 硬性排除已播放过的歌曲
            if kw == self.last_song_kw:
                print(f"  [音乐] ⏭ 跳过（与上次播放相同）: {kw}")
                continue
            path, dur, label = download(kw)
            if not path:
                print(f"  [音乐] 下载失败: {kw}")
                continue
            # v123: 硬性排除已播放过的歌曲（label级比对）
            if label and label in self.played_songs:
                print(f"  [音乐] ⏭ 跳过（已播放过）: {label}")
                continue

            # 1. 先上传音乐文件
            print(f"  [音乐] 上传: {os.path.basename(path)}")
            up_result = self.robot.upload(path)
            print(f"  [音乐] 上传结果: {up_result}")
            if isinstance(up_result, dict) and up_result.get("code") != 0:
                print(f"  [音乐] ⚠️ 上传可能失败: {up_result}")

            # 2. 摆pose + 上传后等待 并行：开始摆pose（不等待完成），同时等上传就绪
            music_pose = random.choice(list(MUSIC_POSE_OPTIONS.keys()))
            pose_desc = MUSIC_POSE_OPTIONS[music_pose]
            pose_cn = MOTION_CN_MAP.get(music_pose, music_pose)
            print(f"  [动作] 摆放音乐pose（{music_pose}·{pose_cn}）")
            self.robot.play_motion(music_pose, wait=False, scene="唱歌")
            # 上传后短暂等待确保文件就绪（pose动作已开始执行）
            time.sleep(0.2)

            # 3. 设置音量并播放
            self.robot.set_volume(82)
            fname = os.path.basename(path)
            play_ok = self.robot.play(fname)
            if not play_ok:
                time.sleep(0.3)  # 播放失败重试间隔（robot.play()内置0.3s，此处再等一轮）
                play_ok = self.robot.play(fname)
            if not play_ok:
                print(f"  [音乐] ✗ 播放失败: {fname}")
                self.robot.stop()
                self.robot.reset_pose()
                continue
            # 记录活动状态（摔倒恢复用），含播放开始时间
            # is_first_play: 首次播放标记，摔倒恢复时据此判断是否从头开始
            # orig_song_path/orig_song_dur: v123 保存原始文件信息，确保多次摔倒都能从原始文件截取
            self.pre_fall_activity = {"type": "music", "song_kw": kw, "song_path": path, "song_dur": dur,
                                      "song_name": label, "emotion": emotion, "play_start": time.time(),
                                      "is_first_play": True,
                                      "orig_song_path": path, "orig_song_dur": dur}

            # 4. 等待音乐播放完毕（保持pose中，可被摔倒中断）
            play_time = dur + EXTRA
            print(f"  [音乐] 播放 {fname}，{play_time:.1f}s（保持{pose_cn} pose中...）")
            # v122: 拆分等待，播放3秒后标记 is_first_play=False
            # 原因：is_first_play=True 导致摔倒恢复时"从头播放"而非断点续播
            first_wait = min(3.0, play_time)
            if not self._interruptible_sleep(first_wait):
                # 摔倒已处理，活动已恢复，直接返回
                self.robot.stop()
                self.pre_fall_activity = None
                return True
            # 已稳定播放3秒，后续摔倒可从断点续播
            if self.pre_fall_activity and self.pre_fall_activity.get("type") == "music":
                self.pre_fall_activity["is_first_play"] = False
            remaining_wait = play_time - first_wait
            if remaining_wait > 0 and not self._interruptible_sleep(remaining_wait):
                # 摔倒已处理，活动已恢复，直接返回
                self.robot.stop()
                self.pre_fall_activity = None
                return True
            self.robot.stop()
            self.pre_fall_activity = None  # 清除活动状态

            # 5. 音乐播完，复位 → 鞠躬
            print(f"  [音乐] 播放完毕，复位")
            self.robot.reset_pose()
            print(f"  [音乐] 表演结束，鞠躬")
            self.robot.bow()
            self.played_songs.append(label)
            self.last_song_kw = kw  # v123: 记录本次播放的歌曲关键词
            return True

        print("  [音乐] 无法播放，跳过")
        return False

    # ===== 跳舞表演 =====
    def _do_dance(self, emotion):
        """问要不要看跳舞→跳舞（传感器检测停止）→鞠躬→起立→问候"""
        intro = generate_dance_intro(emotion)
        print(f"  [跳舞告知] {intro}")
        # 用告知动作说跳舞过渡语（区别于安慰动作）
        self._announce_with_motion(intro, emotion)
        # 跳舞并用传感器检测停止
        dance_motion = random.choice(DANCE_MOTION_POOL)
        dance_cn = MOTION_CN_MAP.get(dance_motion, dance_motion)
        print(f"  [跳舞] 选择: {dance_motion}（{dance_cn}）")
        self.pre_fall_activity = {"type": "dance", "motion": dance_motion, "emotion": emotion}
        self.robot.play_motion(dance_motion, wait=False, scene="跳舞")
        self._wait_dance_stop(timeout=DANCE_STOP_TIMEOUT)
        # 舞蹈结束后，鞠躬→起立→问候
        self.pre_fall_activity = None
        self.robot.bow()
        self.robot.stand_up(get_emotion_level(emotion))

    # ===== 带时间限制的表演工具 =====
    def _play_full_music(self, emotion, user_text):
        """放音乐，按完整时长+EXTRA秒播放。安慰结束后直接说过渡语→摆pose→播放，跳过告知动作缩短间隔。"""
        kws = recommend_music(emotion, user_text, self.played_songs)
        if not kws:
            kws = ["晴天", "夜空中最亮的星", "平凡之路"]
        for kw in kws:
            print(f"  [音乐] 尝试（完整播放）: {kw}")
            # v123: 硬性排除已播放过的歌曲（不依赖AI"不要推荐"的软限制）
            if kw == self.last_song_kw:
                print(f"  [音乐] ⏭ 跳过（与上次播放相同）: {kw}")
                continue
            path, dur, label = download(kw)
            if not path:
                print(f"  [音乐] 下载失败: {kw}")
                continue
            # v123: 硬性排除已播放过的歌曲（label级比对）
            if label and label in self.played_songs:
                print(f"  [音乐] ⏭ 跳过（已播放过）: {label}")
                continue
            # 告知（含歌名），有告知动作但跳过复位直接过渡到音乐pose
            song_for_intro = label if label else kw
            intro = generate_music_intro(emotion, song_name=song_for_intro)
            print(f"  [音乐告知] {intro}")
            self._announce_no_reset(intro, emotion)
            # 1. 上传
            print(f"  [音乐] 上传: {os.path.basename(path)}")
            self.robot.upload(path)
            # 2. 摆pose（不等待完成，与上传就绪等待并行）
            music_pose = random.choice(list(MUSIC_POSE_OPTIONS.keys()))
            pose_cn = MOTION_CN_MAP.get(music_pose, music_pose)
            print(f"  [动作] 摆放音乐pose（{music_pose}·{pose_cn}）")
            self.robot.play_motion(music_pose, wait=False, scene="唱歌")
            time.sleep(0.15)  # 摆pose后极短缓冲（upload已返回成功）
            # 3. 播放
            self.robot.set_volume(82)
            fname = os.path.basename(path)
            play_ok = self.robot.play(fname)
            if not play_ok:
                time.sleep(0.3)
                play_ok = self.robot.play(fname)
            if not play_ok:
                print(f"  [音乐] ✗ 播放失败: {fname}")
                self.robot.reset_pose()
                continue
            # 记录活动状态（摔倒恢复用），含播放开始时间
            # is_first_play: 首次播放标记，摔倒恢复时据此判断是否从头开始
            # orig_song_path/orig_song_dur: v123 保存原始文件信息，确保多次摔倒都能从原始文件截取
            self.pre_fall_activity = {"type": "music", "song_kw": kw, "song_path": path, "song_dur": dur,
                                      "song_name": label, "emotion": emotion, "play_start": time.time(),
                                      "is_first_play": True,
                                      "orig_song_path": path, "orig_song_dur": dur}
            # 4. 按完整时长等待（可被摔倒中断）
            actual_wait = dur + EXTRA
            print(f"  [音乐] 播放中，等待{actual_wait:.1f}s（完整时长）")
            # v122: 拆分等待，播放3秒后标记 is_first_play=False
            first_wait = min(3.0, actual_wait)
            if not self._interruptible_sleep(first_wait):
                self.robot.stop()
                self.pre_fall_activity = None
                return True
            if self.pre_fall_activity and self.pre_fall_activity.get("type") == "music":
                self.pre_fall_activity["is_first_play"] = False
            remaining_wait = actual_wait - first_wait
            if remaining_wait > 0 and not self._interruptible_sleep(remaining_wait):
                self.robot.stop()
                self.pre_fall_activity = None
                return True
            self.robot.stop()
            self.pre_fall_activity = None  # 清除活动状态
            # 5. 复位 → 鞠躬
            print(f"  [音乐] 播放完毕，复位")
            self.robot.reset_pose()
            print(f"  [音乐] 表演结束，鞠躬")
            self.robot.bow()
            self.played_songs.append(label)
            self.last_song_kw = kw  # v123: 记录本次播放的歌曲关键词
            return True
        print("  [音乐] 无法播放，跳过")
        return False

    def _play_user_song(self, emotion, song_name):
        """播放用户指定的歌曲
        v125: 新增功能 - 用户说"唱首XXX"时调用
        """
        if not song_name:
            print("  [用户歌曲] 未提供歌曲名，改用随机音乐")
            self._play_full_music(emotion, "")
            return
        
        print(f"  [用户歌曲] 搜索并播放: {song_name}")
        # 使用用户提供的歌曲名搜索下载
        path, dur, label = download(song_name)
        if not path:
            print(f"  [用户歌曲] 下载失败: {song_name}，改用推荐音乐")
            self._play_full_music(emotion, f"{song_name} 找不到")
            return
        
        # 告知歌曲信息
        actual_name = label if label else song_name
        intro = f"好的，给你放首《{actual_name}》，希望你喜欢～"
        print(f"  [音乐告知] {intro}")
        self._announce_no_reset(intro, emotion)

        # 上传
        print(f"  [音乐] 上传: {os.path.basename(path)}")
        up_result = self.robot.upload(path)
        if isinstance(up_result, dict) and up_result.get("code") != 0:
            print(f"  [音乐] ⚠️ 上传可能失败: {up_result}")

        # 摆pose
        music_pose = random.choice(list(MUSIC_POSE_OPTIONS.keys()))
        pose_cn = MOTION_CN_MAP.get(music_pose, music_pose)
        print(f"  [动作] 摆放音乐pose（{music_pose}·{pose_cn}）")
        self.robot.play_motion(music_pose, wait=False, scene="唱歌")
        time.sleep(0.15)

        # 播放音乐
        self.robot.set_volume(82)
        fname = os.path.basename(path)
        play_ok = self.robot.play(fname)
        if not play_ok:
            time.sleep(0.3)
            play_ok = self.robot.play(fname)
        if not play_ok:
            print(f"  [音乐] ✗ 播放失败: {fname}")
            self.robot.stop()
            self.robot.reset_pose()
            return

        # 记录活动状态（摔倒恢复用）
        self.pre_fall_activity = {"type": "music", "song_kw": song_name, "song_path": path, "song_dur": dur,
                                  "song_name": actual_name, "emotion": emotion, "play_start": time.time(),
                                  "is_first_play": True,
                                  "orig_song_path": path, "orig_song_dur": dur}

        MUSIC_EXTRA = 2.0
        # 播放并等待（拆分等待，3秒后标记 is_first_play=False 以支持断点续播）
        total_wait = dur + MUSIC_EXTRA
        print(f"  [音乐] 播放中，等待{total_wait:.1f}s（完整时长）")
        first_wait = min(3.0, total_wait)
        if not self._interruptible_sleep(first_wait):
            # 摔倒已处理，活动已恢复，直接返回
            self.robot.stop()
            self.pre_fall_activity = None
            return
        # 已稳定播放3秒，后续摔倒可从断点续播
        if self.pre_fall_activity and self.pre_fall_activity.get("type") == "music":
            self.pre_fall_activity["is_first_play"] = False
        remaining_wait = total_wait - first_wait
        while remaining_wait > 0:
            if self.fall_event.is_set():
                self._handle_fall()
                # 摔倒恢复已完成（_resume_activity），直接退出
                self.robot.stop()
                self.pre_fall_activity = None
                return
            if remaining_wait > 0 and not self._interruptible_sleep(min(remaining_wait, 1.0)):
                # 被中断（摔倒已处理）
                self.robot.stop()
                self.pre_fall_activity = None
                return
            remaining_wait -= 1.0

        self.robot.stop()
        self.pre_fall_activity = None
        
        # 结束
        self.robot.stop()
        # 复位 → 鞠躬
        print(f"  [音乐] 播放完毕，复位")
        self.robot.reset_pose()
        print(f"  [音乐] 表演结束，鞠躬")
        self.robot.bow()
        
        # 记录播放历史
        if label:
            self.played_songs.append(label)
        self.last_song_kw = song_name

    def _do_dance_for_intent(self, emotion, intent=None, exclude_motion=None):
        """根据intent跳舞：intent=("dance",)随机选 / intent=("dance","WeAreTakingOff")指定舞"""
        specific = None
        if intent and len(intent) > 1 and intent[1]:
            specific = intent[1]
        excl = exclude_motion or self.last_dance_motion
        self._do_dance_full(emotion, exclude_motion=excl, specific_motion=specific)

    def _do_dance_full(self, emotion, exclude_motion=None, specific_motion=None):
        """跳舞，等待动作完成。
        参考 fall_detect_dance_sampler.py 的成功逻辑：
        1. 随机选择舞蹈（排除上一次跳的舞）
        2. 告知动作名称
        3. 使用 API 状态轮询等待动作完成
        4. 完成后等待恢复期（2秒）
        5. 停止动作
        6. 复位→鞠躬→起立
        exclude_motion: 需要排除的舞蹈（通常为上一次跳的舞）
        specific_motion: 用户指定的舞蹈motion名（如"WeAreTakingOff"）
        """
        # 指定舞蹈 or 随机选择（排除上一次跳的舞）
        if specific_motion and specific_motion in DANCE_MOTION_POOL:
            dance_motion = specific_motion
        elif exclude_motion and exclude_motion in DANCE_MOTION_POOL and len(DANCE_MOTION_POOL) > 1:
            candidates = [m for m in DANCE_MOTION_POOL if m != exclude_motion]
            dance_motion = random.choice(candidates)
        else:
            dance_motion = random.choice(DANCE_MOTION_POOL)
        dance_cn = MOTION_CN_MAP.get(dance_motion, dance_motion)
        self.last_dance_motion = dance_motion  # 记录本次舞蹈，供下次排除
        self.last_perform_type = "dance"  # v123: 记录表演类型
        print(f"  [跳舞] 选择: {dance_motion}（{dance_cn}）")

        # 告知（含舞蹈名）——_announce_with_motion 会临时覆盖 pre_fall_activity
        intro = generate_dance_intro(emotion, dance_name=dance_cn)
        print(f"  [跳舞告知] {intro}")
        self._announce_with_motion(intro, emotion)

        # 告知结束后重新设置跳舞的活动状态（防止被 _announce_with_motion 覆盖/清除）
        self.pre_fall_activity = {"type": "dance", "motion": dance_motion, "emotion": emotion}

        # ====== 1. 开始执行舞蹈动作 ======
        print(f"  [跳舞] 开始执行: {dance_motion}（{dance_cn}）...")
        result = self.robot.req("PUT", "/motions", {
            "operation": "start", "version": "v1",
            "motion": {"name": dance_motion, "repeat": 1, "speed": "normal"}
        }, timeout=10)
        if result.get("code") != 0:
            print(f"  [跳舞] ⚠️ 动作启动失败: {result}")
            self.pre_fall_activity = None
            return
        print(f"  [跳舞] ✓ 动作已启动，等待完成...")
        start_time_ref = time.time()

        # ====== 2. 等待动作完成（v123: 使用 _wait_dance_api_done 纯API轮询）======
        done = self.robot._wait_dance_api_done(_bot=self, timeout=DANCE_TIMEOUT)
        if not done:
            elapsed = time.time() - start_time_ref
            # v123修复：摔倒中断时立即return，不执行后续复位逻辑
            # FallWatcher 会负责：停止动作→爬起→恢复活动
            if self.fall_event.is_set():
                print(f"  [跳舞] ⚠️ 摔倒中断（{elapsed:.1f}s），同步处理摔倒")
                # 停止当前动作
                self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)
                # v112模式：同步处理摔倒（爬起+恢复），处理完后cycle方法继续下一步
                self._handle_fall()
                return
            else:
                print(f"  [跳舞] ⏰ 超时（{elapsed:.1f}s/{DANCE_TIMEOUT}s），强制停止")
                self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)
        else:
            elapsed = time.time() - start_time_ref
            print(f"  [跳舞] ✓ 舞蹈执行完毕（耗时 {elapsed:.1f}s）")

        # ====== 3. 动作完成后等待恢复期（关键！让机器人稳定）======
        print(f"  [跳舞] ⏳ 等待恢复期（2秒）...")
        time.sleep(2.0)

        # ====== 4. 确保停止所有动作 ======
        print(f"  [跳舞] 🛑 停止动作...")
        self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)
        time.sleep(0.5)  # 停止后短暂等待确保生效

        # ====== 5. 复位→鞠躬→起立 ======
        print(f"  [跳舞] 复位→鞠躬→起立...")
        self.robot.reset_pose()
        time.sleep(0.3)  # 复位后短暂等待
        self.robot.bow()
        time.sleep(0.3)  # 鞠躬后短暂等待
        self.robot.stand_up(get_emotion_level(emotion))

        # ====== 6. 清理状态 ======
        self.pre_fall_activity = None  # 清除活动状态
        print(f"  [跳舞] ✓ 跳舞完成")

    def _wait_dance_stop(self, timeout=DANCE_STOP_TIMEOUT):
        """用传感器数据检测舞蹈动作是否执行完毕。
        舞蹈中：gyro 方差高（>80）、均值高（>8）
        舞蹈停止：gyro 方差低（<50）、均值低（<3）、euler 正常（70~130°）
        连续 DANCE_STOP_CONFIRM_FRAMES 帧满足静止条件 → 确认舞蹈结束。
        同时支持摔倒中断。
        """
        gyro_buf = []  # 短窗口 gyro |x| 缓冲
        still_counter = 0  # 静止帧计数器
        start = time.time()
        print(f"  [跳舞] 传感器检测中，等待舞蹈停止（兜底{timeout}s）...")

        while time.time() - start < timeout:
            # 摔倒中断检测
            if self.fall_event.is_set() and not self._handling_fall:
                print(f"  [跳舞] ⚠️ 摔倒中断")
                return

            # 读取传感器
            gyro_data = self.robot.get_gyro()
            if gyro_data is None:
                time.sleep(DANCE_STOP_CHECK_INTERVAL)
                continue

            gyro_x = abs(gyro_data.get("gyro-x", 0))
            euler_x = gyro_data.get("euler-x", 92.0)

            # 短窗口缓冲
            gyro_buf.append(gyro_x)
            if len(gyro_buf) > DANCE_STOP_WIN_SIZE:
                gyro_buf.pop(0)

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
                    print(f"  [跳舞] ✓ 检测到舞蹈停止（静止{still_counter}帧，耗时{elapsed:.1f}s）"
                          f" gyro_var={g_var:.1f} gyro_mean={g_mean:.1f} euler={euler_x:.1f}°")
                    return
            else:
                if still_counter > 0:
                    # 静止被打破（舞蹈又动了），重置计数
                    still_counter = 0

            time.sleep(DANCE_STOP_CHECK_INTERVAL)

        # 超时兜底
        elapsed = time.time() - start
        print(f"  [跳舞] ⏰ 超时（{elapsed:.1f}s/{timeout}s），强制结束")

    def _perform_arranged(self, emotion, user_text):
        """随机选放音乐（完整播放）或跳舞（等动作完成）
        v123: 避免连续同类型表演（上次放音乐→这次优先跳舞，反之亦然）"""
        # 决定表演类型：如果上次类型已知，70%概率选不同类型
        if self.last_perform_type == "music":
            choose_music = random.random() < 0.3  # 上次音乐，这次30%再选音乐
        elif self.last_perform_type == "dance":
            choose_music = random.random() < 0.7  # 上次跳舞，这次70%选音乐
        else:
            choose_music = random.choice([True, False])  # 首次，随机

        if choose_music:
            print(f"  [表演] 选择：放音乐（完整播放）{'（上次跳舞，优先音乐）' if self.last_perform_type == 'dance' else ''}")
            result = self._play_full_music(emotion, user_text)
            if result:
                self.last_perform_type = "music"
            return result
        else:
            print(f"  [表演] 选择：跳舞（等动作完成）{'（上次音乐，优先跳舞）' if self.last_perform_type == 'music' else ''}")
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"
            return True

    def _do_pushup(self, emotion):
        """俯卧撑表演：告知→做俯卧撑（确认做完）→鞠躬→起立
        v123: 记录 last_perform_type

        核心设计：俯卧撑时机器人身体大幅倾斜（euler-x 从93°变到≈0°~180°），
        陀螺仪数据会触发摔倒误判。因此全程禁用摔倒检测，
        改用纯动作完成检测算法（wait_motion_done 自适应轮询）。
        """
        intro = generate_pushup_intro(emotion)
        print(f"  [俯卧撑告知] {intro}")
        # 用告知动作说过渡语
        self._announce_with_motion(intro, emotion)

        # ====== 摔倒检测安全关闭（try/finally 保证异常时也能恢复）======
        fall_was_paused = False
        if self.fall_watcher:
            fall_was_paused = self.fall_watcher.paused
            self.fall_watcher.paused = True
            # 清理中间状态：防止残留的异常计数在恢复后误触发
            self.fall_watcher.fall_counter = 0
            self.fall_watcher.euler_x_samples.clear()
            self.fall_watcher.euler_x_samples_raw.clear()
            self.fall_watcher._short_gyro_buf.clear()
            self.fall_watcher._impact_detected = False
            self.fall_watcher._prev_accel_total = None
            print(f"  [摔倒检测] ⏸ 已关闭（俯卧撑模式：纯动作检测）")

        try:
            # 执行俯卧撑动作
            print(f"  [俯卧撑] 执行 PushUp...")
            self.robot.play_motion("PushUp", wait=False, scene="讲话")

            # ====== 纯动作完成检测（自适应退避轮询算法）======
            # 不依赖摔倒检测，完全靠 /motions state 轮询判断
            done = self.robot.wait_motion_done(timeout=36, _bot=self,
                                               motion_name="PushUp")  # min_run_time 自动查表 MOTION_TIME_TABLE
            if not done:
                # 超时或中断时额外等待3秒缓冲（机器人可能有延迟上报）
                print(f"  [俯卧撑] ⚠️ 首次检测未确认完成，缓冲3s后补查...")
                time.sleep(3.0)
                try:
                    r = self.robot.req("GET", "/motions", timeout=2)
                    state = r.get("data", {}).get("state")
                    state_str = str(state).strip().lower() if state is not None else ""
                    if state_str in ("0", "idle", ""):
                        print(f"  [俯卧撑] ✓ 补查确认完成(state=idle)")
                        done = True
                    else:
                        print(f"  [俯卧撑] ⚠️ 补查仍显示执行中(state={state})，强制继续")
                except Exception:
                    print(f"  [俯卧撑] ⚠️ 补查请求失败，强制继续")

            print(f"  [俯卧撑] 动作{'✓已完成' if done else '⚠可能未完成'}，进入鞠躬→起立流程")
            # 鞠躬 → 起立
            self.robot.bow()
            self.robot.stand_up(get_emotion_level(emotion))
            print(f"  [俯卧撑] 表演结束")
            self.last_perform_type = "pushup"

        finally:
            # ====== 无论成功/异常，都恢复摔倒检测 ======
            if self.fall_watcher:
                # 仅恢复到之前的状态（如果本来就是暂停的不重复操作）
                if not fall_was_paused:
                    self.fall_watcher.paused = False
                # 全部重置，给滤波器一个干净的起点（机器人刚做完俯卧撑姿态可能不稳）
                self.fall_watcher.fall_counter = 0
                self.fall_watcher.euler_x_samples.clear()
                self.fall_watcher.euler_x_samples_raw.clear()
                self.fall_watcher._short_gyro_buf.clear()
                self.fall_watcher._impact_detected = False
                self.fall_watcher._prev_accel_total = None
                # 重置互补滤波器（俯卧撑后 euler 偏移大，不重置会误判）
                self.fall_watcher.cf_euler_x = 92.0
                self.fall_watcher.cf_accel_angle = 92.0
                self.fall_watcher.cf_stand_frames = 0
                print(f"  [摔倒检测] ▶ 已恢复（全部状态已重置）")

    # ===== 情绪跳转限制（第二版） =====
    def _limit_emotion_transition(self, current_level, detected_emotion):
        """根据当前强度等级限制情绪跳转目标。返回: (level, emotion) 或 ("__FORCE__", emotion, [allowed]) 或 None"""
        if not detected_emotion or detected_emotion not in ALL_EMOTIONS:
            return None
        new_level = get_emotion_level(detected_emotion)
        if current_level == "high" and new_level in ("low", "positive"):
            return ("__FORCE__", detected_emotion, ("mid", "mid_high"))
        if current_level == "mid_high" and new_level == "positive":
            return ("__FORCE__", detected_emotion, ("mid", "low"))
        return (new_level, detected_emotion)

    def _process_jump(self, current_level, initial_emotion, detected_emotion, user_text):
        """处理情绪跳转，返回 (level, emotion, user_text)"""
        limit = self._limit_emotion_transition(current_level, detected_emotion)
        if not limit:
            return None
        if limit[0] == "__FORCE__":
            _, det_e, allowed = limit
            new_level = self._map_emotion_to_allowed(initial_emotion, det_e, user_text, allowed)
            print(f"  [跳转] {current_level}→{new_level}（{det_e}被限制）")
            return (new_level, detected_emotion, user_text)
        return (limit[0], limit[1], user_text)

    def _map_emotion_to_allowed(self, initial_emotion, detected_emotion, user_text, allowed):
        """DeepSeek判断应该跳转到哪个允许的强度级别"""
        allowed_cn = {"mid": "中强度(5-7)", "mid_high": "中高强度(7-8)", "low": "低强度(3-5)", "positive": "正面(2-3)"}
        allowed_str = "、".join(allowed_cn[a] for a in allowed)
        prompt = f"""初始情绪是「{initial_emotion}」，用户说：「{user_text}」
检测到情绪是「{detected_emotion}」，当前强度不能跳转到低强度或正面。
必须从以下级别中选择：{allowed_str}
只返回级别英文名（mid / mid_high / low / positive），不要解释。"""
        result = _call_deepseek(prompt, max_tokens=20, temp=0.3)
        if result:
            result = result.strip().strip('"')
            if result in allowed:
                return result
        return allowed[0]

    def _ask_comfort(self, emotion, timeout_sec):
        """安慰式询问，返回: ("timeout",) / ("emotion", new_e, reply) / ("continue", reply)
        v124: 关键词优先——只要检测到不同情绪的关键词就切换"""
        checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)
        self.previous_checkins.append(checkin)
        print(f"  [安慰询问] {checkin}")
        self._comfort_with_motion(checkin, emotion)
        reply = self.wait_speech(timeout=timeout_sec)

        # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
        if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
            face_emotion = reply[1]
            face_result = reply[2]
            address = face_result.get("address", self._current_address)
            print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}，立即触发跳转")
            
            # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
            # 保存称呼后，将元组重新赋值为只有 address 的空数据
            reply = ("_face_emotion_", face_emotion, {"address": address})
            
            self._current_address = address
            self._current_voice = face_result.get("voice", self._current_voice)
            self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
            if self.face_manager:
                self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
            return ("emotion", face_emotion, "")  # 立即返回情绪跳转

        if not reply:
            timeout_resp = generate_timeout_response(emotion)
            print(f"  [安慰询问-超时] {timeout_resp}")
            self._comfort_with_motion(timeout_resp, emotion)
            return ("timeout",)
        print(f"  [用户说] {reply}")
        # ★ v125: 使用智能意图检测（关键词优先 → DeepSeek分析）
        intent = detect_intent(reply, current_emotion=emotion)
        print(f"  [意图检测结果] {intent}")
        
        # 如果是情绪切换，return情绪
        if intent[0] == "emotion":
            return intent
        # 如果是表演意图，return表演（让调用方处理具体表演）
        if intent[0] in ["play_song", "play_music", "dance", "pushup", "demand"]:
            return intent
        # continue 或其他
        return intent

    def _ask_comfort_loop(self, emotion, timeout_sec):
        """安慰式询问（超时则重新问），返回: ("emotion", new_e, reply) / ("continue", reply)"""
        while True:
            result = self._ask_comfort(emotion, timeout_sec)
            if result[0] == "timeout":
                continue
            return result

    def _perform_choice(self, emotion):
        """问用户想看什么表演，返回: "music" / "dance" / "pushup"（默认）"""
        prompt = f"""用户刚看完表演，情绪「{emotion}」。
问用户接下来想看什么：听歌、看跳舞、还是看俯卧撑？
要求：15-30字，自然随意。只返回这句话。"""
        choice_q = _call_deepseek(prompt, max_tokens=50, temp=0.9) or "接下来想看什么？听歌、跳舞、还是俯卧撑？"
        print(f"  [表演选择] {choice_q}")
        self._comfort_with_motion(choice_q, emotion)
        reply = self.wait_speech(timeout=10)
        if not reply:
            print("  [表演选择] 超时，默认俯卧撑")
            return "pushup"
        print(f"  [用户选择] {reply}")
        if any(kw in reply for kw in ["歌", "音乐", "听"]):
            return "music"
        if any(kw in reply for kw in ["舞", "跳舞", "跳个", "跳支"]):
            return "dance"
        return "pushup"

    def _ask_with_repeat_on_timeout(self, emotion, timeout_sec, action_func):
        """询问用户，超时则重新询问并执行指定动作。
        返回: ("emotion", new_e, reply) / ("continue", reply) / None(超时次数用完)
        v125: 使用智能意图检测
        """
        while True:
            checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)
            self.previous_checkins.append(checkin)
            print(f"  [询问] {checkin}")
            self._comfort_with_motion(checkin, emotion)
            reply = self.wait_speech(timeout=timeout_sec)

            # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
            if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
                face_emotion = reply[1]
                face_result = reply[2]
                address = face_result.get("address", self._current_address)
                print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}，立即触发跳转")
                
                # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
                reply = ("_face_emotion_", face_emotion, {"address": address})
                
                self._current_address = address
                self._current_voice = face_result.get("voice", self._current_voice)
                if self.face_manager:
                    self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
                return ("emotion", face_emotion, "")  # 立即返回情绪跳转

            if not reply:
                print(f"  [询问] 超时，执行动作...")
                action_func()
                continue
            print(f"  [用户说] {reply}")
            # ★ v125: 使用智能意图检测
            intent = detect_intent(reply, current_emotion=emotion)
            print(f"  [意图检测结果] {intent}")
            
            # 如果是情绪切换，return情绪
            if intent[0] == "emotion":
                return intent
            # 如果是表演意图，return表演
            if intent[0] in ["play_song", "play_music", "dance", "pushup", "demand"]:
                return intent
            # continue 或其他
            return intent

    def _ask_comfort_with_limit(self, emotion, timeout_sec, max_timeouts, jump_target):
        """安慰式询问，超时重问，达到max_timeouts次后跳转到指定强度。
        返回: ("emotion", new_e, reply) / ("jump",) / ("continue", reply)
        v125: 使用智能意图检测
        """
        timeout_count = 0
        while True:
            checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)
            self.previous_checkins.append(checkin)
            print(f"  [安慰询问] {checkin}")
            self._comfort_with_motion(checkin, emotion)
            reply = self.wait_speech(timeout=timeout_sec)

            # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
            if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
                face_emotion = reply[1]
                face_result = reply[2]
                address = face_result.get("address", self._current_address)
                print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}，立即触发跳转")
                
                # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
                reply = ("_face_emotion_", face_emotion, {"address": address})
                
                self._current_address = address
                self._current_voice = face_result.get("voice", self._current_voice)
                if self.face_manager:
                    self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
                return ("emotion", face_emotion, "")  # 立即返回情绪跳转

            if not reply:
                timeout_count += 1
                timeout_resp = generate_timeout_response(emotion)
                print(f"  [安慰询问-超时{timeout_count}] {timeout_resp}")
                self._comfort_with_motion(timeout_resp, emotion)
                if timeout_count >= max_timeouts:
                    print(f"  [安慰询问] 连续超时{max_timeouts}次，跳转{jump_target}")
                    return ("jump",)
                continue
            print(f"  [用户说] {reply}")
            # ★ v125: 使用智能意图检测
            intent = detect_intent(reply, current_emotion=emotion)
            print(f"  [意图检测结果] {intent}")
            
            # 如果是情绪切换，return情绪
            if intent[0] == "emotion":
                return intent
            # 如果是表演意图，return表演
            if intent[0] in ["play_song", "play_music", "dance", "pushup", "demand"]:
                return intent
            # continue 或其他
            return intent

    def _ask_comfort_with_demand(self, emotion, timeout_sec):
        """安慰式询问，可捕捉用户需求（看表演）和心情跳转。
        返回: 
          ("emotion", new_e, reply) - 情绪切换
          ("play_song", song_name) - 播放指定歌曲
          ("play_music",) - 放音乐
          ("dance",) - 跳舞
          ("pushup",) - 俯卧撑
          ("demand",) - 通用表演需求
          ("stop",) - 结束
          ("timeout",) - 超时
          ("continue",) - 继续对话
        v125: 使用智能意图检测（关键词+DeepSeek分析）
              + 连续3次表演请求后允许检测结束词退出
        """
        checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)
        self.previous_checkins.append(checkin)
        print(f"  [安慰询问] {checkin}")
        self._comfort_with_motion(checkin, emotion)
        reply = self.wait_speech(timeout=timeout_sec)

        # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
        if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
            face_emotion = reply[1]
            face_result = reply[2]
            address = face_result.get("address", self._current_address)
            print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}，立即触发跳转")
            
            # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
            # 保存称呼后，将元组重新赋值为只有 address 的空数据
            reply = ("_face_emotion_", face_emotion, {"address": address})
            
            self._current_address = address
            self._current_voice = face_result.get("voice", self._current_voice)
            self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
            if self.face_manager:
                self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
            return ("emotion", face_emotion, "")  # 立即返回情绪跳转

        if not reply:
            timeout_resp = generate_timeout_response(emotion)
            print(f"  [安慰询问-超时] {timeout_resp}")
            self._comfort_with_motion(timeout_resp, emotion)
            return ("timeout",)
        print(f"  [用户说] {reply}")
        
        # ★ v125: 检查结束词（1. 正面情绪直接检查 2. 连续3次表演后也检查）
        if _is_stop(reply) or self.consecutive_perform_count >= 3:
            if _is_stop(reply):
                print(f"  [结束词检测] 检测到结束词，允许退出")
                return ("stop",)
            elif self.consecutive_perform_count >= 3:
                # 连续3次表演后，检查是否有结束意图
                print(f"  [连续表演] 已连续{self.consecutive_perform_count}次，检查结束意图...")
                # 用DeepSeek判断是否是结束意图
                is_end = check_if_user_wants_to_stop(reply)
                if is_end:
                    print(f"  [结束意图] 用户想退出")
                    return ("stop",)
        
        # ★ v125: 智能意图检测（关键词优先 → DeepSeek分析）
        intent = detect_intent(reply, current_emotion=emotion)
        print(f"  [意图检测结果] {intent}")
        
        # ★ v125: 更新连续表演计数
        if intent[0] in ["play_song", "play_music", "dance", "pushup", "demand"]:
            self.consecutive_perform_count += 1
            print(f"  [连续表演计数] {self.consecutive_perform_count}/3")
        elif intent[0] == "emotion":
            # 检测到情绪，重置计数
            self.consecutive_perform_count = 0
        else:
            # continue, timeout 等情况不改变计数
            pass
        
        # 直接返回意图结果，让调用方处理
        return intent

    # ★ 人脸识别版询问方法（3层优先级）
    def _ask_with_face_recognition(self, emotion, timeout_sec, is_checkin=False):
        """★ 人脸识别版安慰式询问（带3层优先级）

        优先级规则:
        1. 人脸识别成功 → 使用人脸识别结果（称呼+情绪）
        2. 人脸失败 + 关键词成功 → 使用关键词结果
        3. 都失败 → 使用内容分析结果
        4. 都没有 → 使用默认值（称呼"朋友"）

        Args:
            emotion: 当前情绪
            timeout_sec: 等待回复超时时间
            is_checkin: 是否是递进式询问（False=安慰询问，True=递进询问）

        Returns:
            同 _ask_comfort_with_demand 的返回值格式
        """
        # 1. 生成询问话术（使用人脸识别结果生成带称呼的话）
        if is_checkin:
            # 递进询问
            checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)
        else:
            # 安慰询问
            checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)

        self.previous_checkins.append(checkin)
        print(f"  [安慰询问] {checkin}")

        # ★ 暂停期间跳过人脸检测，直接说话
        if self._face_detection_paused:
            self._comfort_with_motion(checkin, emotion)
            return self.wait_speech(timeout=timeout_sec)

        # 2. 并行执行：说询问话 + 人脸识别
        # 启动人脸识别（如果可用）
        face_result = None
        if self._use_face_recognition and self.face_manager:
            # 在说话的同时进行人脸识别
            import threading
            def async_detect():
                nonlocal face_result
                # ★★★ 强制全新检测（忽略任何缓存）★★★
                face_result = self.face_manager.detect(force_new=True)

            detect_thread = threading.Thread(target=async_detect)
            detect_thread.start()

            # 先说询问话
            self._comfort_with_motion(checkin, emotion)

            # 等待人脸识别完成（最多2秒）
            detect_thread.join(timeout=2.0)
        else:
            self._comfort_with_motion(checkin, emotion)

        # 3. 等待用户回复
        reply = self.wait_speech(timeout=timeout_sec)

        # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
        if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
            face_emotion = reply[1]
            face_result = reply[2]
            address = face_result.get("address", self._current_address)
            print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}，立即触发跳转")
            
            # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
            reply = ("_face_emotion_", face_emotion, {"address": address})
            
            self._current_address = address
            self._current_voice = face_result.get("voice", self._current_voice)
            self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
            self._face_jump_just_triggered = True  # ★ 标记刚触发过人脸情绪跳转
            self._face_detection_paused = True  # ★ 暂停人脸检测
            self._face_detection_stop_event.set()  # ★ 立即停止正在运行的人脸检测线程
            if self.face_manager:
                self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
            return ("emotion", face_emotion, "")  # 立即返回情绪跳转

        if not reply:
            timeout_resp = generate_timeout_response(emotion)
            print(f"  [安慰询问-超时] {timeout_resp}")
            self._comfort_with_motion(timeout_resp, emotion)
            return ("timeout",)

        print(f"  [用户说] {reply}")

        # ★ 4. 三层优先级识别
        # 第1层：人脸识别情绪
        keyword_emotion = None
        keyword_confidence = 0.0

        # 第2层：关键词识别
        kw_result = _detect_emotion_keyword_with_confidence(reply)
        if kw_result[0]:
            keyword_emotion = kw_result[0]
            keyword_confidence = kw_result[1]
            print(f"  [关键词识别] {keyword_emotion} (置信度 {keyword_confidence:.1f})")

        # 第3层：内容分析（detect_intent）
        intent = detect_intent(reply, current_emotion=emotion)
        content_emotion = intent[1] if intent[0] == "emotion" else None

        # ★ 优先级决策
        final_emotion = emotion
        final_address = self._current_address
        final_voice = self._current_voice

        if face_result and face_result.get("success"):
            # 人脸识别成功，使用其结果
            final_emotion = face_result.get("emotion", emotion)
            final_address = face_result.get("address", "朋友")
            final_voice = face_result.get("voice", "x4_yezi")
            print(f"  [人脸识别-优先] emotion={final_emotion}, address={final_address}, voice={final_voice}")
        elif keyword_emotion and keyword_confidence >= 0.6:
            # 关键词成功
            final_emotion = keyword_emotion
            print(f"  [关键词识别-使用] emotion={final_emotion}")
        elif content_emotion:
            # 内容分析成功
            final_emotion = content_emotion
            print(f"  [内容分析-使用] emotion={final_emotion}")
        else:
            # 都失败，使用默认值
            print(f"  [识别-使用默认值] emotion={emotion}, address=朋友")

        # 更新当前称呼和音色（用于后续的话术生成）
        self._current_address = final_address
        self._current_voice = final_voice

        # ★ 5. 检查结束词
        if _is_stop(reply) or self.consecutive_perform_count >= 3:
            if _is_stop(reply):
                print(f"  [结束词检测] 检测到结束词，允许退出")
                return ("stop",)
            elif self.consecutive_perform_count >= 3:
                print(f"  [连续表演] 已连续{self.consecutive_perform_count}次，检查结束意图...")
                is_end = check_if_user_wants_to_stop(reply)
                if is_end:
                    print(f"  [结束意图] 用户想退出")
                    return ("stop",)

        # ★ 6. 智能意图检测（用于表演等意图）
        print(f"  [意图检测结果] {intent}")

        # 更新连续表演计数
        if intent[0] in ["play_song", "play_music", "dance", "pushup", "demand"]:
            self.consecutive_perform_count += 1
            print(f"  [连续表演计数] {self.consecutive_perform_count}/3")
        elif intent[0] == "emotion":
            self.consecutive_perform_count = 0

        return intent

    def _ask_response_question(self, emotion, timeout_sec):
        """正面情绪的回应询问，语气轻松活泼，可捕捉需求/心情/结束词。
        返回: 
          ("emotion", new_e, reply) - 情绪切换
          ("play_song", song_name) - 播放指定歌曲
          ("play_music",) - 放音乐
          ("dance",) - 跳舞
          ("pushup",) - 俯卧撑
          ("demand",) - 通用表演需求
          ("stop",) - 结束
          ("timeout",) - 超时
          ("continue",) - 继续对话
        v125: 使用智能意图检测
              + 连续3次表演请求后允许检测结束词退出
        """
        checkin = generate_response_question(emotion, self.round_num, self.previous_checkins, self._current_address)
        self.previous_checkins.append(checkin)
        print(f"  [回应询问] {checkin}")
        self._comfort_with_motion(checkin, emotion)
        reply = self.wait_speech(timeout=timeout_sec)

        # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
        if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
            face_emotion = reply[1]
            face_result = reply[2]
            address = face_result.get("address", self._current_address)
            print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}，立即触发跳转")
            
            # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
            reply = ("_face_emotion_", face_emotion, {"address": address})
            
            self._current_address = address
            self._current_voice = face_result.get("voice", self._current_voice)
            self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
            if self.face_manager:
                self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
            return ("emotion", face_emotion, "")

        # ★ 人脸情绪优先：用户没说话但检测到人脸情绪 → 触发情绪跳转
        if not reply and self._pending_face_result:
            face_emotion = self._pending_face_result.get("emotion")
            print(f"  [人脸情绪-跳转] 用户未开口，检测到情绪: {face_emotion}")
            address = self._pending_face_result.get("address", self._current_address)
            self._current_address = address
            self._current_voice = self._pending_face_result.get("voice", self._current_voice)
            self._pending_face_result = None  # ★ 清空 pending 结果
            self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
            if self.face_manager:
                self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
            return ("emotion", face_emotion, "")

        if not reply:
            timeout_resp = generate_timeout_response(emotion)
            print(f"  [回应询问-超时] {timeout_resp}")
            self._comfort_with_motion(timeout_resp, emotion)
            return ("timeout",)
        print(f"  [用户说] {reply}")
        
        # ★ v125: 检查结束词（1. 正面情绪直接检查 2. 连续3次表演后也检查）
        if _is_stop(reply) or self.consecutive_perform_count >= 3:
            if _is_stop(reply):
                print(f"  [结束词检测] 检测到结束词，允许退出")
                return ("stop",)
            elif self.consecutive_perform_count >= 3:
                # 连续3次表演后，检查是否有结束意图
                print(f"  [连续表演] 已连续{self.consecutive_perform_count}次，检查结束意图...")
                is_end = check_if_user_wants_to_stop(reply)
                if is_end:
                    print(f"  [结束意图] 用户想退出")
                    return ("stop",)
        
        # ★ v125: 智能意图检测（关键词优先 → DeepSeek分析）
        intent = detect_intent(reply, current_emotion=emotion)
        print(f"  [意图检测结果] {intent}")
        
        # ★ v125: 更新连续表演计数
        if intent[0] in ["play_song", "play_music", "dance", "pushup", "demand"]:
            self.consecutive_perform_count += 1
            print(f"  [连续表演计数] {self.consecutive_perform_count}/3")
        elif intent[0] == "emotion":
            # 检测到情绪，重置计数
            self.consecutive_perform_count = 0
        else:
            # continue, timeout 等情况不改变计数
            pass
        
        # 直接返回意图结果，让调用方处理
        return intent

    # ★ 带人脸识别的安慰话方法
    def _comfort_with_face_motion(self, base_text, emotion, detected_emotion=None):
        """★ 带人脸识别的安慰话（插入称呼和看到情绪的反应）

        如果人脸识别成功，会将 base_text 转换为带称呼的格式：
        "称呼，我看到你有点XX呢，..." + base_text

        Args:
            base_text: 基础安慰话
            emotion: 当前情绪
            detected_emotion: 人脸识别的情绪（可选）
        """
        if self._use_face_recognition and self.face_manager:
            # 人脸识别结果
            face_result = self.face_manager.detect()
            if face_result and face_result.get("success"):
                address = face_result.get("address", self._current_address)
                face_emotion = face_result.get("emotion")

                # 生成带称呼的开场白
                intro = generate_face_comfort_intro(emotion, address, face_emotion)
                print(f"  [人脸识别-开场白] {intro}")

                # 更新当前称呼和音色
                self._current_address = address
                self._current_voice = face_result.get("voice", self._current_voice)

                # 先说开场白
                self._comfort_with_motion(intro, emotion)

                # 再说安慰话
                self._comfort_with_motion(base_text, emotion)
                return

        # 如果没有人脸识别或识别失败，使用原有逻辑
        self._comfort_with_motion(base_text, emotion)

    # ===== 高强度(9-10) =====
    def cycle_high(self, emotion, user_text):
        """高强度(9-10):
        安慰→表演→询问1(6秒，超时→表演)→表演→询问2(6秒，超时→俯卧撑)→俯卧撑→安慰式询问(超时重问，2次→中高)
        """
        self._check_fall()
        initial_emotion = emotion

        # 1. 安慰
        self.round_num += 1
        resp, mid, mdesc = generate_comfort_with_address(emotion, user_text, self._current_address, self.round_num, self.previous_responses)
        print(f"  [高强度-安慰-{self.round_num}] {resp}")
        self.previous_responses.append(resp)
        self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 表演
        print("\n  [高强度] 表演...")
        self._perform_arranged(emotion, user_text)

        # 3. 询问1（6秒，超时→表演）
        result = self._ask_with_repeat_on_timeout(emotion, timeout_sec=10,
                                                  action_func=lambda: self._perform_arranged(emotion, user_text))
        if result and result[0] == "stop":
            return ("stop", "")
        elif result and result[0] == "emotion":
            return self._process_jump("high", initial_emotion, result[1], result[2])
        # result[0] == "continue": 用户说了但没识别情绪，继续下一步

        # 4. 表演
        print("\n  [高强度] 再表演...")
        self._perform_arranged(emotion, user_text)

        # 5. 询问2（6秒，超时→俯卧撑）
        result = self._ask_with_repeat_on_timeout(emotion, timeout_sec=10, action_func=lambda: self._do_pushup(emotion))
        if result and result[0] == "stop":
            return ("stop", "")
        elif result and result[0] == "emotion":
            return self._process_jump("high", initial_emotion, result[1], result[2])
        # result[0] == "continue": 用户说了但没识别情绪，执行俯卧撑

        # 6. 俯卧撑
        print("\n  [高强度] 俯卧撑...")
        self._do_pushup(emotion)
        # 7. 安慰式询问（超时重问，连续超时2次→中高强度）
        final = self._ask_comfort_with_limit(emotion, timeout_sec=10, max_timeouts=2, jump_target="mid_high")
        if final[0] == "stop":
            return ("stop", "")
        elif final[0] == "emotion":
            return self._process_jump("high", initial_emotion, final[1], final[2])
        elif final[0] == "jump":
            return ("mid_high", initial_emotion, "")
        return None

    # ===== 中高强度(7-8) =====
    def cycle_mid_high(self, emotion, user_text):
        """中高强度(7-8):
        安慰→表演→询问1(6秒，超时→表演)→跳舞→安慰式询问(超时重问，2次→中)
        """
        self._check_fall()
        initial_emotion = emotion

        # 1. 安慰
        self.round_num += 1
        resp, mid, mdesc = generate_comfort_with_address(emotion, user_text, self._current_address, self.round_num, self.previous_responses)
        print(f"  [中高强度-安慰-{self.round_num}] {resp}")
        self.previous_responses.append(resp)
        self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 表演
        print("\n  [中高强度] 表演...")
        self._perform_arranged(emotion, user_text)

        # 3. 询问1（6秒，超时→表演）
        result = self._ask_with_repeat_on_timeout(emotion, timeout_sec=10,
                                                  action_func=lambda: self._perform_arranged(emotion, user_text))
        if result and result[0] == "stop":
            return ("stop", "")
        elif result and result[0] == "emotion":
            return self._process_jump("mid_high", initial_emotion, result[1], result[2])
        # result[0] == "continue": 用户说了但没识别情绪，继续下一步

        # 4. 跳舞
        print("\n  [中高强度] 跳舞...")
        self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
        self.last_perform_type = "dance"  # v123

        # 5. 安慰式询问（超时重问，连续超时2次→中强度）
        final = self._ask_comfort_with_limit(emotion, timeout_sec=10, max_timeouts=2, jump_target="mid")
        if final[0] == "stop":
            return ("stop", "")
        elif final[0] == "emotion":
            return self._process_jump("mid_high", initial_emotion, final[1], final[2])
        elif final[0] == "jump":
            return ("mid", initial_emotion, "")
        return None

    # ===== 中强度(5-7) =====
    def cycle_mid(self, emotion, user_text):
        """中强度(5-7):
        安慰→放音乐→询问1(6秒，超时→跳舞)→跳舞→循环(安慰式询问(6秒，超时→表演)→表演)
        安慰式询问可捕捉需求(问歌/舞/俯卧撑)和心情跳转
        
        ★ CBT增强：如果检测到认知扭曲，使用CBT对话替代常规安慰
        """
        self._check_fall()
        
        # ★ CBT检测：判断是否应该使用CBT模式
        use_cbt = False
        if emotion in CBT_SUITABLE_EMOTIONS:
            use_cbt = should_use_cbt(emotion, user_text)
            if use_cbt:
                confidence = self._cbt_engine.get_activation_confidence(emotion, user_text)
                print(f"  [CBT检测] 检测到认知扭曲特征，置信度={confidence:.2f}，启用CBT模式")
        
        if use_cbt:
            return self._cycle_mid_cbt(emotion, user_text)
        
        # 1. 安慰（常规模式）
        self.round_num += 1
        resp, mid, mdesc = generate_comfort_with_address(emotion, user_text, self._current_address, self.round_num, self.previous_responses)
        print(f"  [中强度-安慰-{self.round_num}] {resp}")
        self.previous_responses.append(resp)
        self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 放音乐（完整）
        print("\n  [中强度] 放音乐...")
        self._play_full_music(emotion, user_text)
        self.last_perform_type = "music"  # v123

        # 3. 询问1（6秒，超时→跳舞）
        result = self._ask_comfort(emotion, timeout_sec=10)
        if result[0] == "timeout":
            print("\n  [中强度] 超时，跳舞...")
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"  # v123
        elif result[0] == "stop":
            return ("stop", "")
        elif result[0] == "emotion":
            return (get_emotion_level(result[1]), result[1], result[2])
        else:
            user_text = result[1]
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"  # v123

        # 4. 循环：安慰式询问(6秒，超时→表演) → 表演
        while True:
            result = self._ask_comfort_with_demand(emotion, timeout_sec=10)
            # ★ v125: 处理新的意图类型
            if result[0] == "play_song":
                # 播放指定歌曲
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                # 放音乐（随机）
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                # 跳舞（可能指定了具体舞蹈）
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                # 俯卧撑
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                # 通用表演需求，问用户想看什么
                choice = self._perform_choice(emotion)
                if choice == "music":
                    self._play_full_music(emotion, user_text)
                    self.last_perform_type = "music"
                elif choice == "dance":
                    self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
                    self.last_perform_type = "dance"
                elif choice == "pushup":
                    self._do_pushup(emotion)
                    self.last_perform_type = "pushup"
                else:
                    self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "stop":
                return ("stop", "")
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                # 超时，表演
                print("\n  [中强度] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                user_text = result[1] if len(result) > 1 else user_text
                self._perform_arranged(emotion, user_text)

    # ===== CBT增强的中强度循环 =====
    def _cycle_mid_cbt(self, emotion, user_text):
        """CBT增强的中强度循环(5-7)：
        结构：安慰(带CBT过渡) → CBT对话 → 音乐放松 → 询问→跳转/继续
        
        CBT对话流程（由cbt_engine驱动）：
          IDENTIFY（识别自动思维）→ CHALLENGE（质疑）→ RESTRUCTURE（重建）→ CLOSE
        
        设计原则：
        - CBT不是替代陪伴，而是在陪伴中嵌入引导式自我探索
        - CBT对话最多5轮，避免过度消耗用户能量
        - CBT结束后用音乐放松，让新认知有时间沉淀
        """
        self._check_fall()
        initial_emotion = emotion
        
        # 1. 开始CBT会话
        session = start_cbt_session(emotion, user_text)
        self._cbt_active = True
        print(f"  [中强度-CBT] 启动CBT会话 | 轮次={session.round_count}")
        
        # 2. CBT过渡：先说一句过渡话，进入自我探索
        intro_text = generate_cbt_intro(emotion, user_text, self._current_address)
        print(f"  [中强度-CBT-过渡] {intro_text}")
        self._comfort_with_motion(intro_text, emotion, pre_selected_motion=("H_Bec_L", CBT_MOTION_OPTIONS["H_Bec_L"]))
        time.sleep(1.5)  # 给用户一点反应时间
        
        # 3. CBT对话循环
        cbt_text, cbt_phase = get_cbt_response()  # 第一轮：机器人提问
        self._cbt_round_count = 1
        
        while cbt_phase != CBTPhase.CLOSE and self._cbt_round_count < CBT_MAX_ROUNDS:
            # 用CBT专属动作发声
            print(f"  [中强度-CBT-{cbt_phase}-{self._cbt_round_count}] {cbt_text}")
            
            # 选择一个适合当前CBT阶段的动作
            if cbt_phase == CBTPhase.IDENTIFY:
                motion_id = "H_Bec_L"  # 侧身倾听
            elif cbt_phase == CBTPhase.CHALLENGE:
                motion_id = "Hd_Wacth_F"  # 前倾注视
            elif cbt_phase == CBTPhase.RESTRUCTURE:
                motion_id = "H_Rise_B"  # 举双手（鼓励）
            else:
                motion_id = "H_Wave_R"  # 挥手（温和）
            motion_desc = CBT_MOTION_OPTIONS.get(motion_id, "")
            
            self._comfort_with_motion(cbt_text, emotion, pre_selected_motion=(motion_id, motion_desc))
            
            # 等待用户回应
            print(f"\n  [中强度-CBT] 等待用户回应（{LISTEN_TIMEOUT}秒）...")
            user_reply = self.wait_speech(timeout=LISTEN_TIMEOUT)
            
            if not user_reply:
                # 超时：用温和的提醒继续
                print("  [中强度-CBT] 超时，发送提醒")
                reminder = "没关系，不用着急回答，想到什么就说什么。"
                self._comfort_with_motion(reminder, emotion, pre_selected_motion=("H_Bec_L", CBT_MOTION_OPTIONS["H_Bec_L"]))
                user_reply = ""  # 空字符串，让引擎自己生成下一轮问题
            else:
                print(f"  [中强度-CBT] 用户回应: {user_reply}")
                
                # 检测情绪跳转（用户在CBT过程中表达了新情绪）
                intent = detect_intent(user_reply, current_emotion=emotion)
                if intent[0] == "emotion":
                    new_e = intent[1]
                    print(f"  [中强度-CBT] 检测到情绪跳转: {new_e}，退出CBT")
                    end_cbt_session()
                    self._cbt_active = False
                    return self._process_jump("mid_cbt", initial_emotion, new_e, user_reply)
                elif intent[0] == "stop":
                    print(f"  [中强度-CBT] 用户要求结束")
                    end_cbt_session()
                    self._cbt_active = False
                    goodbye = generate_goodbye(emotion)
                    self._goodbye_with_bow(goodbye, emotion)
                    return ("stop", emotion, "")
                elif intent[0] in ["dance", "play_music", "play_song", "pushup", "demand"]:
                    # 用户想表演，暂停CBT
                    print(f"  [中强度-CBT] 用户想表演({intent[0]})，暂停CBT")
                    end_cbt_session()
                    self._cbt_active = False
                    # 执行表演后继续
                    if intent[0] == "dance":
                        self._do_dance_for_intent(emotion, intent=intent)
                    elif intent[0] == "play_music":
                        self._play_full_music(emotion, user_reply)
                    elif intent[0] == "play_song":
                        self._play_user_song(emotion, intent[1] if len(intent) > 1 else user_reply)
                    elif intent[0] == "pushup":
                        self._do_pushup(emotion)
                    elif intent[0] == "demand":
                        choice = self._perform_choice(emotion)
                        if choice == "music":
                            self._play_full_music(emotion, user_reply)
                        elif choice == "dance":
                            self._do_dance_for_intent(emotion, intent=intent)
                        elif choice == "pushup":
                            self._do_pushup(emotion)
                    # 表演后询问，不再返回CBT
                    post_greeting = generate_post_show_greeting(emotion)
                    self._comfort_with_motion(post_greeting, emotion)
                    continue
            
            # 生成下一轮CBT回应
            cbt_text, cbt_phase = get_cbt_response(user_reply)
            self._cbt_round_count += 1
        
        # 4. CBT结束，温暖收尾
        distortion_name = ""
        if session and session.primary_distortion:
            distortion_name = session.primary_distortion.name_cn
        closing_text = generate_cbt_closing_with_summary(emotion, user_text, distortion_name, self._current_address)
        print(f"  [中强度-CBT-收尾] {closing_text}")
        self._comfort_with_motion(closing_text, emotion, pre_selected_motion=("H_Rise_B", CBT_MOTION_OPTIONS["H_Rise_B"]))
        
        # 结束CBT会话
        end_cbt_session()
        self._cbt_active = False
        self._cbt_round_count = 0
        print(f"  [中强度-CBT] CBT会话结束")
        
        # 5. CBT后用音乐帮助沉淀新的认知
        print(f"\n  [中强度-CBT] 播放音乐，帮助沉淀...")
        self._play_full_music(emotion, user_text)
        self.last_perform_type = "music"
        
        # 6. 询问（此时用户情绪可能已变化）
        result = self._ask_comfort(emotion, timeout_sec=10)
        if result[0] == "timeout":
            print("\n  [中强度-CBT] 超时，跳舞...")
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"
        elif result[0] == "stop":
            return ("stop", "")
        elif result[0] == "emotion":
            return (get_emotion_level(result[1]), result[1], result[2])
        else:
            user_text = result[1] if len(result) > 1 else user_text
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"
        
        # 7. 循环：像普通cycle_mid一样进入询问循环
        while True:
            result = self._ask_comfort_with_demand(emotion, timeout_sec=10)
            if result[0] == "play_song":
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                choice = self._perform_choice(emotion)
                if choice == "music":
                    self._play_full_music(emotion, user_text)
                elif choice == "dance":
                    self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
                elif choice == "pushup":
                    self._do_pushup(emotion)
                else:
                    self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "stop":
                return ("stop", "")
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                print("\n  [中强度-CBT] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                user_text = result[1] if len(result) > 1 else user_text
                self._perform_arranged(emotion, user_text)

    # ===== 低强度(3-5) =====
    def cycle_low(self, emotion, user_text):
        """低强度(3-5):
        安慰→跳舞→循环(安慰式询问(8秒，超时→表演)→表演)
        获取到用户新情绪就跳转
        """
        self._check_fall()
        # 1. 安慰
        self.round_num += 1
        resp, mid, mdesc = generate_comfort_with_address(emotion, user_text, self._current_address, self.round_num, self.previous_responses)
        print(f"  [低强度-安慰-{self.round_num}] {resp}")
        self.previous_responses.append(resp)
        self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 跳舞
        print("\n  [低强度] 跳舞...")
        self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
        self.last_perform_type = "dance"  # v123

        # 3. 循环：安慰式询问(8秒，超时→表演) → 表演
        while True:
            result = self._ask_comfort_with_demand(emotion, timeout_sec=10)
            # ★ v125: 处理新的意图类型
            if result[0] == "play_song":
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                # 用户想看表演
                self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "stop":
                return ("stop", "")
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                # 超时，表演
                print("\n  [低强度] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                user_text = result[1] if len(result) > 1 else user_text
                self._perform_arranged(emotion, user_text)

    # ===== 正面(2-3) =====
    def cycle_positive(self, emotion, user_text):
        """正面(2-3):
        回应→表演→循环(回应询问(轻松活泼)→超时表演)
        回应询问可捕捉需求/心情/结束词
        """
        self._check_fall()
        # 1. 回应
        self.round_num += 1
        resp = generate_comfort_response(emotion, user_text, self.round_num, self.previous_responses)
        print(f"  [正面-回应-{self.round_num}] {resp}")
        self.previous_responses.append(resp)
        self._comfort_with_motion(resp, emotion)

        # 2. 表演
        print("\n  [正面] 表演...")
        self._perform_arranged(emotion, user_text)

        # 3. 循环：回应询问(轻松活泼) → 超时表演
        while True:
            result = self._ask_response_question(emotion, timeout_sec=10)
            if result[0] == "stop":
                return ("stop", "")
            # ★ v125: 处理新的意图类型
            elif result[0] == "play_song":
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                # 用户想看表演，问他想看什么
                choice = self._perform_choice(emotion)
                if choice == "music":
                    self._play_full_music(emotion, user_text)
                    self.last_perform_type = "music"  # v123
                elif choice == "dance":
                    self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
                    self.last_perform_type = "dance"  # v123
                elif choice == "pushup":
                    self._do_pushup(emotion)
                    self.last_perform_type = "pushup"  # v123
                else:
                    self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                # 超时，表演
                print("\n  [正面] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                self._perform_arranged(emotion, user_text)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 人脸识别版 Cycle 方法（3层优先级）
    # ★ 使用方式：在 run() 中将 cycle_xxx 替换为 cycle_xxx_with_face
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    def cycle_high_with_face(self, emotion, user_text):
        """★ 人脸识别版高强度(9-10)

        相比原版，新增：
        - 安慰话使用 generate_comfort_with_address 生成带称呼的话
        - 询问使用 _ask_with_face_recognition 实现3层优先级
        """
        self._check_fall()
        initial_emotion = emotion

        # 1. 人脸识别 + 安慰
        self.round_num += 1

        # ★★★ 关键逻辑 ★★★
        # 如果刚触发过人脸情绪跳转，跳过人脸检测，使用已保存的称呼
        # 情绪等检测服务：每次都是全新的变量服务（清零）
        # 称呼：只保留第一个检测到的
        if self._face_jump_just_triggered:
            self._face_jump_just_triggered = False  # 重置跳转标志
            self._face_detection_paused = True  # ★ 暂停人脸检测（后续询问也要跳过）
            face_result = None  # 不检测，使用已保存的 _current_address
            print(f"  [高强度-安慰] ★ 跳过人脸检测，使用保存的称呼: {self._current_address}")
            # 生成安慰话（使用已保存的称呼）
            resp, mid, mdesc = generate_comfort_with_address(
                emotion, user_text, self._current_address, self.round_num, self.previous_responses
            )
            print(f"  [高强度-安慰-{self.round_num}] {resp}")
            self.previous_responses.append(resp)
            self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))
        else:
            # ★★★ 正常流程：启动人脸检测 ★★★
            face_result = None
            if self._use_face_recognition and self.face_manager:
                import threading
                def async_detect():
                    nonlocal face_result
                    face_result = self.face_manager.detect()
                detect_thread = threading.Thread(target=async_detect)
                detect_thread.start()
                # 等待检测完成（最多2秒）
                try:
                    detect_thread.join(timeout=2.0)
                except NameError:
                    pass
            
            # 人脸识别成功 → 使用带称呼的话术
            if face_result and face_result.get("success"):
                # ★ 补充生成称呼和音色
                gender = face_result.get("gender")
                age = face_result.get("age")
                face_result["address"] = FaceRecognitionManager._generate_address(gender, age)
                face_result["voice"] = FaceRecognitionManager._select_voice(gender, age)
                address = face_result.get("address", "朋友")
                self._current_address = address  # ★ 保存新称呼
                self._current_voice = face_result.get("voice", "x4_yezi")
                
                print(f"  [人脸识别-安慰] 检测到: {gender}/{age}岁，称呼: {address}")
                
                # 生成带称呼的安慰话
                addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                    emotion, user_text, address, self.round_num, self.previous_responses
                )
                print(f"  [高强度-安慰-{self.round_num}] {addressed_text}")
                self.previous_responses.append(addressed_text)
                self._comfort_with_motion(addressed_text, emotion, pre_selected_motion=(addr_mid, addr_mdesc))
            else:
                # 未检测到人脸，使用默认称呼
                resp, mid, mdesc = generate_comfort_with_address(
                    emotion, user_text, self._current_address, self.round_num, self.previous_responses
                )
                print(f"  [高强度-安慰-{self.round_num}] {resp}")
                self.previous_responses.append(resp)
                self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 表演
        print("\n  [高强度] 表演...")
        self._perform_arranged(emotion, user_text)

        # 3. 询问1（6秒，超时→表演）★ 使用人脸识别版
        result = self._ask_with_face_recognition(emotion, timeout_sec=10)
        if result and result[0] == "stop":
            return ("stop", "")
        elif result and result[0] == "emotion":
            return self._process_jump("high", initial_emotion, result[1], result[2] if len(result) > 2 else "")

        # 4. 表演
        print("\n  [高强度] 再表演...")
        self._perform_arranged(emotion, user_text)

        # 5. 询问2（6秒，超时→俯卧撑）★ 使用人脸识别版
        result = self._ask_with_face_recognition(emotion, timeout_sec=10)
        if result and result[0] == "stop":
            return ("stop", "")
        elif result and result[0] == "emotion":
            return self._process_jump("high", initial_emotion, result[1], result[2] if len(result) > 2 else "")

        # 6. 俯卧撑
        print("\n  [高强度] 俯卧撑...")
        self._do_pushup(emotion)

        # 7. 安慰式询问（超时重问，连续超时2次→中高强度）★ 使用人脸识别版
        final = self._ask_with_face_recognition(emotion, timeout_sec=10)
        if final[0] == "stop":
            return ("stop", "")
        elif final[0] == "emotion":
            return self._process_jump("high", initial_emotion, final[1], final[2] if len(final) > 2 else "")
        elif final[0] == "jump":
            return ("mid_high", initial_emotion, "")
        return None

    def cycle_mid_high_with_face(self, emotion, user_text):
        """★ 人脸识别版中高强度(7-8)"""
        self._check_fall()
        initial_emotion = emotion

        # 1. 人脸识别 + 安慰
        self.round_num += 1

        # ★★★ 关键逻辑 ★★★
        # 情绪等检测服务：每次都是全新的变量服务（清零）
        # 称呼：只保留第一个检测到的
        if self._face_jump_just_triggered:
            self._face_jump_just_triggered = False  # 重置跳转标志
            self._face_detection_paused = True  # ★ 暂停人脸检测（后续询问也要跳过）
            face_result = None
            print(f"  [中高强度-安慰] ★ 跳过人脸检测，使用保存的称呼: {self._current_address}")
            resp, mid, mdesc = generate_comfort_with_address(
                emotion, user_text, self._current_address, self.round_num, self.previous_responses
            )
            print(f"  [中高强度-安慰-{self.round_num}] {resp}")
            self.previous_responses.append(resp)
            self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))
        else:
            face_result = None
            if self._use_face_recognition and self.face_manager:
                import threading
                def async_detect():
                    nonlocal face_result
                    face_result = self.face_manager.detect()
                detect_thread = threading.Thread(target=async_detect)
                detect_thread.start()
                try:
                    detect_thread.join(timeout=2.0)
                except NameError:
                    pass
            
            if face_result and face_result.get("success"):
                gender = face_result.get("gender")
                age = face_result.get("age")
                face_result["address"] = FaceRecognitionManager._generate_address(gender, age)
                face_result["voice"] = FaceRecognitionManager._select_voice(gender, age)
                address = face_result.get("address", "朋友")
                self._current_address = address
                self._current_voice = face_result.get("voice", "x4_yezi")
                print(f"  [人脸识别-安慰] 检测到: {gender}/{age}岁，称呼: {address}")
                addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                    emotion, user_text, address, self.round_num, self.previous_responses
                )
                print(f"  [中高强度-安慰-{self.round_num}] {addressed_text}")
                self.previous_responses.append(addressed_text)
                self._comfort_with_motion(addressed_text, emotion, pre_selected_motion=(addr_mid, addr_mdesc))
            else:
                resp, mid, mdesc = generate_comfort_with_address(
                    emotion, user_text, self._current_address, self.round_num, self.previous_responses
                )
                print(f"  [中高强度-安慰-{self.round_num}] {resp}")
                self.previous_responses.append(resp)
                self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 表演
        print("\n  [中高强度] 表演...")
        self._perform_arranged(emotion, user_text)

        # 3. 询问1（6秒，超时→表演）
        result = self._ask_with_face_recognition(emotion, timeout_sec=10)
        if result and result[0] == "stop":
            return ("stop", "")
        elif result and result[0] == "emotion":
            return self._process_jump("mid_high", initial_emotion, result[1], result[2] if len(result) > 2 else "")

        # 4. 跳舞
        print("\n  [中高强度] 跳舞...")
        self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
        self.last_perform_type = "dance"

        # 5. 安慰式询问（超时重问，连续超时2次→中强度）
        final = self._ask_with_face_recognition(emotion, timeout_sec=10)
        if final[0] == "stop":
            return ("stop", "")
        elif final[0] == "emotion":
            return self._process_jump("mid_high", initial_emotion, final[1], final[2] if len(final) > 2 else "")
        elif final[0] == "jump":
            return ("mid", initial_emotion, "")
        return None

    def cycle_mid_with_face(self, emotion, user_text):
        """★ 人脸识别版中强度(5-7)
        ★ CBT增强：检测到认知扭曲时使用CBT模式
        """
        self._check_fall()

        # 1. 人脸识别 + 安慰
        self.round_num += 1

        # ★★★ 关键逻辑 ★★★
        # 情绪等检测服务：每次都是全新的变量服务（清零）
        # 称呼：只保留第一个检测到的
        if self._face_jump_just_triggered:
            self._face_jump_just_triggered = False  # 重置跳转标志
            self._face_detection_paused = True  # ★ 暂停人脸检测（后续询问也要跳过）
            face_result = None
            print(f"  [中强度-安慰] ★ 跳过人脸检测，使用保存的称呼: {self._current_address}")
            resp, mid, mdesc = generate_comfort_with_address(
                emotion, user_text, self._current_address, self.round_num, self.previous_responses
            )
            print(f"  [中强度-安慰-{self.round_num}] {resp}")
            self.previous_responses.append(resp)
            self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))
        else:
            face_result = None
            if self._use_face_recognition and self.face_manager:
                import threading
                def async_detect():
                    nonlocal face_result
                    face_result = self.face_manager.detect()
                detect_thread = threading.Thread(target=async_detect)
                detect_thread.start()
                try:
                    detect_thread.join(timeout=2.0)
                except NameError:
                    pass
            
            if face_result and face_result.get("success"):
                gender = face_result.get("gender")
                age = face_result.get("age")
                face_result["address"] = FaceRecognitionManager._generate_address(gender, age)
                face_result["voice"] = FaceRecognitionManager._select_voice(gender, age)
                address = face_result.get("address", "朋友")
                self._current_address = address
                self._current_voice = face_result.get("voice", "x4_yezi")
                print(f"  [人脸识别-安慰] 检测到: {gender}/{age}岁，称呼: {address}")
                addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                    emotion, user_text, address, self.round_num, self.previous_responses
                )
                print(f"  [中强度-安慰-{self.round_num}] {addressed_text}")
                self.previous_responses.append(addressed_text)
                self._comfort_with_motion(addressed_text, emotion, pre_selected_motion=(addr_mid, addr_mdesc))
            else:
                resp, mid, mdesc = generate_comfort_with_address(
                    emotion, user_text, self._current_address, self.round_num, self.previous_responses
                )
                print(f"  [中强度-安慰-{self.round_num}] {resp}")
                self.previous_responses.append(resp)
                self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # ★ CBT检测：在安慰后、音乐前判断是否应启用CBT
        use_cbt = False
        if emotion in CBT_SUITABLE_EMOTIONS:
            use_cbt = should_use_cbt(emotion, user_text)
            if use_cbt:
                confidence = self._cbt_engine.get_activation_confidence(emotion, user_text)
                print(f"  [CBT检测] 检测到认知扭曲特征，置信度={confidence:.2f}，启用CBT模式")
        
        if use_cbt:
            return self._cycle_mid_cbt_with_face(emotion, user_text)

        # 2. 放音乐
        print("\n  [中强度] 放音乐...")
        self._play_full_music(emotion, user_text)
        self.last_perform_type = "music"

        # 3. 询问1
        result = self._ask_comfort(emotion, timeout_sec=10)
        if result[0] == "timeout":
            print("\n  [中强度] 超时，跳舞...")
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"
        elif result[0] == "stop":
            return ("stop", "")
        elif result[0] == "emotion":
            return (get_emotion_level(result[1]), result[1], result[2])
        else:
            user_text = result[1]
            self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
            self.last_perform_type = "dance"

        # 4. 循环：安慰式询问 → 表演
        while True:
            result = self._ask_with_face_recognition(emotion, timeout_sec=10)
            if result[0] == "play_song":
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                choice = self._perform_choice(emotion)
                if choice == "music":
                    self._play_full_music(emotion, user_text)
                    self.last_perform_type = "music"
                elif choice == "dance":
                    self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
                    self.last_perform_type = "dance"
                elif choice == "pushup":
                    self._do_pushup(emotion)
                    self.last_perform_type = "pushup"
                else:
                    self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "stop":
                return ("stop", "")
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                print("\n  [中强度] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                user_text = result[1] if len(result) > 1 else user_text
                self._perform_arranged(emotion, user_text)

    # ===== CBT增强的中强度循环（人脸识别版）=====
    def _cycle_mid_cbt_with_face(self, emotion, user_text):
        """CBT增强的中强度循环（人脸识别版）
        CBT对话本身不需要人脸识别（是语音对话），所以直接复用_cycle_mid_cbt
        """
        return self._cycle_mid_cbt(emotion, user_text)

    def cycle_low_with_face(self, emotion, user_text):
        """★ 人脸识别版低强度(3-5)"""
        self._check_fall()

        # 1. 人脸识别 + 安慰
        self.round_num += 1

        # ★★★ 关键逻辑 ★★★
        # 情绪等检测服务：每次都是全新的变量服务（清零）
        # 称呼：只保留第一个检测到的
        if self._face_jump_just_triggered:
            self._face_jump_just_triggered = False  # 重置跳转标志
            self._face_detection_paused = True  # ★ 暂停人脸检测（后续询问也要跳过）
            face_result = None
            print(f"  [低强度-安慰] ★ 跳过人脸检测，使用保存的称呼: {self._current_address}")
            resp, mid, mdesc = generate_comfort_with_address(
                emotion, user_text, self._current_address, self.round_num, self.previous_responses
            )
            print(f"  [低强度-安慰-{self.round_num}] {resp}")
            self.previous_responses.append(resp)
            self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))
        else:
            face_result = None
            if self._use_face_recognition and self.face_manager:
                import threading
                def async_detect():
                    nonlocal face_result
                    face_result = self.face_manager.detect()
                detect_thread = threading.Thread(target=async_detect)
                detect_thread.start()
                try:
                    detect_thread.join(timeout=2.0)
                except NameError:
                    pass
            
            if face_result and face_result.get("success"):
                gender = face_result.get("gender")
                age = face_result.get("age")
                face_result["address"] = FaceRecognitionManager._generate_address(gender, age)
                face_result["voice"] = FaceRecognitionManager._select_voice(gender, age)
                address = face_result.get("address", "朋友")
                self._current_address = address
                self._current_voice = face_result.get("voice", "x4_yezi")
                print(f"  [人脸识别-安慰] 检测到: {gender}/{age}岁，称呼: {address}")
                addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                    emotion, user_text, address, self.round_num, self.previous_responses
                )
                print(f"  [低强度-安慰-{self.round_num}] {addressed_text}")
                self.previous_responses.append(addressed_text)
                self._comfort_with_motion(addressed_text, emotion, pre_selected_motion=(addr_mid, addr_mdesc))
            else:
                resp, mid, mdesc = generate_comfort_with_address(
                    emotion, user_text, self._current_address, self.round_num, self.previous_responses
                )
                print(f"  [低强度-安慰-{self.round_num}] {resp}")
                self.previous_responses.append(resp)
                self._comfort_with_motion(resp, emotion, pre_selected_motion=(mid, mdesc))

        # 2. 跳舞
        print("\n  [低强度] 跳舞...")
        self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
        self.last_perform_type = "dance"

        # 3. 循环：安慰式询问 → 表演
        while True:
            result = self._ask_with_face_recognition(emotion, timeout_sec=10)
            if result[0] == "play_song":
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "stop":
                return ("stop", "")
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                print("\n  [低强度] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                user_text = result[1] if len(result) > 1 else user_text
                self._perform_arranged(emotion, user_text)

    def cycle_positive_with_face(self, emotion, user_text):
        """★ 人脸识别版正面情绪(2-3)"""
        self._check_fall()

        # 1. 人脸识别 + 回应
        self.round_num += 1

        # ★★★ 关键逻辑 ★★★
        # 情绪等检测服务：每次都是全新的变量服务（清零）
        # 称呼：只保留第一个检测到的
        if self._face_jump_just_triggered:
            self._face_jump_just_triggered = False  # 重置跳转标志
            self._face_detection_paused = True  # ★ 暂停人脸检测（后续询问也要跳过）
            face_result = None
            print(f"  [正面-回应] ★ 跳过人脸检测，使用保存的称呼: {self._current_address}")
            addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                emotion, user_text, self._current_address, self.round_num, self.previous_responses
            )
            print(f"  [正面-回应-{self.round_num}] {addressed_text}")
            self.previous_responses.append(addressed_text)
            self._comfort_with_motion(addressed_text, emotion)
        else:
            face_result = None
            if self._use_face_recognition and self.face_manager:
                import threading
                def async_detect():
                    nonlocal face_result
                    face_result = self.face_manager.detect()
                detect_thread = threading.Thread(target=async_detect)
                detect_thread.start()
                try:
                    detect_thread.join(timeout=2.0)
                except NameError:
                    pass
            
            if face_result and face_result.get("success"):
                gender = face_result.get("gender")
                age = face_result.get("age")
                face_result["address"] = FaceRecognitionManager._generate_address(gender, age)
                face_result["voice"] = FaceRecognitionManager._select_voice(gender, age)
                address = face_result.get("address", "朋友")
                self._current_address = address
                self._current_voice = face_result.get("voice", "x4_yezi")
                print(f"  [人脸识别-回应] 检测到: {gender}/{age}岁，称呼: {address}")
                addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                    emotion, user_text, address, self.round_num, self.previous_responses
                )
                print(f"  [正面-回应-{self.round_num}] {addressed_text}")
                self.previous_responses.append(addressed_text)
                self._comfort_with_motion(addressed_text, emotion)
            else:
                addressed_text, addr_mid, addr_mdesc = generate_comfort_with_address(
                    emotion, user_text, self._current_address, self.round_num, self.previous_responses
                )
                print(f"  [正面-回应-{self.round_num}] {addressed_text}")
                self.previous_responses.append(addressed_text)
                self._comfort_with_motion(addressed_text, emotion)

        # 2. 表演
        print("\n  [正面] 表演...")
        self._perform_arranged(emotion, user_text)

        # 3. 循环：回应询问 → 超时表演
        while True:
            result = self._ask_response_question(emotion, timeout_sec=10)
            if result[0] == "stop":
                return ("stop", "")
            elif result[0] == "play_song":
                song_name = result[1]
                self._play_user_song(emotion, song_name)
                self.last_perform_type = "music"
                continue
            elif result[0] == "play_music":
                self._play_full_music(emotion, user_text)
                self.last_perform_type = "music"
                continue
            elif result[0] == "dance":
                self._do_dance_for_intent(emotion, intent=result, exclude_motion=self.last_dance_motion)
                self.last_perform_type = "dance"
                continue
            elif result[0] == "pushup":
                self._do_pushup(emotion)
                self.last_perform_type = "pushup"
                continue
            elif result[0] == "demand":
                choice = self._perform_choice(emotion)
                if choice == "music":
                    self._play_full_music(emotion, user_text)
                    self.last_perform_type = "music"
                elif choice == "dance":
                    self._do_dance_full(emotion, exclude_motion=self.last_dance_motion)
                    self.last_perform_type = "dance"
                elif choice == "pushup":
                    self._do_pushup(emotion)
                    self.last_perform_type = "pushup"
                else:
                    self._perform_arranged(emotion, user_text)
                continue
            elif result[0] == "emotion":
                return (get_emotion_level(result[1]), result[1], result[2])
            elif result[0] == "timeout":
                print("\n  [正面] 超时，表演...")
                self._perform_arranged(emotion, user_text)
            else:
                self._perform_arranged(emotion, user_text)

    # ===== 辅助方法结束 =====

    # ===== 可中断等待（摔倒时自动处理并恢复）=====
    def _interruptible_sleep(self, seconds, check_interval=0.3):
        """可被摔倒信号中断的 sleep。
        检测到摔倒 → 自动调用 _handle_fall() → 处理完继续等待剩余时间。
        返回 True=正常睡完，False=被摔倒中断（已处理）。"""
        end = time.time() + seconds
        while time.time() < end:
            if self.fall_event.is_set() and not self._handling_fall:
                self._handle_fall()
                return False  # 告诉调用方：被中断了，可能需要跳过后续步骤
            time.sleep(check_interval)
        return True

    # ===== 摔倒检测 + 爬起 =====
    class FallWatcher(threading.Thread):
        """后台摔倒检测线程：双通道算法（即时姿态 + 趋势确认 + 舞蹈排除）
        混合模式：初始阶段用 v113 严格阈值检测躺/趴状态，站稳后切换到 v117 宽松阈值+舞蹈排除。
        """

        def __init__(self, bot):
            super().__init__(daemon=True)
            self.bot = bot
            self.fall_counter = 0
            self.running = True
            self.paused = False  # 暂停检测（俯卧撑等场景）
            self.direction = "rear"  # 默认后倒
            # 互补滤波状态（摔倒后方向判断 + 站立重置，不用于摔倒检测）
            self.cf_euler_x = 92.0  # 滤波后的 euler-x（初始值站立≈92°）
            self.cf_gyro_x = 0.0  # 上一帧 gyro-x（角速度）
            self.cf_last_time = None  # 上一帧时间戳
            self.cf_accel_angle = 92.0  # accel 角度（低通滤波后的值）
            self.cf_stand_frames = 0  # 站立稳定帧计数器（用于重置滤波器）
            # 双通道算法状态
            self._prev_accel_total = None  # 上一帧合加速度（用于冲击检测）
            self._short_gyro_buf = []  # 短窗口 gyro 数据（最近 FALL_DANCE_WIN_SIZE 帧）
            self._impact_detected = False  # 本帧是否检测到冲击
            self.euler_x_samples = []  # 确认帧 euler_x 缓冲（方向判断用）
            self.euler_x_samples_raw = []  # 确认帧 accel_x 缓冲（方向辅助判断）
            # 初始模式状态（v113 严格阈值检测躺/趴，站稳后切换 v117）
            self._init_mode = INIT_MODE_ENABLED  # True = 初始模式（v113），False = 正常模式（v117）
            self._init_stand_frames = 0  # 初始模式站立稳定帧计数器

        def run(self):
            while self.running:
                if self.paused:
                    time.sleep(0.5)
                    continue
                gyro = self.bot.robot.get_gyro()
                if gyro is None:
                    time.sleep(FALL_CHECK_INTERVAL)
                    continue
                is_fallen = self._check(gyro)
                if is_fallen:
                    direction_cn = {"front": "前倒", "rear": "后倒"}.get(self.direction, self.direction)
                    print(f"\n  ⚠️  [摔倒检测] 检测到【{direction_cn}】！")
                    if not self.bot._handling_fall:
                        self.bot.fall_event.set()  # 通知主线程
                        time.sleep(0.5)
                        # 等主线程处理完爬起后 clear
                        while self.bot.fall_event.is_set() and self.running:
                            time.sleep(0.3)
                    else:
                        # 正在处理摔倒，忽略本次检测，重置计数
                        self.fall_counter = 0
                        time.sleep(1.0)  # 等一会儿再检测
                time.sleep(FALL_CHECK_INTERVAL)

        def _check(self, gyro_data):
            """混合模式摔倒检测：
            - 初始模式（_init_mode=True）：使用 v113 严格阈值检测躺/趴状态
              → accel-y < 0.7（严格），|euler-x - 92| > 37°（严格），accel-x > 0.6（额外侧向）
            - 正常模式（_init_mode=False）：使用 v117 宽松阈值 + 舞蹈排除
              → accel-y < 0.5（宽松），|euler-x - 92| > 50°，有舞蹈排除机制
            - 模式切换：初始模式检测到站稳后，切换到正常模式
            返回 True = 确认摔倒，False = 未摔倒（或判断为舞蹈）
            """
            if gyro_data is None:
                return False

            accel_x = gyro_data.get("accel-x", 0)
            accel_y = gyro_data.get("accel-y", 0)
            accel_z = gyro_data.get("accel-z", 0)
            gyro_x = gyro_data.get("gyro-x", 0)
            accel_total = (accel_x ** 2 + accel_y ** 2 + accel_z ** 2) ** 0.5

            # ========== 互补滤波更新（用于方向判断 + 站立重置）=========
            # （与摔倒检测解耦，摔倒检测不依赖滤波结果）
            accel_magnitude = math.sqrt(accel_y ** 2 + accel_z ** 2)
            raw_accel_angle = math.degrees(math.atan2(accel_x, accel_magnitude)) if accel_magnitude > 0.01 else 0
            self.cf_accel_angle = CF_ACCEL_LPF * raw_accel_angle + (1 - CF_ACCEL_LPF) * self.cf_accel_angle
            now = time.time()
            dt = now - self.cf_last_time if self.cf_last_time is not None else FALL_CHECK_INTERVAL
            self.cf_last_time = now
            dt = max(dt, 0.001)
            gyro_angle = self.cf_euler_x + gyro_x * dt
            self.cf_euler_x = CF_GYRO_WEIGHT * gyro_angle + (1 - CF_GYRO_WEIGHT) * self.cf_accel_angle
            if CF_STAND_MIN <= self.cf_euler_x <= CF_STAND_MAX:
                self.cf_stand_frames += 1
                if self.cf_stand_frames >= CF_STAND_FRAMES:
                    self.cf_euler_x = self.cf_accel_angle
                    self.cf_stand_frames = CF_STAND_FRAMES
            else:
                self.cf_stand_frames = 0

            euler_x_raw = gyro_data.get("euler-x", 92.0)

            # ========== 初始模式：v113 严格阈值检测 ==========
            if self._init_mode:
                euler_deviation = abs(euler_x_raw - 92.0)
                is_abnormal = (
                        accel_y < INIT_ACCEL_Y_THRESHOLD or
                        accel_x > INIT_ACCEL_X_THRESHOLD or
                        accel_total < INIT_ACCEL_TOTAL_MIN or
                        euler_deviation > INIT_EULER_DEVIATION
                )

                # 站立检测：累积站立帧数
                is_standing = (
                        accel_y >= INIT_STAND_ACCEL_Y and
                        INIT_STAND_EULER_MIN <= euler_x_raw <= INIT_STAND_EULER_MAX
                )
                if is_standing:
                    self._init_stand_frames += 1
                    if self._init_stand_frames >= INIT_STAND_FRAMES:
                        self._init_mode = False
                        self._init_stand_frames = 0
                        print(f"[摔倒检测] 站稳确认，切换到正常模式（v117阈值+舞蹈排除）")
                else:
                    self._init_stand_frames = 0

                if is_abnormal:
                    self.fall_counter += 1
                    self.euler_x_samples.append(euler_x_raw)
                    self.euler_x_samples_raw.append(accel_x)
                    if self.fall_counter >= INIT_CONFIRM_FRAMES:
                        avg_euler = sum(self.euler_x_samples) / len(self.euler_x_samples)
                        avg_accel_x = sum(self.euler_x_samples_raw) / len(
                            self.euler_x_samples_raw) if self.euler_x_samples_raw else 0
                        print(
                            f"[摔倒检测-初始模式] 确认摔倒: avg_euler-x={avg_euler:.1f} deg, avg_accel-x={avg_accel_x:.2f}g（{len(self.euler_x_samples)}帧）")
                        if avg_euler < REAR_FALL_EULER_MIN:
                            self.direction = "rear"
                        elif avg_euler > REAR_FALL_EULER_POSITIVE:
                            self.direction = "rear"
                        elif avg_euler < FRONT_FALL_EULER_MAX:
                            self.direction = "front"
                        else:
                            self.direction = "rear"
                        self.euler_x_samples.clear()
                        self.euler_x_samples_raw.clear()
                        self._init_stand_frames = 0
                        return True
                else:
                    self.fall_counter = 0
                    self.euler_x_samples.clear()
                    self.euler_x_samples_raw.clear()
                return False

            # ========== 正常模式：v117 宽松阈值 + 舞蹈排除 ==========
            # ========== 快通道：即时姿态异常判断 ==========
            # 用传感器直接返回的 euler-x 计算偏移（站立时 euler-x ≈ 93°）
            # 注意：不能用 atan2 重新计算的角度，因为 atan2(accel_x, sqrt(ay²+az²)) 站立时≈0°
            euler_x_raw = gyro_data.get("euler-x", 92.0)
            euler_deviation = abs(euler_x_raw - 92.0)
            is_fast_abnormal = (
                    accel_y < FALL_ACCEL_Y_FAST or
                    accel_total < FALL_ACCEL_TOTAL_MIN or
                    euler_deviation > FALL_EULER_DEVIATION_FAST
            )

            # ========== 冲击检测 ==========
            if self._prev_accel_total is not None:
                accel_delta = abs(accel_total - self._prev_accel_total)
                if accel_delta > FALL_IMPACT_DELTA:
                    self._impact_detected = True
            self._prev_accel_total = accel_total

            # ========== 短窗口 gyro 缓冲（用于舞蹈排除）=========
            self._short_gyro_buf.append(abs(gyro_x))
            if len(self._short_gyro_buf) > FALL_DANCE_WIN_SIZE:
                self._short_gyro_buf.pop(0)
            if len(self._short_gyro_buf) >= 3:
                g_mean = sum(self._short_gyro_buf) / len(self._short_gyro_buf)
                g_var = sum((g - g_mean) ** 2 for g in self._short_gyro_buf) / len(self._short_gyro_buf)
            else:
                g_mean = 0.0
                g_var = 0.0

            # ========== 慢通道：趋势确认 ==========
            if is_fast_abnormal:
                self.fall_counter += 1
                self.euler_x_samples.append(self.cf_euler_x)
                self.euler_x_samples_raw.append(accel_x)
                if self.fall_counter >= FALL_CONFIRM_FRAMES:
                    # ---- 姿态严重异常检查 ----
                    # 如果当前帧姿态严重异常（euler偏移>70° 或 accel-y<0.2），
                    # 直接确认摔倒，跳过舞蹈排除（摔倒过程中 gyro 大值留在短窗口会干扰排除判断）
                    is_severe_abnormal = (
                            euler_deviation > FALL_SEVERE_EULER_DEVIATION or
                            accel_y < FALL_SEVERE_ACCEL_Y
                    )
                    if not is_severe_abnormal:
                        # ---- 舞蹈排除检查（仅姿态轻度异常时） ----
                        is_dancing = (
                                             g_var > FALL_DANCE_VAR_TH and g_mean > 5.0
                                     ) or (
                                             g_mean > FALL_DANCE_GYRO_MEAN_MAX and g_var > 40.0
                                     )
                        if is_dancing:
                            self.fall_counter = 0
                            self.euler_x_samples.clear()
                            self.euler_x_samples_raw.clear()
                            self._impact_detected = False
                            return False
                    # ---- 不是舞蹈，确认摔倒 ----
                    avg_euler = sum(self.euler_x_samples) / len(self.euler_x_samples)
                    avg_accel_x = sum(self.euler_x_samples_raw) / len(
                        self.euler_x_samples_raw) if self.euler_x_samples_raw else 0
                    print(
                        f"[摔倒检测-正常模式] 确认摔倒: avg_euler-x={avg_euler:.1f} deg(滤波), avg_accel-x={avg_accel_x:.2f}g（{len(self.euler_x_samples)}帧）")
                    if avg_euler < REAR_FALL_EULER_MIN:
                        self.direction = "front"
                    elif avg_euler > REAR_FALL_EULER_POSITIVE:
                        self.direction = "rear"
                    elif avg_euler < FRONT_FALL_EULER_MAX:
                        self.direction = "front"
                    else:
                        self.direction = "rear"
                    self.euler_x_samples.clear()
                    self.euler_x_samples_raw.clear()
                    self.cf_stand_frames = 0
                    self._impact_detected = False
                    return True
            else:
                self.fall_counter = max(0, self.fall_counter - 1)
                if self.fall_counter == 0:
                    self.euler_x_samples.clear()
                    self.euler_x_samples_raw.clear()
            return False

    def _handle_fall(self):
        """摔倒处理：停→说摔倒语→爬起→复位→恢复检测→说恢复语→恢复活动。由主线程调用。"""
        if self._handling_fall:
            return  # 防止递归
        self._handling_fall = True
        try:
            print(f"\n  [摔倒处理] 开始处理摔倒...")

            # 0. 保存摔倒前的活动状态（在停止之前保存）
            saved_activity = self.pre_fall_activity
            self.pre_fall_activity = None  # 清除，避免循环
            # v122: 记录摔倒时刻，用于恢复时计算实际播放时长
            if saved_activity:
                saved_activity["fall_time"] = time.time()

            # 1. 停止当前播放的音乐和动作
            self.robot.stop()
            self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)

            # 2. 获取方向
            if self.fall_watcher:
                direction = self.fall_watcher.direction
            direction_cn = {"front": "前倒", "rear": "后倒"}.get(direction, direction)
            print(f"  [摔倒处理] 方向: {direction_cn}")

            # 3. 先说摔倒感叹语（说完了再爬起），根据方向生成不同的话
            fall_msg = generate_fall_exclamation(direction)
            fall_emotion = self.current_emotion or None
            print(f"  [摔倒处理] 摔倒语: {fall_msg}")
            if fall_emotion:
                self.robot.speak(fall_msg, emotion=fall_emotion, speed=55, voice=self._current_voice)
            else:
                self.robot.speak(fall_msg, emotion="温柔", speed=55, voice=self._current_voice)

            # 4. 说完后爬起
            action_list = {
                "front": RECOVERY_FRONT,
                "rear": RECOVERY_REAR,
            }.get(direction, RECOVERY_REAR)

            # ★ v122 关键修复：暂停 FallWatcher + 清除 fall_event
            # 原因：爬起过程中机器人还在地上，FallWatcher 会持续检测到摔倒并 set fall_event，
            # 导致 wait_motion_done 被中断（即使 check_fall_event=False，也干扰其他逻辑）
            # 同时大量"检测到摔倒"日志刷屏，干扰调试
            if self.fall_watcher:
                self.fall_watcher.paused = True
                print(f"  [摔倒处理] ✓ FallWatcher 已暂停（防止爬起期间误判）")
            self.fall_event.clear()
            print(f"  [摔倒处理] ✓ fall_event 已清除")

            time.sleep(0.3)  # 确保说话完全结束、动作通道释放（TTS已含dur等待，0.3s足够）
            for i, (name, version) in enumerate(action_list[:RECOVERY_MAX_ATTEMPTS]):
                print(f"  [摔倒处理] 尝试 [{i + 1}]: {name} (speed=slow)")
                result = self.robot.req("PUT", "/motions", {
                    "operation": "start", "version": version,
                    "motion": {"name": name, "repeat": 1, "speed": "slow"}
                }, timeout=10)
                if result.get("code") != 0:
                    print(f"  [摔倒处理] ⚠️ 动作播放失败: {result}")
                    continue

                # 等待爬起动作完成：用自适应退避轮询 + 最小运行时间保护 + 传感器确认
                # v122: min_run_time 自动查表 + sensor_verify_stand 传感器二次验证
                direction_cn = {"front": "前倒", "rear": "后倒"}.get(direction, direction)
                # 从 MOTION_TIME_TABLE 获取动作时长
                table_entry = MOTION_TIME_TABLE.get(name, MOTION_TIME_DEFAULT)
                base_timeout = table_entry[0] + 4  # 预估时长 + 4s 余量
                print(f"  [摔倒处理] 等待{direction_cn}爬起动作完成（最多{base_timeout:.0f}s，传感器确认站起）...")
                got_up = self.robot.wait_motion_done(
                    timeout=base_timeout, motion_name=name,
                    sensor_verify_stand=True, sensor_verify_retries=3,
                    _bot=self, check_fall_event=False
                    # v122: 传 _bot 让传感器验证生效，但 check_fall_event=False
                    # 防止 FallWatcher 在爬起期间反复 set fall_event 导致立即返回
                )
                if not got_up:
                    # 超时再补查一次
                    time.sleep(1.0)  # 补查缓冲（从2s缩短，wait_motion_done已做自适应轮询）
                    try:
                        r = self.robot.req("GET", "/motions", timeout=2)
                        state = r.get("data", {}).get("state")
                        state_str = str(state).strip().lower() if state is not None else ""
                        if state_str in ("0", "idle", ""):
                            print(f"  [摔倒处理] ✓ 补查确认{direction_cn}爬起完成")
                            got_up = True
                        else:
                            print(f"  [摔倒处理] ⚠️ {direction_cn}爬起可能未完全完成(state={state})，继续")
                    except Exception:
                        pass
                if got_up:
                    print(f"  [摔倒处理] ✓ {direction_cn}爬起动作确认完成")
                else:
                    print(f"  [摔倒处理] ⚠️ {direction_cn}爬起超时，强制继续下一步")
                break  # 只尝试第一个动作，完成后直接进入下一步

            # 爬起后复位到标准站立姿势
            self.robot.reset_pose()
            time.sleep(0.5)  # 复位后缓冲（reset_pose已发舵机指令，500ms足够到位）

            # 恢复摔倒检测（重置计数器 + 清除标志，让恢复活动期间可检测二次摔倒）
            if self.fall_watcher:
                self.fall_watcher.fall_counter = 0
                # v122: 全面重置 FallWatcher 状态（爬起后姿态变化大，旧数据会误判）
                self.fall_watcher.euler_x_samples.clear()
                self.fall_watcher.euler_x_samples_raw.clear()
                self.fall_watcher._short_gyro_buf.clear()
                self.fall_watcher._impact_detected = False
                self.fall_watcher._prev_accel_total = None
                # 重置互补滤波器（爬起后 euler 偏移大，不重置会误判）
                self.fall_watcher.cf_euler_x = 92.0
                self.fall_watcher.cf_accel_angle = 92.0
                self.fall_watcher.cf_stand_frames = 0
                self.fall_watcher.paused = False  # v122: 恢复 FallWatcher 检测
                print(f"  [摔倒处理] ✓ FallWatcher 全面重置并恢复检测")
            self._handling_fall = False  # 提前清除，使后续 _interruptible_sleep 能检测二次摔倒
            self.fall_event.clear()  # 清除信号，让 FallWatcher 恢复检测

            # 确认站起来后说恢复语（DeepSeek 生成）
            act_type = saved_activity.get("type") if saved_activity else None
            recovery_msg = generate_standup_recovery(activity_type=act_type)
            recovery_emotion = self.current_emotion or None
            print(f"  [摔倒处理] 恢复语: {recovery_msg}")
            if recovery_emotion:
                self.robot.speak(recovery_msg, emotion=recovery_emotion, speed=55, voice=self._current_voice)
            else:
                self.robot.speak(recovery_msg, emotion="温柔", speed=55, voice=self._current_voice)
            time.sleep(0.2)  # 恢复语后极短缓冲（speak已等TTS播完）

            # 8. 恢复摔倒前的活动（活动内部的 _interruptible_sleep 可检测二次摔倒）
            if saved_activity:
                self._resume_activity(saved_activity)

            print(f"  [摔倒处理] ✅ 恢复完成，继续陪伴流程\n")
        finally:
            self._handling_fall = False

    def _resume_activity(self, activity):
        """恢复摔倒前的活动：跳舞/音乐/安慰/打招呼/告知，全部支持再摔倒检测"""
        act_type = activity.get("type")
        emotion = activity.get("emotion", "平静")

        if act_type == "dance":
            # v123修复：摔倒后恢复跳舞，换一支不同的舞，配上匹配的告知语
            # 始终排除摔倒前跳的那支舞，选一支新的
            saved_motion = activity.get("motion")
            exclude = saved_motion or self.last_dance_motion
            candidates = [m for m in DANCE_MOTION_POOL if m != exclude]
            dance_motion = random.choice(candidates) if candidates else random.choice(DANCE_MOTION_POOL)
            dance_cn = MOTION_CN_MAP.get(dance_motion, dance_motion)
            self.last_dance_motion = dance_motion  # 更新记录
            print(f"  [活动恢复] 恢复跳舞（换新舞）: {dance_motion}（{dance_cn}），排除: {exclude}")

            # ====== 1. 站起后等待机器人稳定（防止立即跳舞导致再次摔倒）======
            print(f"  [活动恢复] ⏳ 等待机器人稳定（2秒）...")
            time.sleep(2.0)

            # ====== 2. 告知（仅日志标签，不再TTS说话，恢复语已说过"继续跳舞"）======
            intro = generate_dance_intro(emotion, dance_name=dance_cn)
            print(f"  [活动恢复告知] {intro}")

            # ====== 3. 保存活动状态 ======
            self.pre_fall_activity = {"type": "dance", "motion": dance_motion, "emotion": emotion}

            # ====== 4. 开始执行舞蹈动作（使用 start_motion）======
            print(f"  [活动恢复] 开始执行: {dance_motion}（{dance_cn}）...")
            result = self.robot.req("PUT", "/motions", {
                "operation": "start", "version": "v1",
                "motion": {"name": dance_motion, "repeat": 1, "speed": "normal"}
            }, timeout=10)
            if result.get("code") != 0:
                print(f"  [活动恢复] ⚠️ 动作启动失败: {result}")
                self.pre_fall_activity = None
                return
            print(f"  [活动恢复] ✓ 动作已启动，等待完成...")

            # ====== 5. 等待动作完成（v123: 使用 _wait_dance_api_done 纯API轮询）======
            done = self.robot._wait_dance_api_done(_bot=self, timeout=DANCE_TIMEOUT)
            if not done:
                # v123修复：摔倒中断时立即return，不执行后续复位逻辑
                if self.fall_event.is_set():
                    print(f"  [活动恢复] ⚠️ 跳舞期间再次摔倒，同步处理摔倒")
                    self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)
                    # v112模式：同步处理摔倒
                    self._handle_fall()
                    return
                else:
                    print(f"  [活动恢复] ⚠️ 超时，强制停止舞蹈")
                    self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)
            else:
                print(f"  [活动恢复] ✓ 舞蹈执行完毕")

            # ====== 6. 动作完成后等待恢复期（关键！让机器人稳定）======
            print(f"  [活动恢复] ⏳ 等待恢复期（2秒）...")
            time.sleep(2.0)

            # ====== 7. 确保停止所有动作 ======
            print(f"  [活动恢复] 🛑 停止动作...")
            self.robot.req("PUT", "/motions", {"operation": "stop"}, timeout=5)
            time.sleep(0.5)  # 停止后短暂等待确保生效

            # ====== 8. 复位→鞠躬→起立 ======
            print(f"  [活动恢复] 复位→鞠躬→起立...")
            self.robot.reset_pose()
            time.sleep(0.3)  # 复位后短暂等待
            self.robot.bow()
            time.sleep(0.3)  # 鞠躬后短暂等待
            self.robot.stand_up(get_emotion_level(emotion))

            # ====== 8. 清理状态 ======
            self.pre_fall_activity = None
            print(f"  [活动恢复] ✓ 跳舞恢复完成")

        elif act_type == "music":
            # 音乐恢复：摔倒后从断点继续；首次播放时从头开始
            song_path = activity.get("song_path")
            song_kw = activity.get("song_kw", "")
            song_dur = activity.get("song_dur", 0)
            song_name = activity.get("song_name", "")
            play_start = activity.get("play_start", 0)
            emotion = activity.get("emotion", "平静")
            is_first_play = activity.get("is_first_play", False)  # 是否首次播放
            # v123: 保留原始文件信息，解决第二次摔倒时截取文件已被删除的问题
            orig_song_path = activity.get("orig_song_path") or song_path
            orig_song_dur = activity.get("orig_song_dur") or song_dur

            # 站起后等待稳定（防止立即播放导致再次摔倒）
            print(f"  [活动恢复] ⏳ 等待机器人稳定（2秒）...")
            time.sleep(2.0)

            recovered = False
            trunc_path = None  # 初始化，避免 UnboundLocalError

            # 只有摔倒恢复（非首次播放）才从断点继续
            # v123: 使用 orig_song_path（原始文件）计算断点，避免截取文件已被删除
            if not is_first_play and orig_song_path and os.path.exists(orig_song_path):
                # 计算实际已播放时长（扣除摔倒处理时间）
                # v122 修复：用 fall_time 精确计算播放到了哪里
                fall_time = activity.get("fall_time", 0)
                if fall_time > 0 and play_start > 0:
                    # 播放到摔倒时刻的实际时长
                    elapsed = fall_time - play_start
                else:
                    elapsed = time.time() - play_start if play_start > 0 else 0
                # trunc_start = 实际播放位置（基于原始文件的绝对位置）
                # v123: 使用 orig_song_dur（原始总时长）计算，确保正确
                trunc_start = max(0, min(elapsed, orig_song_dur - 10))
                remaining = orig_song_dur - trunc_start

                if trunc_start > 5:
                    print(f"  [活动恢复] 恢复音乐: {song_name}（从{trunc_start:.0f}s处继续，剩余{remaining:.0f}s）")
                # v123: 从原始文件截取，不是从上次的截取文件
                trunc_path = truncate_audio(orig_song_path, trunc_start)
                if trunc_path and os.path.exists(trunc_path):
                    trunc_fname = os.path.basename(trunc_path)
                    trunc_size = os.path.getsize(trunc_path)

                    # === 步骤1: 彻底停止当前所有播放 ===
                    self.robot.stop()
                    time.sleep(0.3)

                    # === 步骤2: 上传断点文件（唯一文件名）===
                    up_result = self.robot.upload(trunc_path)
                    up_ok = isinstance(up_result, dict) and up_result.get("code") == 0
                    print(
                        f"  [活动恢复] 上传: {trunc_fname}({trunc_size // 1024}KB) → code={up_result.get('code', '?') if isinstance(up_result, dict) else up_result}")

                    if not up_ok:
                        print(f"  [活动恢复] ⚠️ 上传失败，尝试重新上传...")
                        time.sleep(0.5)
                        up_result = self.robot.upload(trunc_path)
                        up_ok = isinstance(up_result, dict) and up_result.get("code") == 0
                        print(f"  [活动恢复] 重传结果: {up_result}")

                    # === 步骤3: 等待机器人端写入 + 播放前再次stop确保清空旧状态 ===
                    time.sleep(0.8)  # 等待机器人写入（原1.5s偏长）
                    self.robot.stop()  # 再次stop，清除可能的残留播放状态
                    time.sleep(0.2)

                    # === 步骤4: 设置音量并播放（带重试）===
                    self.robot.set_volume(82)
                    play_ok = self.robot.play(trunc_fname)
                    if not play_ok:
                        time.sleep(0.4)
                        play_ok = self.robot.play(trunc_fname)
                        print(f"  [活动恢复] 播放重试1: {'✅' if play_ok else '✗'}")
                    if not play_ok:
                        time.sleep(0.6)
                        play_ok = self.robot.play(trunc_fname)
                        print(f"  [活动恢复] 播放重试2: {'✅' if play_ok else '✗'}")

                    if play_ok:
                        print(f"  [活动恢复] ✅ 从断点恢复成功: {song_name}（{trunc_start:.0f}s起, 文件={trunc_fname}）")
                        # 恢复音乐：先播放音乐，再执行动作配合
                        actual_wait = remaining + EXTRA
                        # v123: 保存原始文件信息，确保第二次摔倒时能从原始文件重新截取
                        self.pre_fall_activity = {"type": "music", "song_kw": song_kw, "song_path": trunc_path,
                                                  "song_dur": remaining, "song_name": song_name, "emotion": emotion,
                                                  "play_start": time.time(), "is_first_play": False,
                                                  "orig_song_path": orig_song_path, "orig_song_dur": orig_song_dur}
                        print(f"  [音乐] 继续播放（{trunc_start:.0f}s起），等待{actual_wait:.1f}s...")
                        # 先播放音乐，等0.2秒后再触发动作（避免动作覆盖音乐）
                        time.sleep(0.2)
                        self.robot.play_motion("ActionAging", wait=False, scene="唱歌")
                        print(f"  [动作] 执行: ActionAging（老态蹒跚），配合音乐...")
                        if not self._interruptible_sleep(actual_wait):
                            self.robot.stop()
                            self.pre_fall_activity = None
                            # v122: 清理截取文件
                            if trunc_path and os.path.exists(trunc_path):
                                try:
                                    os.remove(trunc_path)
                                    print(f"  [清理] 已删除截取文件: {os.path.basename(trunc_path)}")
                                except Exception:
                                    pass
                            return
                        self.robot.stop()
                        self.robot.reset_pose()
                        self.robot.bow()
                        # v122: 播放结束后清理截取文件
                        if trunc_path and os.path.exists(trunc_path):
                            try:
                                os.remove(trunc_path)
                                print(f"  [清理] 已删除截取文件: {os.path.basename(trunc_path)}")
                            except Exception:
                                pass
                        recovered = True
                    else:
                        print(f"  [活动恢复] ⚠️ 断点文件播放全部失败(code={up_result})，降级从头播放")
                else:
                    print(f"  [活动恢复] ⚠️ 截取失败或文件不存在，从头播放")

            if not recovered:
                # 兜底：从头重新播放同一首歌
                actual_wait = song_dur + EXTRA
                print(f"  [活动恢复] 恢复音乐: {song_name}（从头播放，等待{actual_wait:.1f}s）")
                if song_path and os.path.exists(song_path):
                    fname = os.path.basename(song_path)
                    self.robot.set_volume(82)
                    play_ok = self.robot.play(fname)
                    if play_ok:
                        print(f"  [活动恢复] ✅ 音乐恢复成功: {song_name}")
                        # 恢复音乐：先播放音乐，再执行动作配合
                        self.pre_fall_activity = {"type": "music", "song_kw": song_kw, "song_path": song_path,
                                                  "song_dur": song_dur, "song_name": song_name, "emotion": emotion,
                                                  "play_start": time.time(), "is_first_play": False,
                                                  "orig_song_path": orig_song_path, "orig_song_dur": orig_song_dur}
                        print(f"  [音乐] 从头播放，等待{actual_wait:.1f}s...")
                        # 先播放音乐，等0.2秒后再触发动作（避免动作覆盖音乐）
                        time.sleep(0.2)
                        self.robot.play_motion("ActionAging", wait=False, scene="唱歌")
                        print(f"  [动作] 执行: ActionAging（老态蹒跚），配合音乐...")
                        if not self._interruptible_sleep(actual_wait):
                            self.robot.stop()
                            self.pre_fall_activity = None
                            return
                        self.robot.stop()
                        self.robot.reset_pose()
                        self.robot.bow()
                        recovered = True

            if not recovered:
                # 最后兜底：重新下载并播放
                print(f"  [活动恢复] 重新搜索播放（关键词: {song_kw}）")
                kws = recommend_music(emotion, "", self.played_songs)
                kw = kws[0] if kws else song_kw
                path, dur, label = download(kw)
                if path:
                    fname = os.path.basename(path)
                    self.robot.upload(path)
                    time.sleep(0.8)  # 重新下载后上传就绪等待
                    self.robot.set_volume(82)
                    self.robot.play(fname)
                    # 播放新音乐：先播放音乐，再执行动作配合
                    actual_wait = dur + EXTRA
                    self.pre_fall_activity = {"type": "music", "song_kw": kw, "song_path": path,
                                              "song_dur": dur, "song_name": label, "emotion": emotion,
                                              "play_start": time.time()}
                    print(f"  [音乐] 重新播放: {label}，等待{actual_wait:.1f}s...")
                    # 先播放音乐，等0.2秒后再触发动作（避免动作覆盖音乐）
                    time.sleep(0.2)
                    self.robot.play_motion("ActionAging", wait=False, scene="唱歌")
                    print(f"  [动作] 执行: ActionAging（老态蹒跚），配合音乐...")
                    if not self._interruptible_sleep(actual_wait):
                        self.robot.stop()
                        self.pre_fall_activity = None
                        return
                    self.robot.stop()
                    self.robot.reset_pose()
                    self.robot.bow()
                else:
                    print(f"  [活动恢复] ⚠️ 音乐恢复均失败，跳过")

            # 清理截取的临时文件
            if trunc_path and os.path.exists(trunc_path) and trunc_path != song_path:
                try:
                    os.remove(trunc_path)
                except Exception:
                    pass
            self.pre_fall_activity = None

        elif act_type in ("comfort", "greet", "announce"):
            # 安慰/打招呼/告知：重新说类似意思但不同的话（同一个动作）
            original_text = activity.get("text", "")
            motion_id = activity.get("motion_id", "")
            type_cn = {"comfort": "安慰", "greet": "打招呼", "announce": "告知"}.get(act_type, act_type)

            # 生成新的话（意思相近但措辞不同）
            if act_type == "comfort":
                new_text = generate_repeat_comfort(original_text, emotion)
            elif act_type == "greet":
                new_text = generate_repeat_greeting(emotion)
            else:
                new_text = generate_repeat_comfort(original_text, emotion)

            # 如果生成失败，使用默认恢复语
            if not new_text:
                new_text = "站稳了，没事啦～"

            print(f"  [活动恢复] 恢复{type_cn}: {original_text[:30]}...")
            print(f"  [活动恢复] 重新生成话语: {new_text}")

            if motion_id:
                cn = MOTION_CN_MAP.get(motion_id, motion_id)
                print(f"  [活动恢复] 重新执行动作: {motion_id}（{cn}）")
                self.pre_fall_activity = activity  # 保持原状态
                self.robot.play_motion(motion_id, scene="讲话")
            self.robot.speak(new_text, emotion or "", speed=55, voice=self._current_voice, _bot=self)
            if self.fall_event.is_set():
                return  # speak内部已处理摔倒
            wait_sec = self._motion_tail_wait(new_text, base=0.2)  # 与正常流程统一的智能等待
            self._interruptible_sleep(wait_sec)
            self.robot.reset_pose()
            self.pre_fall_activity = None

        else:
            print(f"  [活动恢复] 未知活动类型: {act_type}，跳过恢复")

    def _check_fall(self):
        """检查是否摔倒了，如果摔倒了就处理。在关键等待点调用。"""
        if self.fall_event.is_set():
            self._handle_fall()
            return True
        return False

    def _listen_and_detect(self, emotion):
        """统一监听逻辑：等待用户说话 → 检测意图 → 返回结果
        v125: 使用智能意图检测（关键词优先 → DeepSeek分析）
        返回值：
          ("continue", emotion, user_text)  — 情绪未变或未识别，用原情绪继续
          ("switch", new_emotion, user_text) — 检测到不同情绪，需要跳转
          ("stop", emotion, "")              — 正面情绪检测到结束词
          ("timeout", emotion, "")           — 超时未说话
          表演意图 → 直接return，由调用方处理具体表演
        """
        print("\n[等待用户说话...]")
        reply = self.wait_speech(timeout=LISTEN_TIMEOUT)

        # ★ 人脸情绪优先：检测到人脸情绪 → 立即触发情绪跳转
        if isinstance(reply, tuple) and reply[0] == "_face_emotion_":
            face_emotion = reply[1]
            face_result = reply[2]
            address = face_result.get("address", self._current_address)
            voice = face_result.get("voice", self._current_voice)
            print(f"[用户说] ('_face_emotion_', '{face_emotion}', ...) - 人脸情绪立即跳转")
            print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}")
            
            # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
            reply = ("_face_emotion_", face_emotion, {"address": address})
            
            self._current_address = address
            self._current_voice = voice
            self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
            if self.face_manager:
                self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
            return ("switch", face_emotion, "")

        if not reply:
            # 超时
            timeout_resp = generate_timeout_response(emotion)
            print(f"[超时] {timeout_resp}")
            self._comfort_with_motion(timeout_resp, emotion)
            return ("timeout", emotion, "")

        print(f"[用户说] {reply}")

        # ★ v125: 使用智能意图检测（关键词优先 → DeepSeek分析）
        intent = detect_intent(reply, current_emotion=emotion)
        print(f"  [意图检测结果] {intent}")
        
        # ★ 优先检测表演意图（跳舞/放歌/俯卧撑等）
        if intent[0] in ["dance", "play_music", "play_song", "pushup", "demand"]:
            # 返回特殊标记，让调用方处理具体表演
            # 使用 ("performance", intent_type, reply) 格式
            return ("performance", intent[0], reply)
        
        # 检测到情绪切换
        if intent[0] == "emotion":
            new_e = intent[1]
            new_level = get_emotion_level(new_e)
            print(f"  -> 情绪切换:【{new_e}】 等级:【{new_level}】")
            return ("switch", new_e, reply)
        
        # 检测结束词（不再限制只有正面情绪才能结束）
        if intent[0] == "stop":
            print("[结束] 用户说了结束词")
            return ("stop", emotion, "")
        
        # continue 或其他
        return ("continue", emotion, reply)

    # ===== 主运行 =====
    def run(self):
        print("=" * 60)
        print("  Yanshee 情绪陪伴机器人 v124")
        print("  AgentMemory跨会话记忆 + CoT推理链 + 多轮上下文管理")
        print(f"  双通道摔倒检测 | HTS动作系统 | {len(ALL_EMOTIONS)}种情绪 | 5级强度循环")
        print("=" * 60)

        print(f"\n[连接] 正在连接 {ROBOT_IP}:{ROBOT_PORT} ...")
        if not self.robot.check_connection():
            print("[错误] 无法连接机器人！")
            return
        print("[连接] 成功！")

        # 启动起立（固定用calibration校准姿势）
        print("\n[启动] 起立准备...")
        print("  [动作] 校准起立（calibration）")
        self.robot.req("PUT", "/motions", {
            "operation": "start", "version": "v1",
            "motion": {"name": "calibration", "repeat": 1, "speed": "normal"}
        }, timeout=10)
        # 不等待，立即返回，不卡住

        # ★ 启动摔倒检测后台线程（最先启动，在任何动作之前）
        print("\n[摔倒检测] 启动后台检测...")
        self.fall_watcher = self.FallWatcher(self)
        self.fall_watcher.start()
        print("[摔倒检测] ✓ 已启动")

        # ★ 人脸识别：开机即启动摄像头
        print("\n[人脸识别] 启动摄像头...")
        self.face_manager = init_face_recognition(ROBOT_IP)
        self._use_face_recognition = self.face_manager.is_available()
        if self._use_face_recognition:
            print("[人脸识别] ✓ 摄像头已就绪")
        else:
            print("[人脸识别] ⚠ 摄像头不可用，将使用原有识别方式")

        # 启动打招呼动作：用H_Wave_B（v2双手交替挥手）（启动阶段不保存活动状态，只是起手式）
        print("\n[启动] 打招呼动作...")
        self.robot.play_motion("H_Wave_B", wait=False, scene="讲话")
        if not self._interruptible_sleep(0.8):  # 启动挥手动作完成等待（H_Wave_B约1-2s，但只需等它开始即可复位）
            # 摔倒了，_handle_fall已处理，跳过复位直接到问候语
            pass
        else:
            self.robot.reset_pose()
        self._interruptible_sleep(0.3)  # 复位后短暂缓冲再开始问候

        # 问候（先做动作→保持→讲话→复位）
        greeting = generate_greeting()
        print(f"[问候] {greeting}")
        self._greet_with_motion(greeting)

        # —— v124 AgentMemory：会话开始 ——
        try:
            _AGENT_MEMORY.on_session_start()
            _AGENT_MEMORY.save()
        except Exception as _e:
            print(f"[AgentMemory] on_session_start失败: {_e}")

        emotion = None
        user_text = ""
        try:
            for _ in range(50):  # 最多50轮
                # ===== 第一步：获取初始情绪/意图（仅首轮）=====
                if not emotion:
                    print("\n[等待用户说话...]")
                    t = self.wait_speech(timeout=LISTEN_TIMEOUT)
                    if not t:
                        retry = generate_retry_ask()
                        self._comfort_with_motion(retry, "平静")
                        continue
                    
                    # ★ v125: 处理人脸情绪元组（检测到人脸情绪时立即跳转）
                    if isinstance(t, tuple) and t[0] == "_face_emotion_":
                        face_emotion = t[1]
                        face_result = t[2]
                        address = face_result.get("address", self._current_address)
                        voice = face_result.get("voice", self._current_voice)
                        print(f"[用户说] ('_face_emotion_', '{face_emotion}', ...) - 人脸情绪立即跳转")
                        print(f"  [人脸情绪-跳转] 检测到情绪: {face_emotion}，称呼: {address}")
                        
                        # ★★★ 关键修复：只保留称呼，清空其他所有人脸数据 ★★★
                        t = ("_face_emotion_", face_emotion, {"address": address})
                        
                        self._current_address = address
                        self._current_voice = voice
                        self._face_jump_cooldown_until = time.time() + 10  # ★ 设置10秒冷却时间
                        self._face_jump_just_triggered = True  # ★ 标记刚触发过人脸情绪跳转
                        self._face_detection_paused = True  # ★ 暂停人脸检测（防止后续检测干扰）
                        self._face_detection_stop_event.set()  # ★ 立即停止正在运行的人脸检测线程
                        if self.face_manager:
                            self.face_manager.clear_cache()  # ★ 清除缓存避免重复触发
                        emotion = face_emotion
                        # ★ 设置情绪后继续循环，进入对应的 cycle 方法
                        self._current_emotion = emotion
                        continue
                    
                    print(f"[用户说] {t}")
                    # ★ v125: 使用智能意图检测
                    intent = detect_intent(t)
                    print(f"  [意图检测结果] {intent}")
                    
                    # 优先检测表演意图
                    if intent[0] in ["dance", "play_music", "play_song", "pushup", "demand"]:
                        # 表演意图 → 直接执行，然后继续监听
                        intent_type = intent[0]
                        print(f"  [表演意图] {intent_type}")
                        if intent_type == "dance":
                            self._do_dance_for_intent(emotion or "平静", intent=intent)
                        elif intent_type == "play_music":
                            self._play_full_music(emotion or "平静", t)
                        elif intent_type == "play_song":
                            self._play_user_song(emotion or "平静", intent[1] if len(intent) > 1 else t)
                        elif intent_type == "pushup":
                            self._do_pushup(emotion or "平静")
                        elif intent_type == "demand":
                            self._perform_choice_and_do(emotion or "平静")
                        # 表演后询问
                        post_greeting = generate_post_show_greeting(emotion or "平静")
                        print(f"\n [表演后询问] {post_greeting}")
                        self._comfort_with_motion(post_greeting, emotion or "平静")
                        # 表演完继续循环监听
                        continue
                    
                    # 如果是情绪
                    if intent[0] == "emotion":
                        emotion = intent[1]
                    elif intent[0] == "stop":
                        # 用户想结束对话
                        goodbye = generate_goodbye("平静")
                        print(f"\n[告别] {goodbye}")
                        self._goodbye_with_bow(goodbye, "平静")
                        break
                    elif intent[0] == "continue":
                        # 没识别出情绪，继续
                        self._comfort_with_motion("你是难过、焦虑，还是遇到什么烦心事了？", "平静")
                        continue
                    else:
                        # 其他情况（stop等）
                        continue
                    user_text = t
                    # —— 危机预警检测（检测到情绪后立即执行）——
                    try:
                        intensity = EMOTION_INTENSITY.get(emotion, 5)
                        crisis_result = self.crisis_detector.detect_crisis(
                            current_intensity=intensity,
                            current_emotion=emotion,
                            user_text=user_text
                        )
                        print(f"  [危机检测] 风险分数: {crisis_result['risk_score']}, "
                              f"等级: {crisis_result['level']}, "
                              f"建议: {crisis_result['recommendation'][:30]}...")
                        
                        # 更新危机状态
                        self._crisis_mode = crisis_result['is_crisis']
                        self._crisis_level = crisis_result['level']
                        
                        # 如果检测到危机，优先处理
                        if crisis_result['is_crisis']:
                            self._handle_crisis(emotion, crisis_result)
                    except Exception as _e:
                        print(f"  [危机检测] 失败: {_e}")
                    
                    # —— v124 多轮对话：记录首轮用户输入 ——
                    dialogue_observe_user_text(t, emotion)
                    level = get_emotion_level(emotion)
                    self.current_emotion = emotion
                    print(f"  -> 情绪:【{emotion}】 强度等级:【{level}】")
                    # —— v124 AgentMemory：记录本轮情绪 ——
                    try:
                        _AGENT_MEMORY.update_emotion(emotion, level)
                    except Exception as _e:
                        print(f"[AgentMemory] update_emotion失败: {_e}")

                level = get_emotion_level(emotion)

                # ===== 第二步：根据情绪强度执行对应cycle =====
                # 新的返回值约定：
                # - (level, emotion, user_text) → 需要跳转
                # - None → 无跳转，继续本cycle
                # - ("stop", "") → 正面情绪结束
                #
                # ★ 人脸识别开关：
                #   - self._use_face_recognition=True → 使用人脸识别版 Cycle
                #   - self._use_face_recognition=False → 使用原有 Cycle（不改变原有功能）
                result = None
                if level == "high":
                    if self._use_face_recognition:
                        result = self.cycle_high_with_face(emotion, user_text)
                    else:
                        result = self.cycle_high(emotion, user_text)
                elif level == "mid_high":
                    if self._use_face_recognition:
                        result = self.cycle_mid_high_with_face(emotion, user_text)
                    else:
                        result = self.cycle_mid_high(emotion, user_text)
                elif level == "mid":
                    if self._use_face_recognition:
                        result = self.cycle_mid_with_face(emotion, user_text)
                    else:
                        result = self.cycle_mid(emotion, user_text)
                elif level == "low":
                    if self._use_face_recognition:
                        result = self.cycle_low_with_face(emotion, user_text)
                    else:
                        result = self.cycle_low(emotion, user_text)
                elif level == "positive":
                    if self._use_face_recognition:
                        result = self.cycle_positive_with_face(emotion, user_text)
                    else:
                        result = self.cycle_positive(emotion, user_text)

                # 处理 cycle 返回值
                if result is not None:
                    if isinstance(result, tuple) and result[0] == "stop":
                        # 正面情绪结束
                        goodbye = generate_goodbye(emotion)
                        print(f"\n[告别] {goodbye}")
                        self._goodbye_with_bow(goodbye, emotion)
                        break
                    elif isinstance(result, tuple) and len(result) >= 3:
                        # 情绪跳转：(level, emotion, user_text)
                        new_level, emotion, user_text = result[0], result[1], result[2]
                        self.current_emotion = emotion
                        print(f"  -> 跳转到:【{emotion}】等级:【{new_level}】")
                        continue

                # ===== 第三步：递进式询问心情 → 统一监听 → 检测新情绪 → 跳转或继续 =====
                # 每轮结束后用DeepSeek生成不重复的递进式询问话术
                checkin = generate_checkin_question(emotion, self.round_num, self.previous_checkins, self._current_address)
                print(f"  [递进询问-第{self.round_num}轮] {checkin}")
                self.previous_checkins.append(checkin)
                self._comfort_with_motion(checkin, emotion)
                # —— v124 AgentMemory：更新对话深度 ——
                try:
                    _AGENT_MEMORY.update_depth(self.round_num)
                except Exception:
                    pass

                action, new_emotion, new_text = self._listen_and_detect(emotion)

                # —— v124 多轮对话：记录本轮用户输入 ——
                if new_text:
                    dialogue_observe_user_text(new_text, new_emotion or emotion)

                if action == "stop":
                    # 正面情绪检测到结束词
                    goodbye = generate_goodbye(emotion)
                    print(f"\n[告别] {goodbye}")
                    self._goodbye_with_bow(goodbye, emotion)
                    break
                elif action == "switch":
                    # 检测到不同情绪，更新后下一轮自动跳转对应cycle
                    emotion = new_emotion
                    user_text = new_text
                    self.current_emotion = emotion
                    level = get_emotion_level(emotion)
                    print(f"  -> 跳转到:【{emotion}】等级:【{level}】")
                    continue
                elif action == "timeout":
                    continue
                elif action == "performance":
                    # ★ v125: 表演意图（跳舞/放歌/俯卧撑等），执行对应表演
                    intent_type = new_emotion  # 这里 new_emotion 实际存的是 intent_type
                    user_reply = new_text
                    print(f"  [表演意图] {intent_type}")
                    
                    # 执行对应表演
                    if intent_type == "dance":
                        self._do_dance_for_intent(emotion, intent=intent)
                    elif intent_type == "play_music":
                        self._play_full_music(emotion, user_reply)
                    elif intent_type == "play_song":
                        self._play_user_song(emotion, user_reply)
                    elif intent_type == "pushup":
                        self._do_pushup(emotion)
                    elif intent_type == "demand":
                        self._perform_choice_and_do(emotion)
                    # 表演后询问
                    post_greeting = generate_post_show_greeting(emotion)
                    print(f"\n [表演后询问] {post_greeting}")
                    self._comfort_with_motion(post_greeting, emotion)
                    # 表演完后继续循环
                    continue
                else:
                    # "continue"：情绪未变或未识别，继续
                    if new_text:
                        user_text = new_text

        except KeyboardInterrupt:
            print("\n[中断]")
        finally:
            # —— v124 AgentMemory：会话结束保存 ——
            try:
                _AGENT_MEMORY.save()
            except Exception:
                pass
            self.robot.stop()
        print("\n[结束]")


