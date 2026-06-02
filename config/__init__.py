#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 机器人连接
ROBOT_IP = os.getenv("YANSHEE_IP", "192.168.3.241")
ROBOT_PORT = int(os.getenv("YANSHEE_PORT", "9090"))
BASE_URL = f"http://{ROBOT_IP}:{ROBOT_PORT}/v1"

# 视频流（MJPEG）SSH 自动部署
VIDEO_STREAM_PORT = int(os.getenv("YANSHEE_STREAM_PORT", "8080"))
VIDEO_STREAM_TIMEOUT = int(os.getenv("YANSHEE_STREAM_TIMEOUT", "15"))
YANSHEE_SSH_USER = os.getenv("YANSHEE_SSH_USER", "pi")
# 若 SSH 密钥未配置，请设置此密码（通过环境变量或直接修改此处）
# Windows: set YANSHEE_SSH_PASSWORD=你的密码 && python main.py
YANSHEE_SSH_PASSWORD = os.getenv("YANSHEE_SSH_PASSWORD", "raspberry")
YANSHEE_SSH_PORT = int(os.getenv("YANSHEE_SSH_PORT", "22"))
VIDEO_STREAM_REMOTE_DIR = os.getenv("VIDEO_STREAM_REMOTE_DIR", "/tmp")
# 设为 True 跳过 SSH 自动部署（机器人已有 MJPEG 服务时）
VIDEO_STREAM_SKIP_SSH = os.getenv("VIDEO_STREAM_SKIP_SSH", "").lower() in ("1", "true", "yes")

# ADB 自动开启 SSH（SSH 不可用时通过 ADB 启用）
YANSHEE_ADB_PORT = int(os.getenv("YANSHEE_ADB_PORT", "5555"))
# ADB 连接超时（秒）
ADB_CONNECT_TIMEOUT = int(os.getenv("ADB_CONNECT_TIMEOUT", "8"))
# ADB 启用 SSH 的命令序列（按顺序尝试）
ADB_SSH_START_CMDS = [
    "start sshd",                          # Android init 方式
    "svc ssh start",                       # 通过 svc
    "setprop service.adb.tcp.port 5555 && start adbd",  # 确保 adb TCP
]

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-6d9e890a3e00428eba82ce364beee57b")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "music_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 监听参数
LISTEN_TIMEOUT = 18
SILENCE_GAP = 2.0
EXTRA = 1.5
MUSIC_EXTRA = 2.0

# 摔倒检测 v2：快通道(即时姿态)+慢通道(趋势确认)+舞蹈排除(gyro方差)

# 快通道：即时姿态异常阈值
FALL_ACCEL_Y_FAST = 0.5          # 正常站立 ≈ 0.92~0.99
FALL_EULER_DEVIATION_FAST = 50
FALL_ACCEL_TOTAL_MIN = 0.5       # 自由落体

# 严重异常阈值（跳过舞蹈排除）
FALL_SEVERE_EULER_DEVIATION = 70  # 前倒后 euler≈-1°，偏移≈93°
FALL_SEVERE_ACCEL_Y = 0.2         # 倒地后 accel-y≈0

# 冲击检测
FALL_IMPACT_DELTA = 0.4           # 相邻帧 accel_total 突变

# 慢通道：趋势确认
FALL_CONFIRM_FRAMES = 3           # 3帧×0.2s=0.6s
FALL_CHECK_INTERVAL = 0.2

# 舞蹈排除：短窗口 gyro 方差
FALL_DANCE_WIN_SIZE = 5
FALL_DANCE_VAR_TH = 80.0          # >80 → 可能舞蹈；摔倒后迅速<80
FALL_DANCE_GYRO_MEAN_MAX = 12.0

# 互补滤波（用于方向判断，不用于摔倒检测）
CF_GYRO_WEIGHT = 0.97             # gyro积分权重，越大越相信gyro
CF_ACCEL_LPF = 0.15
CF_STAND_FRAMES = 6
CF_STAND_MIN = 55.0               # 站立 euler-x ≈ 92°
CF_STAND_MAX = 130.0

# 方向判断（基于滤波后 avg_euler_x）
FRONT_FALL_EULER_MAX = 50.0       # 0~50° 前倒
REAR_FALL_EULER_MIN = -55.0       # <-55° 后倒
REAR_FALL_EULER_POSITIVE = 140.0  # >120° 后倒(实测152~157°)

# 站立恢复检测
STAND_CONFIRM_FRAMES = 4
STAND_ACCEL_Y_MIN = 0.6           # 站立 ≈ 0.99
STAND_EULER_X_MIN = 75.0
STAND_EULER_X_MAX = 120.0

# 初始模式（v113严格阈值，检测躺/趴状态）
INIT_MODE_ENABLED = True
INIT_STAND_ACCEL_Y = 0.75
INIT_STAND_EULER_MIN = 55.0
INIT_STAND_EULER_MAX = 130.0
INIT_STAND_FRAMES = 3
INIT_ACCEL_Y_THRESHOLD = 0.7
INIT_ACCEL_X_THRESHOLD = 0.6
INIT_ACCEL_TOTAL_MIN = 0.5
INIT_EULER_DEVIATION = 37
INIT_CONFIRM_FRAMES = 3

# 舞蹈停止检测（gyro方差趋0 + 姿态恢复站立 → 完成）
DANCE_STOP_WIN_SIZE = 5
DANCE_STOP_GYRO_VAR_MAX = 50.0    # 舞蹈中通常>80
DANCE_STOP_GYRO_MEAN_MAX = 3.0    # 舞蹈中通常>8
DANCE_STOP_EULER_MIN = 70.0
DANCE_STOP_EULER_MAX = 130.0
DANCE_STOP_CONFIRM_FRAMES = 4
DANCE_STOP_CHECK_INTERVAL = 0.3
DANCE_STOP_TIMEOUT = 120

# 爬起动作
RECOVERY_FRONT = [("Base_GetupF", "v1"), ("GetupFront", "v1")]
RECOVERY_REAR = [("Base_GetupB", "v1"), ("GetupRear", "v1")]
RECOVERY_MAX_ATTEMPTS = 3
DANCE_TIMEOUT = 180

# 前倒爬起推腿辅助
FRONT_GETUP_ASSIST_ANGLES = {"LeftHipFB": 120, "RightHipFB": 60}
FRONT_GETUP_ASSIST_RUNTIME = 600
FRONT_GETUP_ASSIST_DELAY = 3.0

# 动作时间查表: (预估时长s, 最小运行保护s)
MOTION_TIME_TABLE = {
    # 爬起
    "Base_GetupF": (8.0, 5.0),
    "Base_GetupB": (7.0, 4.5),
    "GetupFront": (8.0, 5.0),
    "GetupRear": (7.0, 4.5),
    # 俯卧撑
    "PushUp": (34.0, 34.0),
    # 复位
    "Reset": (3.0, 1.5),
    "reset_without_head": (2.5, 1.2),
    "calibration": (3.0, 1.5),
    # 舞蹈
    "LittleApple": (45.0, 5.0),
    "GangnamStyle": (90.0, 5.0),
    "SorrySorry": (50.0, 5.0),
    "青春修炼手册": (55.0, 5.0),
    "HappyBirthday": (35.0, 5.0),
    "SweetAndSour": (40.0, 5.0),
    "WeAreTakingOff": (45.0, 5.0),
    "GetUp": (30.0, 5.0),
    "从这里出发": (35.0, 5.0),
    "SiegeOfTroy": (50.0, 5.0),
    "CombinationOfSongs": (60.0, 5.0),
}
MOTION_TIME_DEFAULT = (10.0, 3.0)

# 动作中文名
MOTION_CN_MAP = {
    "RaiseRightHand": "举手打招呼",
    "Victory": "胜利欢呼",
    "ActionAging": "老态蹒跚",
    "reset_without_head": "轻柔复位",
    "Reset": "标准复位",
    "calibration": "校准姿势",
    "Forward": "向前走",
    "Backward": "向后退",
    "TurnLeft": "左转",
    "TurnRight": "右转",
    "TurnLeft1": "左转变体",
    "TurnRight1": "右转变体",
    "Leftward": "向左移",
    "Bow": "鞠躬",
    # Layers v2
    "H_Wave_B": "双手交替挥手",
    "H_Wave_R": "右手挥手打招呼",
    "H_Wave_L": "左手挥手打招呼",
    "H_HWave_C": "双手高举过头顶挥动",
    "H_HWave_L": "左手高举挥动",
    "H_WaveRH": "右手高举挥动",
    "H_Rise_R": "右手举起",
    "H_Rise_L": "左手举起（抚胸）",
    "H_Rise_B": "双手缓缓举起（舒展）",
    "H_Bec_B": "身体前倾弯腰",
    "H_Bec_L": "身体向左倾斜",
    "H_Bec_R": "身体向右倾斜",
    "H_Str_R": "右手向前伸出",
    "H_Str_L": "左手向前伸出",
    "H_Str_B": "双手向前伸展（拥抱）",
    "Hd_Wacth_F": "头部前倾注视前方",
    "Hd_Wacth_L": "头部转向左侧",
    "Hd_Wacth_R": "头部转向右侧",
    "Hd_SwivelH": "头部左右摆动",
    "Reset_S": "全身复位",
    # 舞蹈
    "SiegeOfTroy": "特洛伊围攻舞",
    "LittleApple": "小苹果舞",
    "GangnamStyle": "江南Style",
    "SorrySorry": "Sorry Sorry舞",
    "青春修炼手册": "青春修炼手册舞",
    "HappyBirthday": "生日快乐舞",
    "SweetAndSour": "酸甜舞",
    "WeAreTakingOff": "起飞舞",
    "CombinationOfSongs": "歌曲串烧",
    "GetUp": "起立舞",
    "从这里出发": "从这里出发",
    # 表演
    "PushUp": "俯卧撑",
}

# 跳舞动作池（v1 HTS，自带音乐）
DANCE_MOTION_POOL = [
    "LittleApple",
    "GangnamStyle",
    "青春修炼手册", "HappyBirthday",
    "SweetAndSour", "WeAreTakingOff",
    "GetUp", "从这里出发",
]

# 安慰动作（原地姿势类，DeepSeek按情绪选择）
COMFORT_MOTION_OPTIONS = {
    # v1 HTS
    "RaiseRightHand": "举手——引起注意表示在关注你，温柔关怀",
    "Victory": "胜利欢呼——传递力量和鼓励，给人力量感",
    # v2 Layers
    "H_Wave_B": "双手交替挥手——像在温柔地招呼你靠近，传递陪伴感",
    "H_Wave_R": "右手挥手——轻柔地抬起右手，像朋友间的默契问候",
    "H_Wave_L": "左手挥手——温柔地挥动左手，像在安抚不安的情绪",
    "H_Bec_L": "身体向左倾斜——微微侧身倾听，像在认真感受你的心情",
    "H_Bec_R": "身体向右倾斜——侧身靠近，像在说'我听到了'",
    "H_Rise_B": "双手缓缓举起——像在为你撑开一片天空，给你安全感",
}

# 问候动作（DeepSeek选）
GREETING_MOTION_OPTIONS = {
    "RaiseRightHand": "举手打招呼——友好自然地抬起右手",
    "Victory": "胜利欢呼——带着活力和开心打招呼",
    "H_Wave_B": "双手交替挥手——热情地双手挥动，像看到你就很开心",
    "H_HWave_C": "双手高举过头顶挥动——像开心到飞起一样跟你打招呼",
    "H_HWave_L": "左手高举挥动——活泼俏皮地举起左手打招呼",
    "H_WaveRH": "右手高举挥动——潇洒地高举右手向你致意",
}

# 告知动作（预告表演时）
ANNOUNCE_MOTION_OPTIONS = {
    "RaiseRightHand": "举手——引起注意，像在说'接下来有惊喜！'",
    "Victory": "胜利欢呼——开心地预告接下来的表演",
    "H_Wave_B": "双手交替挥手——像在俏皮地预告'要表演啦！'",
    "H_Wave_R": "右手挥手——温柔地抬手，像朋友间的默契预告",
    "H_Bec_L": "身体向左倾斜——微微侧身，带点害羞地预告表演",
}

# 放音乐pose
MUSIC_POSE_OPTIONS = {
    "ActionAging": "老态蹒跚——缓慢摇摆配合音乐，有沉浸感",
}

# 情绪体系（30种，5级强度）
EMOTION_KEY_WORDS = {
    "绝望": ["绝望", "撑不下去", "没希望", "不想活", "活着没意思", "结束生命"],
    "崩溃": ["崩溃", "要疯了", "受够了", "快要炸了", "忍无可忍"],
    "心碎": ["心碎", "撕心裂肺", "痛不欲生", "心如刀割"],
    "痛苦": ["痛苦", "煎熬", "生不如死", "折磨"],
    "悲伤": ["悲伤", "失去", "哀伤", "悲痛", "忧伤", "难过", "伤心", "哭", "难受"],
    "失恋": ["失恋", "分手", "前任", "被抛弃", "被分手"],
    "愤怒": ["愤怒", "生气", "气死", "怒火", "暴怒", "火大"],
    "恐惧": ["恐惧", "害怕", "不敢", "恐怖", "吓死", "惊慌"],
    "委屈": ["委屈", "不公平", "被误解", "冤枉", "凭什么"],
    "无助": ["无助", "没人帮我", "孤立无援", "帮不了自己"],
    "焦虑": ["焦虑", "担心", "紧张", "不安", "心慌"],
    "孤独": ["孤独", "一个人", "没人理解", "寂寞", "形单影只"],
    "迷茫": ["迷茫", "不知道", "方向", "意义", "困惑", "看不清未来"],
    "挫败": ["挫败", "失败", "做不好", "一事无成", "不如意"],
    "自卑": ["自卑", "不如别人", "没用", "没自信", "低人一等"],
    "内耗": ["内耗", "纠结", "想太多", "自我怀疑", "反复纠结"],
    "疲惫": ["疲惫", "累", "精疲力尽", "身心俱疲", "透支"],
    "后悔": ["后悔", "如果当初", "遗憾", "不该这样"],
    "嫉妒": ["嫉妒", "羡慕", "眼红", "凭什么他有"],
    "思念": ["思念", "想你", "怀念", "舍不得", "放不下"],
    "压力大": ["压力大", "喘不过气", "快扛不住", "顶不住"],
    "失落": ["失落", "失望", "希望落空", "白忙一场"],
    "沮丧": ["沮丧", "灰心", "提不起劲", "没动力"],
    "心烦": ["心烦", "烦躁", "闹心", "静不下来", "心乱"],
    "无奈": ["无奈", "没办法", "只能这样", "认命"],
    "孤单": ["孤单", "独自一人", "没人陪", "冷清"],
    "麻木": ["麻木", "没感觉", "无所谓", "行尸走肉"],
    "开心": ["开心", "高兴", "快乐", "愉悦", "欢喜", "兴奋"],
    "平静": ["平静", "还好", "一般", "没什么", "淡定"],
    "满足": ["满足", "知足", "满意", "挺好", "不错"],
    "期待": ["期待", "希望", "盼望", "等不及"],
    "放松": ["放松", "舒服", "轻松", "解脱"],
}

ALL_EMOTIONS = list(EMOTION_KEY_WORDS.keys())

EMOTION_INTENSITY = {
    "绝望": 10, "崩溃": 10, "心碎": 9, "痛苦": 9,
    "悲伤": 8, "失恋": 8, "愤怒": 8, "恐惧": 7, "委屈": 7, "无助": 8,
    "焦虑": 7, "孤独": 6, "迷茫": 6, "挫败": 6, "自卑": 6,
    "内耗": 6, "疲惫": 5, "后悔": 5, "嫉妒": 5, "思念": 5,
    "压力大": 7, "失落": 6, "沮丧": 6,
    "心烦": 4, "无奈": 4, "孤单": 4, "麻木": 3,
    "开心": 2, "平静": 2, "满足": 2, "期待": 3, "放松": 2,
}

# 5级强度分类
EMOTION_LEVELS = {
    "high": {"绝望", "崩溃", "心碎", "痛苦"},
    "mid_high": {"悲伤", "失恋", "愤怒", "恐惧", "委屈", "无助"},
    "mid": {"焦虑", "孤独", "迷茫", "挫败", "自卑", "内耗", "疲惫", "后悔", "嫉妒",
            "思念", "压力大", "失落", "沮丧"},
    "low": {"心烦", "无奈", "孤单", "麻木"},
    "positive": {"开心", "平静", "满足", "期待", "放松"},
}


def get_emotion_level(emotion):
    """获取情绪强度等级"""
    for level, emotions in EMOTION_LEVELS.items():
        if emotion in emotions:
            return level
    return "mid"


POSITIVE_EMOTIONS = EMOTION_LEVELS["positive"]
STOP_WORDS = ["不用了", "不用", "谢谢", "感谢", "结束", "停止", "可以了", "没事了", "退出", "再见", "不用谢", "行了"]

# 情绪→音乐风格关键词
EMOTION_MUSIC_KEYWORDS = {
    "绝望": "希望、力量、温暖、坚持", "崩溃": "释放、宣泄、温柔、安抚",
    "心碎": "治愈、自愈、坚强、时间", "痛苦": "治愈、希望、温暖、陪伴",
    "悲伤": "治愈、温暖、希望、释怀", "失恋": "治愈、自愈、成长、放下",
    "愤怒": "平静、舒缓、释放、和解", "恐惧": "勇气、安全感、温暖、力量",
    "委屈": "共情、理解、温暖、治愈", "无助": "陪伴、温暖、力量、支持",
    "焦虑": "平静、放松、舒缓、安宁", "孤独": "陪伴、温暖、治愈、共情",
    "迷茫": "方向、希望、启发、力量", "挫败": "鼓励、力量、坚持、成长",
    "自卑": "自信、成长、接纳、力量", "内耗": "解脱、轻松、自我接纳、平静",
    "疲惫": "放松、舒缓、治愈、安宁", "后悔": "释怀、接纳、成长、希望",
    "嫉妒": "自我成长、接纳、平和", "思念": "温暖、回忆、治愈、陪伴",
    "压力大": "放松、舒缓、自由、解脱", "失落": "安慰、温暖、力量、重新出发",
    "沮丧": "鼓励、阳光、力量、希望", "心烦": "平静、舒缓、清心、放松",
    "无奈": "释怀、接纳、平静、随缘", "孤单": "陪伴、温暖、治愈",
    "麻木": "唤醒、感受、温柔、治愈", "开心": "快乐、轻快、阳光、幸福",
    "平静": "安宁、放松、舒适、惬意", "满足": "温暖、幸福、感恩、知足",
    "期待": "希望、阳光、憧憬、美好", "放松": "轻松、惬意、舒缓、舒适",
}

# 讯飞TTS
XUNFEI_APP_ID = "10a7c477"
XUNFEI_API_KEY = "fd2d7c87dcc73c631066015647d9e2a7"
XUNFEI_API_SECRET = "MGY4NWY1OGE3YmVhZDVmMTljNDg5MDFl"
USE_XUNFEI_TTS = True

VOICE_DESCRIPTIONS = {
    "x4_yezi": "椰子（温柔女声）", "x4_yifei": "艺菲（甜美女声）",
    "x4_lingfeizhe_emo": "聆飞哲-情感（同理心男声）", "x2_xiaoqian": "晓倩（东北话女声）",
    "x4_lingxiaoyun_talk_emo": "聆小芸-多情感（温暖女声）", "x4_xiaoxi": "晓希（轻快女声）",
    "x4_lingbosong": "聆伯松（可靠男声）", "x2_wanshu": "万叔（成熟男声）",
    "x4_lingfeichen_emo": "聆飞晨-情感（温暖男声）",
}

EMOTION_VOICE = {k: "x4_yezi" for k in [
    "绝望", "崩溃", "心碎", "痛苦", "悲伤", "失恋", "愤怒", "恐惧", "委屈", "无助",
    "焦虑", "孤独", "迷茫", "挫败", "自卑", "内耗", "疲惫", "后悔", "嫉妒", "思念",
    "压力大", "失落", "沮丧", "心烦", "无奈", "孤单", "麻木",
    "开心", "平静", "满足", "期待", "放松",
]}
DEFAULT_VOICE = "x4_yezi"


def get_voice(e):
    return EMOTION_VOICE.get(e, DEFAULT_VOICE)


# 音量：越脆弱→越低(68)；越激烈/正面→越高(85)
EMOTION_VOLUME = {
    "绝望": 68, "崩溃": 68, "心碎": 69, "痛苦": 70,
    "麻木": 70, "疲惫": 72,
    "悲伤": 73, "失恋": 73, "恐惧": 73, "委屈": 74,
    "无助": 73, "沮丧": 75, "自卑": 74,
    "焦虑": 77, "孤独": 77, "迷茫": 78, "挫败": 77,
    "内耗": 77, "后悔": 78, "失落": 77, "心烦": 78,
    "嫉妒": 79, "无奈": 80, "压力大": 79, "孤单": 79,
    "愤怒": 85, "思念": 83,
    "开心": 85, "平静": 82, "满足": 83, "期待": 85, "放松": 82,
}
COMFORT_SPEED = 50
COMFORT_PITCH = 50
COMFORT_VOLUME_TTS = 82

# CBT认知行为疗法配置
CBT_MIN_INTENSITY = 5              # 激活下限
CBT_MAX_INTENSITY = 8              # 高强度(9-10)不适合CBT，需要直接情感支持
CBT_MAX_ROUNDS = 5
CBT_MIN_CHALLENGE_ROUNDS = 2
CBT_MAX_CHALLENGE_ROUNDS = 4
CBT_CONFIDENCE_THRESHOLD = 0.4

CBT_LOOSENING_SIGNALS = [
    "也许", "可能", "其实", "好像", "不确定", "不一定",
    "也许吧", "说得也是", "你说的有道理", "没想过", "换个角度",
    "确实", "想想", "回头", "仔细想", "可能吧", "好像是",
]

CBT_SUITABLE_EMOTIONS = {
    "焦虑", "孤独", "迷茫", "挫败", "自卑", "内耗",
    "疲惫", "后悔", "嫉妒", "思念", "压力大", "失落", "沮丧",
    "悲伤", "失恋", "愤怒", "恐惧", "委屈", "无助",
}

# CBT动作池（倾听类，营造探索氛围）
CBT_MOTION_OPTIONS = {
    "H_Bec_L": "身体向左倾斜——微微侧身倾听",
    "H_Bec_R": "身体向右倾斜——侧身靠近",
    "H_Rise_B": "双手缓缓举起——给你安全感",
    "H_Wave_R": "右手挥手——朋友间的默契问候",
    "Hd_Wacth_F": "头部前倾注视——全神贯注",
    "Hd_Wacth_L": "头部转向左侧——像在思考",
}
