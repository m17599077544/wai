#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音乐工具：搜索、下载、MP3帧级截取、断点续播"""
import os, re, struct, math, requests, time, random
from config import CACHE_DIR

# 音乐工具
def get_duration(path):
    try:
        from mutagen.mp3 import MP3;
        return MP3(path).info.length
    except Exception:
        pass
    try:
        return os.path.getsize(path) / 16000.0
    except Exception:
        return 30.0

def search_music(keyword):
    """搜索音乐，返回多个候选歌曲用于下载重试"""
    try:
        r = requests.post("https://music.163.com/api/search/get/web",
                          data={"s": keyword, "type": 1, "limit": 8}, timeout=8)  # 增加搜索数量
        for item in r.json().get("result", {}).get("songs", [])[:8]:  # 返回更多候选
            name, singer, sid = item["name"], item["artists"][0]["name"], item["id"]
            dur = max(1, int(item.get("duration", 0) / 1000) + 1)
            yield name, singer, sid, dur
    except Exception:
        pass

def get_url(song_id):
    """获取歌曲下载URL，同时检测URL是否可用"""
    for url in [f"https://music.163.com/song/media/outer/url?id={song_id}.mp3",
                f"http://music.163.com/song/media/outer/url?id={song_id}.mp3"]:
        try:
            r = requests.head(url, timeout=5, allow_redirects=True)
            final_url = r.url
            if "126.net" in final_url and ".mp3" in final_url:
                return url
        except Exception:
            pass
    return None

# 预置可用歌曲ID列表（163版权限制下已确认可用的歌曲）
BACKUP_SONGS = [
    (1397619495, "年少有为", "李荣浩"),
    (444711250, "起风了", "买辣椒也用券"),
    (1822660086, "世界那么大", "韩红"),
    (1859279490, "你头顶的羽毛", "李荣浩"),
    (1467510376, "像我这样的人", "毛不易"),
    (1363946681, "消愁", "毛不易"),
    (1398165623, "像我这样的人", "DJ版"),
    (476373132, "小幸运", "田馥甄"),
    (436514124, "追光者", "岑宁儿"),
    (516955018, "起风了", "周深"),
    (1376469133, "知否知否", "郁可唯"),
    (1416865090, "探窗", "张碧晨"),
    (1331744515, "半壶纱", "刘珂矣"),
    (1357936543, "锦鲤", "Aki阿杰"),
    (1449934706, "归寻", "等什么君"),
]

def download(keyword):
    """下载音乐，支持多候选重试和预置备用歌曲"""
    for name, singer, sid, api_dur in search_music(keyword):
        # 只过滤伴奏，保留其他版本
        if "伴奏" in name:
            continue
        url = get_url(sid)
        if not url: continue
        # 强力清理文件名
        clean_name = re.sub(r'[\\/*?:"<>|()\[\]（）\s\-]', '', name)
        clean_singer = re.sub(r'[\\/*?:"<>|()\[\]（）\s\-]', '', singer)
        clean_name = clean_name[:20]
        clean_singer = clean_singer[:15]
        fname = f"{clean_name}_{clean_singer}.mp3"
        path = os.path.join(CACHE_DIR, fname)
        print(f"  [音乐] 下载: {name} - {singer}")
        try:
            r = requests.get(url, timeout=30, stream=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(16384):
                    if chunk: f.write(chunk)
            dur = api_dur if api_dur > 0 else get_duration(path)
            print(f"  [音乐] 完成，时长{dur}秒（文件: {fname}）")
            return path, dur, f"{name} {singer}"
        except Exception as e:
            print(f"  [音乐] 下载失败（{name}）: {e}，尝试下一首...")
            continue  # 继续尝试下一首歌

    # 备用方案：使用预置歌曲列表
    print(f"  [音乐] 搜索无可用歌曲，尝试预置列表...")
    random.shuffle(BACKUP_SONGS)  # 随机打乱顺序
    for sid, name, singer in BACKUP_SONGS[:5]:  # 尝试5首
        url = get_url(sid)
        if not url: continue
        clean_name = re.sub(r'[\\/*?:"<>|()\[\]（）\s\-]', '', name)
        clean_singer = re.sub(r'[\\/*?:"<>|()\[\]（）\s\-]', '', singer)
        fname = f"{clean_name}_{clean_singer}.mp3"
        path = os.path.join(CACHE_DIR, fname)
        print(f"  [音乐] 备用下载: {name} - {singer}")
        try:
            r = requests.get(url, timeout=20, stream=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(16384):
                    if chunk: f.write(chunk)
            dur = get_duration(path)
            print(f"  [音乐] 备用成功，时长{dur}秒（文件: {fname}）")
            return path, dur, f"{name} {singer}"
        except Exception as e:
            print(f"  [音乐] 备用失败: {e}")
            continue

    return None, 0, ""

# ===== 音频截取工具（纯 Python MP3 帧级断点续播，零外部依赖）=====
# MP3 文件结构（从文件头开始）：
#   [ID3v2 标签(可选)] [MP3帧1] [MP3帧2] ... [MP3帧N] [ID3v1 标签(可选)]
# 每个 MP3 帧：4字节帧头(同步词+参数) + 帧数据
#   - 帧时长 ≈ 0.026s (采样率44100/1152样本)
#   - 同步字: 0xFF 0xE0 (11位全1)
#   - 帧头第3字节 bit6-2 = MPEG Audio Version ID + Layer描述
#   - 帧头第3字节 bit1-0 + 第4字节 bit7-5 = 位率索引
#   - 帧头第4字节 bit4-2 = 采样率索引
# 算法：
#   1. 跳过 ID3v2 标签（如果有）
#   2. 解析第一帧头部，获取位率和采样率 → 计算单帧字节大小
#   3. 按目标秒数计算需要跳过的帧数和字节偏移
#   4. 从该偏移处开始扫描同步字，精确定位到帧边界
#   5. 直接拷贝剩余字节到新文件（含 ID3v1 尾标签）
#   速度：<0.01秒（纯字节拷贝，无需解码/重编码）

def _find_first_mp3_frame(data, start=0):
    """在 data 中从 start 位置开始查找第一个 MP3 帧同步字。
    返回 (帧偏移, 位率, 采样率, 帧大小) 或 (None, None, None, None)。
    MP3 帧同步字: 0xFF 后跟高3位为1的字节 (0xE0掩码)。"""
    limit = len(data) - 4
    pos = start
    while pos < limit:
        # 快速查找同步字 0xFF
        if data[pos] != 0xFF:
            pos += 1
            continue
        # 检查第二个字节的高3位是否为1
        if (data[pos + 1] & 0xE0) != 0xE0:
            pos += 1
            continue

        b2 = data[pos + 1]
        b3 = data[pos + 2]
        b4 = data[pos + 3]

        # MPEG Version: bits[19-18] of frame header (b2的bit4-3)
        version_bits = (b2 >> 3) & 0x03
        # Layer: bits[17-16] (b2的bit2-1)
        layer_bits = (b2 >> 1) & 0x03
        # Bitrate index: bits[15-12] (b2的bit0-0 + b3的bit7-5)
        bitrate_idx = ((b2 & 0x01) << 3) | ((b3 >> 5) & 0x07)
        # Sampling rate index: bits[10-9] (b3的bit4-2)
        sr_idx = (b3 >> 2) & 0x03
        # Channel mode 不影响帧大小计算，跳过

        # 验证版本+层组合合法性
        # version_bits: 00=MPEG2.5, 01=reserved, 10=MPEG2, 11=MPEG1
        # layer_bits: 00=reserved, 01=Layer3, 10=Layer2, 11=Layer1
        valid_version_layer = (
                (version_bits == 3 and layer_bits in (1, 2, 3)) or  # MPEG1
                (version_bits == 2 and layer_bits in (1, 2, 3)) or  # MPEG2
                (version_bits == 0 and layer_bits in (1, 2, 3))  # MPEG2.5
        )
        if not valid_version_layer:
            pos += 1
            continue

        # 位率索引: 0=free, 15=bad (都非法)
        if bitrate_idx == 0 or bitrate_idx == 15:
            pos += 1
            continue

        # 采样率索引: 3=reserved (非法)
        if sr_idx == 3:
            pos += 1
            continue

        # 查表获取位率(kbps)和采样率(Hz)
        # MPEG1 Layer III 位率表
        bitrate_map_v1_l3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
        # MPEG2/2.5 Layer III 位率表
        bitrate_map_v2_l3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
        # 采样率表
        sr_map = {0: 44100, 1: 48000, 2: 32000, 3: 0}  # version 1 (MPEG1)
        sr_map_2 = {0: 22050, 1: 24000, 2: 16000, 3: 0}  # version 2/2.5 (MPEG2/2.5)

        if version_bits == 3:  # MPEG1
            bitrate_kbps = bitrate_map_v1_l3[bitrate_idx]
            sample_rate = sr_map[sr_idx]
        else:  # MPEG2 or MPEG2.5
            bitrate_kbps = bitrate_map_v2_l3[bitrate_idx]
            sample_rate = sr_map_2[sr_idx]

        if bitrate_kbps == 0 or sample_rate == 0:
            pos += 1
            continue

        # 计算帧大小 (字节)
        # Layer3 (MP3): frame_size = 144 * bitrate / sample_rate + padding
        # Layer1: frame_size = 384 * bitrate / sample_rate + padding * 4
        # Layer2: frame_size = 1152 * bitrate / sample_rate + padding
        if layer_bits == 3:  # Layer1
            slot_size = 4
            samples_per_frame = 384
        elif layer_bits == 2:  # Layer2
            slot_size = 1
            samples_per_frame = 1152
        else:  # Layer3 (MP3)
            slot_size = 1
            samples_per_frame = 1152

        padding = (b3 >> 1) & 0x01  # 帧头 bit20: padding bit
        frame_size = (samples_per_frame * bitrate_kbps // sample_rate) + (padding * slot_size)

        # 基本合理性检查：帧大小应该在合理范围内
        if frame_size < 4 or frame_size > 8000:
            pos += 1
            continue

        return pos, bitrate_kbps, sample_rate, frame_size

    return None, None, None, None

def _skip_id3v2(data):
    """跳过 ID3v2 标签。返回数据起始位置。
    ID3v2 头部: 'ID3' + 版本(2) + 标志(1) + 大小(4, syncsafe integer)。"""
    if len(data) >= 10 and data[:3] == b'ID3':
        # syncsafe integer: 每7位一个字节，最高位为0
        size_bytes = data[6:10]
        size = ((size_bytes[0] & 0x7F) << 21) | ((size_bytes[1] & 0x7F) << 14) | \
               ((size_bytes[2] & 0x7F) << 7) | (size_bytes[3] & 0x7F)
        return 10 + size  # 跳过头部(10B) + 标签数据
    return 0

def _has_id3v1(data):
    """检查文件末尾是否有 ID3v1 标签（最后128字节以 'TAG' 开头）。"""
    return len(data) >= 128 and data[-128:-125] == b'TAG'

def _estimate_frame_samples(layer_bits):
    """根据 Layer 编号返回每帧采样数"""
    if layer_bits == 3:  # Layer1
        return 384
    elif layer_bits == 2:  # Layer2
        return 1152
    else:  # Layer3 (MP3)
        return 1152

def _scan_to_frame(data, start_pos, target_frame_idx, first_frame_offset, first_frame_size,
                   bitrate, sample_rate, layer_bits, max_scan_bytes=None):
    """逐帧扫描精确定位到目标帧号（VBR 安全）。返回目标帧的起始偏移，或 None。

    对于 CBR 文件，直接用 target_frame_idx * frame_size 计算即可。
    但 VBR 文件每帧位率可能不同，帧大小随之变化，必须逐帧扫描。

    算法：
    1. 从 start_pos 开始扫描同步字
    2. 解析每帧头部获取位率/填充位，计算实际帧大小
    3. 跳到下一帧，直到到达目标帧号
    4. 设置 max_scan_bytes 防止异常数据导致无限扫描
    """
    if max_scan_bytes is None:
        max_scan_bytes = len(data)  # 不限制

    file_size = len(data)
    pos = start_pos
    frame_idx = 0
    samples_per_frame = _estimate_frame_samples(layer_bits)

    # 位率查找表（同 _find_first_mp3_frame）
    bitrate_map_v1_l3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    bitrate_map_v2_l3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
    sr_map = {0: 44100, 1: 48000, 2: 32000, 3: 0}
    sr_map_2 = {0: 22050, 1: 24000, 2: 16000, 3: 0}

    while pos < file_size - 4 and (pos - start_pos) < max_scan_bytes:
        # 检查同步字
        if data[pos] != 0xFF or (data[pos + 1] & 0xE0) != 0xE0:
            pos += 1
            continue

        b2, b3, b4 = data[pos + 1], data[pos + 2], data[pos + 3]
        version_bits = (b2 >> 3) & 0x03
        layer_bits_cur = (b2 >> 1) & 0x03
        bitrate_idx = ((b2 & 0x01) << 3) | ((b3 >> 5) & 0x07)
        sr_idx = (b3 >> 2) & 0x03
        padding = (b3 >> 1) & 0x01

        # 验证合法性
        if (version_bits not in (0, 2, 3) or layer_bits_cur == 0 or
                bitrate_idx in (0, 15) or sr_idx == 3):
            pos += 1
            continue

        # 确保与第一帧的版本/层一致
        if version_bits != ((data[first_frame_offset + 1] >> 3) & 0x03) or \
                layer_bits_cur != ((data[first_frame_offset + 1] >> 1) & 0x03):
            pos += 1
            continue

        # 计算当前帧的位率和帧大小
        if version_bits == 3:  # MPEG1
            cur_bitrate = bitrate_map_v1_l3[bitrate_idx]
            cur_sr = sr_map[sr_idx]
        else:  # MPEG2/2.5
            cur_bitrate = bitrate_map_v2_l3[bitrate_idx]
            cur_sr = sr_map_2[sr_idx]

        if cur_bitrate == 0 or cur_sr == 0:
            pos += 1
            continue

        spf = _estimate_frame_samples(layer_bits_cur)
        slot_size = 4 if layer_bits_cur == 3 else 1  # Layer1=4, Layer2/3=1
        cur_frame_size = (spf * cur_bitrate // cur_sr) + (padding * slot_size)

        if cur_frame_size < 4 or cur_frame_size > 8000:
            pos += 1
            continue

        # 检查是否到达目标帧
        if frame_idx >= target_frame_idx:
            return (pos, frame_idx)  # v122: 返回 (偏移, 帧序号)

        # 跳到下一帧
        frame_idx += 1
        pos += cur_frame_size

    return None  # 未找到目标帧

def _neutralize_id3v2_tlen(data):
    """中和 ID3v2 中的 TLEN（时长）帧，防止截取文件播放位置错误。

    ID3v2 TLEN 帧记录了文件的总时长（毫秒），截取后这个值不再准确，
    会导致播放器显示错误的时长或进度条行为异常。

    解决方案：删除 TLEN 帧，让播放器通过帧数据自行计算时长。
    """
    if len(data) < 10:
        return data

    # 检查 ID3v2 头
    if data[:3] != b'ID3':
        return data

    # 解析 ID3v2 头
    version = data[3]
    flags = data[5]
    # 头大小（4字节 syncsafe integer）
    header_size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
    total_id3v2_size = header_size + 10  # +10 for header itself

    if version == 3 or version == 4:  # ID3v2.3 或 ID3v2.4
        pos = 10
        while pos + 10 <= total_id3v2_size:
            # 帧ID（4字节ASCII）
            frame_id = data[pos:pos + 4]
            if frame_id[0] == 0:  # padding area
                break

            # 帧大小
            if version == 4:
                frame_size = ((data[pos + 4] & 0x7F) << 21) | ((data[pos + 5] & 0x7F) << 14) | (
                            (data[pos + 6] & 0x7F) << 7) | (data[pos + 7] & 0x7F)
            else:  # v2.3
                frame_size = (data[pos + 4] << 24) | (data[pos + 5] << 16) | (data[pos + 6] << 8) | data[pos + 7]

            if frame_size <= 0 or frame_size > total_id3v2_size:
                break

            frame_header_size = 10
            total_frame_size = frame_header_size + frame_size

            if frame_id == b'TLEN':
                # 删除 TLEN 帧：跳过整个帧
                data = data[:pos] + data[pos + total_frame_size:]
                # 用零填充保持 ID3v2 总大小不变
                padding = b'\x00' * total_frame_size
                # 在 ID3v2 结尾（音频数据前）插入填充
                insert_pos = total_id3v2_size - total_frame_size
                data = data[:insert_pos] + padding + data[insert_pos:]
                break  # TLEN 只有一个，找到就退出

            pos += total_frame_size

    return data

def _neutralize_xing_vbri(data, first_frame_offset, first_frame_size):
    """中和 Xing/VBRI 信息帧，防止截取文件播放位置错误。

    VBR MP3 文件通常在第一帧内包含 Xing 或 VBRI 信息头，里面记录了：
    - 总帧数（TOC 表用的帧号）
    - 总字节数
    - 每100帧的字节偏移查找表（TOC）

    截取文件后，这些信息与实际音频数据不匹配，导致播放器：
    1. 显示错误的时长（用旧的总帧数计算）
    2. 拖动进度条时跳到错误位置（用旧的 TOC 表查找）
    3. 某些播放器直接从头开始播放（忽略 TOC 但帧计数错误）

    解决方案：将 Xing/VBRI 头中的帧数、字节数和 TOC 表清零，
    保留帧本身的音频数据结构（帧大小不变），让播放器回退到逐帧解码模式。
    """
    if first_frame_offset is None or first_frame_size < 48:
        return data

    frame_data = data[first_frame_offset:first_frame_offset + first_frame_size]

    # 检查 Xing 头（最常见的 VBR 标记）
    # Xing 头通常在帧侧信息（side information）之后
    # MPEG1 单声道: 偏移 36, 立体声: 偏移 21
    # MPEG2/2.5 单声道: 偏移 21, 立体声: 偏移 13
    b2 = frame_data[1]
    version_bits = (b2 >> 3) & 0x03
    channel_mode = (frame_data[3] >> 6) & 0x03 if len(frame_data) > 3 else 1

    # 计算侧信息长度
    if version_bits == 3:  # MPEG1
        side_info_len = 17 if channel_mode == 3 else 32  # 3=单声道
    else:  # MPEG2/2.5
        side_info_len = 9 if channel_mode == 3 else 17

    xing_offset = side_info_len + 4  # +4 for frame header

    found_xing = False
    if xing_offset + 8 <= len(frame_data):
        tag = frame_data[xing_offset:xing_offset + 4]
        if tag in (b'Xing', b'Info'):
            found_xing = True
            # Xing 头结构: 'Xing'(4) + flags(4) + [frames(4)] + [bytes(4)] + [TOC(100)] + [VBR scale(4)]
            flags = int.from_bytes(frame_data[xing_offset + 4:xing_offset + 8], 'big')
            pos = xing_offset + 8

            # 清除帧数
            if flags & 0x01:  # FRAMES_FLAG
                if pos + 4 <= len(frame_data):
                    frame_data = frame_data[:pos] + b'\x00\x00\x00\x00' + frame_data[pos + 4:]
                pos += 4
            # 清除字节数
            if flags & 0x02:  # BYTES_FLAG
                if pos + 4 <= len(frame_data):
                    frame_data = frame_data[:pos] + b'\x00\x00\x00\x00' + frame_data[pos + 4:]
                pos += 4
            # 清除 TOC 表（100字节）
            if flags & 0x04:  # TOC_FLAG
                if pos + 100 <= len(frame_data):
                    frame_data = frame_data[:pos] + (b'\x00' * 100) + frame_data[pos + 100:]
                pos += 100
            # 清除 VBR scale
            if flags & 0x08:  # VBR_SCALE_FLAG
                if pos + 4 <= len(frame_data):
                    frame_data = frame_data[:pos] + b'\x00\x00\x00\x00' + frame_data[pos + 4:]

    # 检查 VBRI 头（Fraunhofer 编码器使用，较少见）
    if not found_xing and len(frame_data) > 40:
        vbri_offset = 36  # VBRI 固定在偏移36
        if vbri_offset + 4 <= len(frame_data):
            tag = frame_data[vbri_offset:vbri_offset + 4]
            if tag == b'VBRI':
                # VBRI 结构更复杂，直接清零整个 VBRI 区域（从标记到帧末尾之前）
                # 保留 'VBRI' 标记，清除后续数据
                clear_start = vbri_offset + 4
                clear_end = min(len(frame_data), first_frame_size)
                if clear_start < clear_end:
                    frame_data = frame_data[:clear_start] + (b'\x00' * (clear_end - clear_start))

    # 替换原数据中的第一帧
    result = data[:first_frame_offset] + frame_data + data[first_frame_offset + first_frame_size:]
    return result

def truncate_audio(input_path, start_sec, output_path=None):
    """纯 Python 从 MP3 的 start_sec 秒处截取剩余部分，返回输出文件路径。
    基于 MP3 帧结构直接按字节切分，零依赖、零重编码、<0.01秒完成。

    v122 优化：
    - VBR 逐帧扫描：不再假设帧大小恒定，逐帧解析头部获取真实帧大小
    - ID3v2 标签保留：截取文件包含原始 ID3v2（部分解码器依赖）
    - 双向搜索：CBR 快速估算 + VBR 逐帧扫描，兼顾速度和精度
    - 安全边界检查更严格

    失败时返回 None。"""
    if not os.path.exists(input_path):
        return None
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        ts = int(time.time() * 1000) % 100000
        rand = random.randint(100, 999)
        output_path = f"{base}_cut{ts}{rand}_from{start_sec:.0f}s{ext}"

    try:
        with open(input_path, 'rb') as f:
            data = f.read()

        file_size = len(data)
        if file_size < 256:
            print(f"  [音频截取] ⚠️ 文件太小({file_size}B)，可能损坏")
            return None

        # 1. 跳过 ID3v2 标签，保存其数据用于截取文件头部
        id3v2_end = _skip_id3v2(data)
        id3v2_data = data[:id3v2_end] if id3v2_end > 0 else b''

        # 2. 找到第一帧，获取位率、采样率、帧大小
        frame_offset, bitrate, sample_rate, frame_size = _find_first_mp3_frame(data, id3v2_end)
        if frame_offset is None:
            print(f"  [音频截取] ⚠️ 未找到有效的MP3帧头")
            return None

        # 3. 解析第一帧的版本/层信息（用于 VBR 逐帧扫描）
        b2 = data[frame_offset + 1]
        version_bits = (b2 >> 3) & 0x03
        layer_bits = (b2 >> 1) & 0x03
        samples_per_frame = _estimate_frame_samples(layer_bits)

        # 4. 计算目标帧号
        frame_duration = samples_per_frame / sample_rate  # 单帧时长(秒)
        target_frame_idx = int(start_sec / frame_duration)  # 目标帧序号(0-based)

        # 5. 定位目标帧：先尝试 VBR 逐帧扫描，失败则 CBR 估算
        # --- 方案A: VBR 逐帧扫描（精确，适用所有文件）---
        # 从第一帧开始逐帧扫描到目标帧
        # 限制扫描范围为 CBR 估算偏移的 ±50% （加速大文件）
        cbr_estimate_offset = frame_offset + target_frame_idx * frame_size
        max_scan = int(file_size * 0.6)  # 最多扫描 60% 文件大小

        scan_result = _scan_to_frame(
            data, frame_offset, target_frame_idx,
            frame_offset, frame_size,
            bitrate, sample_rate, layer_bits,
            max_scan_bytes=max_scan
        )
        scanned_frame_idx = 0  # v122: 记录实际扫描到的帧号
        if scan_result is not None:
            aligned_offset, scanned_frame_idx = scan_result
        else:
            aligned_offset = None

        # --- 方案B: CBR 估算（VBR 扫描失败时的兜底）---
        if aligned_offset is None:
            aligned_offset = cbr_estimate_offset
            # 从估算位置向前搜索同步字对齐
            search_start = max(frame_offset, aligned_offset - 4 * frame_size)
            scan_pos = search_start
            while scan_pos < min(file_size - 4, aligned_offset + 4 * frame_size):
                if data[scan_pos] == 0xFF and (data[scan_pos + 1] & 0xE0) == 0xE0:
                    # 验证帧头合法性
                    sb2 = data[scan_pos + 1]
                    sv = (sb2 >> 3) & 0x03
                    sl = (sb2 >> 1) & 0x03
                    if sv in (0, 2, 3) and sl in (1, 2, 3):
                        aligned_offset = scan_pos
                        break
                scan_pos += 1

            if aligned_offset is None or aligned_offset >= file_size - 128:
                print(f"  [音频截取] ⚠️ 无法对齐到帧边界")
                return None

        # 安全边界检查
        if aligned_offset >= file_size - 128:
            print(f"  [音频截取] ⚠️ 截取起点({aligned_offset})已超过文件长度({file_size})")
            return None

        # 6. 构造输出数据：ID3v2标签(如果有) + 从对齐位置开始的帧数据 + ID3v1尾标签
        has_id3v1_tail = _has_id3v1(data)
        tail_data = data[-128:] if has_id3v1_tail else b''
        audio_data = data[aligned_offset:(file_size - 128 if has_id3v1_tail else file_size)]

        # v122: 中和截取数据中的 VBR 信息头和 ID3v2 时长标签
        # 原因：截取后文件帧数和字节数与 Xing 头/TLEN 记录不匹配，
        # 导致播放器显示错误时长、拖动跳转位置错误、甚至从头播放
        output_data = id3v2_data + audio_data
        # 中和 ID3v2 中的 TLEN（时长）标签
        # TLEN 记录的是原始文件的总时长(ms)，截取后不再准确
        if id3v2_data:
            output_data = _neutralize_id3v2_tlen(output_data)
        # 找到截取数据中第一帧的偏移（跳过 ID3v2 后就是第一帧）
        output_first_frame_offset = len(id3v2_data)
        # 中和 Xing/VBRI 信息帧
        temp_full = output_data
        temp_full = _neutralize_xing_vbri(temp_full, output_first_frame_offset, frame_size)
        audio_data = temp_full[output_first_frame_offset:]

        with open(output_path, 'wb') as f:
            # 写入 ID3v2 标签（保留原始标签，部分解码器依赖）
            if id3v2_data:
                f.write(id3v2_data)
            # 写入音频帧数据（已中和 Xing/VBRI 头）
            f.write(audio_data)
            # 写入 ID3v1 标签
            if tail_data:
                f.write(tail_data)

        output_size = os.path.getsize(output_path)
        # 计算实际截取位置（秒）——v122 修复
        # 方法：用 _scan_to_frame 返回的实际帧号 × 帧时长（VBR精确）
        # 兜底：字节比例法（CBR精确，VBR近似）
        audio_bytes = len(audio_data)
        input_audio_bytes = file_size - id3v2_end - (128 if has_id3v1_tail else 0)
        skipped_bytes = aligned_offset - frame_offset
        if scanned_frame_idx > 0:
            # 逐帧扫描成功，帧号×帧时长 = 精确的跳过秒数（CBR/VBR 都准确）
            actual_skip_sec = scanned_frame_idx * frame_duration
        elif input_audio_bytes > 0 and skipped_bytes >= 0 and bitrate > 0:
            # 兜底：CBR 估算用字节比例
            total_sec_est = input_audio_bytes * 8 / (bitrate * 1000)
            byte_ratio = skipped_bytes / input_audio_bytes
            actual_skip_sec = byte_ratio * total_sec_est
        else:
            actual_skip_sec = 0

        print(f"  [音频截取] ✅ 纯Python截取: 目标{start_sec:.1f}s, 实际≈{actual_skip_sec:.1f}s, "
              f"输入{file_size // 1024}KB→输出{output_size // 1024}KB, "
              f"位率{bitrate}kbps, 采样率{sample_rate}Hz, 帧长{frame_duration * 1000:.1f}ms, "
              f"ID3v2={'有' if id3v2_data else '无'}, ID3v1={'有' if has_id3v1_tail else '无'}")
        return output_path

    except Exception as e:
        print(f"  [音频截取] 纯Python截取异常: {e}")
        return None

