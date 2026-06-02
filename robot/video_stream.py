#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yanshee 视频流自动管理模块
- 优先通过 SSH 部署 MJPEG 服务
- SSH 不可用时，通过 ADB 自动开启 SSH 然后重试
- ADB 也无法开启 SSH 时，直接走 ADB 部署
- 检测流是否已运行，未运行则自动拉起

使用方式:
    from robot.video_stream import ensure_video_stream
    ensure_video_stream()  # 阻塞直到流可用或超时

配置（在 config/__init__.py 中）:
    VIDEO_STREAM_PORT = 8080          # MJPEG 服务端口
    VIDEO_STREAM_TIMEOUT = 15         # 等待流就绪超时(秒)
    YANSHEE_SSH_USER = "root"         # SSH 用户名
    YANSHEE_SSH_PASSWORD = ""         # SSH 密码（留空用密钥认证）
    YANSHEE_SSH_PORT = 22             # SSH 端口
    VIDEO_STREAM_REMOTE_DIR = "/tmp"  # 机器人上脚本存放目录
    YANSHEE_ADB_PORT = 5555           # ADB TCP 端口
    ADB_CONNECT_TIMEOUT = 8           # ADB 连接超时(秒)
"""
import os
import sys
import time
import subprocess
import socket
import urllib.request

# paramiko: Python 原生 SSH，无需 sshpass，Windows/Linux 通用
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

# 尝试导入 config
try:
    from config import ROBOT_IP, ROBOT_PORT
except ImportError:
    ROBOT_IP = os.getenv("YANSHEE_IP", "192.168.3.241")
    ROBOT_PORT = 9090

# ========== 默认配置（可被 config 覆盖）==========
VIDEO_STREAM_PORT = 8080
VIDEO_STREAM_TIMEOUT = 15
YANSHEE_SSH_USER = "pi"
YANSHEE_SSH_PASSWORD = "raspberry"
YANSHEE_SSH_PORT = 22
VIDEO_STREAM_REMOTE_DIR = "/tmp"
YANSHEE_ADB_PORT = 5555
ADB_CONNECT_TIMEOUT = 8
ADB_SSH_START_CMDS = [
    "start sshd",
    "svc ssh start",
    "setprop service.adb.tcp.port 5555 && start adbd",
]

# 尝试从 config 导入覆盖
try:
    from config import (
        VIDEO_STREAM_PORT as _VSP,
        VIDEO_STREAM_TIMEOUT as _VST,
        YANSHEE_SSH_USER as _YSU,
        YANSHEE_SSH_PASSWORD as _YSP,
        YANSHEE_SSH_PORT as _YSPO,
        VIDEO_STREAM_REMOTE_DIR as _VSRD,
    )
    VIDEO_STREAM_PORT = _VSP
    VIDEO_STREAM_TIMEOUT = _VST
    YANSHEE_SSH_USER = _YSU
    YANSHEE_SSH_PASSWORD = _YSP
    YANSHEE_SSH_PORT = _YSPO
    VIDEO_STREAM_REMOTE_DIR = _VSRD
except ImportError:
    pass

try:
    from config import (
        YANSHEE_ADB_PORT as _YAP,
        ADB_CONNECT_TIMEOUT as _ACT,
        ADB_SSH_START_CMDS as _ASSC,
    )
    YANSHEE_ADB_PORT = _YAP
    ADB_CONNECT_TIMEOUT = _ACT
    ADB_SSH_START_CMDS = _ASSC
except ImportError:
    pass

# 环境变量覆盖
VIDEO_STREAM_PORT = int(os.getenv("YANSHEE_STREAM_PORT", VIDEO_STREAM_PORT))
VIDEO_STREAM_TIMEOUT = int(os.getenv("YANSHEE_STREAM_TIMEOUT", VIDEO_STREAM_TIMEOUT))
YANSHEE_SSH_USER = os.getenv("YANSHEE_SSH_USER", YANSHEE_SSH_USER)
YANSHEE_SSH_PASSWORD = os.getenv("YANSHEE_SSH_PASSWORD", YANSHEE_SSH_PASSWORD)
YANSHEE_SSH_PORT = int(os.getenv("YANSHEE_SSH_PORT", YANSHEE_SSH_PORT))
YANSHEE_ADB_PORT = int(os.getenv("YANSHEE_ADB_PORT", YANSHEE_ADB_PORT))
ADB_CONNECT_TIMEOUT = int(os.getenv("ADB_CONNECT_TIMEOUT", ADB_CONNECT_TIMEOUT))

# 本地脚本路径
_LOCAL_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "deploy", "mjpeg_server.py"
)
_REMOTE_SCRIPT = f"{VIDEO_STREAM_REMOTE_DIR}/mjpeg_server.py"

# ADB 全局状态：是否已完成 connect
_adb_connected = False


# ======================== 基础工具函数 ========================

def _is_port_open(host, port, timeout=2):
    """检测端口是否可达"""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _check_stream(host=None, port=None):
    """检测 MJPEG 流是否可用（发 GET 请求看返回头）"""
    host = host or ROBOT_IP
    port = port or VIDEO_STREAM_PORT
    url = f"http://{host}:{port}/stream"
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=3)
        content_type = resp.headers.get("Content-Type", "")
        resp.close()
        return "multipart/x-mixed-replace" in content_type
    except Exception:
        return False


# ======================== SSH 操作（paramiko 原生，无需 sshpass）====================

def _get_ssh_client(host=None, user=None, password=None, port=None, timeout=15):
    """
    创建并连接 paramiko SSH 客户端。
    返回 (client, error_msg)。成功时 error_msg 为 None。
    支持密码认证和密钥认证两种方式。
    """
    host = host or ROBOT_IP
    user = user or YANSHEE_SSH_USER
    password = password or YANSHEE_SSH_PASSWORD
    port = port or YANSHEE_SSH_PORT

    if not PARAMIKO_AVAILABLE:
        return None, "paramiko 未安装，无法使用 SSH"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if password:
            # 密码认证（无需 sshpass）
            client.connect(
                host, port=port, username=user, password=password,
                timeout=timeout, allow_agent=False, look_for_keys=False
            )
        else:
            # 密钥认证
            client.connect(
                host, port=port, username=user,
                timeout=timeout, allow_agent=True, look_for_keys=True
            )
        return client, None
    except paramiko.AuthenticationException as e:
        return None, f"SSH 认证失败: {e}（用户={user}, 密码={'已设置' if password else '未设置'}）"
    except paramiko.SSHException as e:
        return None, f"SSH 连接异常: {e}"
    except socket.timeout:
        return None, f"SSH 连接超时（{host}:{port}，{timeout}s）"
    except OSError as e:
        return None, f"SSH 网络错误: {e}"
    except Exception as e:
        return None, f"SSH 连接失败: {e}"


def _try_ssh(host, user, password, port, quiet=False):
    """快速检测 SSH 是否可达（paramiko 连接 + echo 验证）"""
    cmd = "echo SSH_OK"
    rc, out, _ = _ssh_cmd(cmd, host, user, password, port, timeout=10)
    ok = (rc == 0 and "SSH_OK" in out)
    if not quiet and not ok:
        print(f"[VideoStream] SSH 不可达（{host}:{port}）")
    return ok


def _ssh_cmd(command, host=None, user=None, password=None, port=None,
             timeout=30):
    """
    通过 paramiko SSH 执行远程命令。
    返回 (returncode, stdout, stderr)
    """
    host = host or ROBOT_IP
    user = user or YANSHEE_SSH_USER
    password = password or YANSHEE_SSH_PASSWORD
    port = port or YANSHEE_SSH_PORT

    connect_timeout = min(timeout, 15)
    client, error = _get_ssh_client(host, user, password, port, timeout=connect_timeout)
    if client is None:
        return -1, "", error

    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out, err
    except Exception as e:
        return -1, "", f"SSH 命令执行失败: {e}"
    finally:
        client.close()


def _scp_upload(local_path, remote_path, host=None, user=None, password=None,
                port=None, timeout=30):
    """
    通过 paramiko SFTP 上传文件到机器人。
    返回 (returncode, stdout, stderr)
    """
    host = host or ROBOT_IP
    user = user or YANSHEE_SSH_USER
    password = password or YANSHEE_SSH_PASSWORD
    port = port or YANSHEE_SSH_PORT

    connect_timeout = min(timeout, 15)
    client, error = _get_ssh_client(host, user, password, port, timeout=connect_timeout)
    if client is None:
        return -1, "", error

    try:
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        return 0, "上传成功", ""
    except Exception as e:
        return -1, "", f"SFTP 上传失败: {e}"
    finally:
        client.close()


# ======================== ADB 操作 ========================

def _adb_connect(host=None, port=None, timeout=None, quiet=False):
    """
    通过 ADB TCP 连接机器人。
    返回 (success, message)
    """
    global _adb_connected
    host = host or ROBOT_IP
    port = port or YANSHEE_ADB_PORT
    timeout = timeout or ADB_CONNECT_TIMEOUT

    if _adb_connected:
        return True, "已连接"

    if not quiet:
        print(f"[VideoStream] ADB 连接 → {host}:{port} ...")

    try:
        result = subprocess.run(
            ["adb", "connect", f"{host}:{port}"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout
        )
        out = ((result.stdout or "") + (result.stderr or "")).strip()
        if "connected" in out.lower() or "already connected" in out.lower():
            _adb_connected = True
            if not quiet:
                print(f"[VideoStream] ✅ ADB 已连接 → {host}:{port}")
            return True, out
        else:
            if not quiet:
                print(f"[VideoStream] ⚠️ ADB 连接返回: {out}")
            return False, out
    except subprocess.TimeoutExpired:
        return False, "ADB 连接超时"
    except FileNotFoundError:
        return False, "ADB 未安装，请安装 Android SDK Platform Tools"


def _adb_shell(command, host=None, port=None, timeout=15):
    """
    通过 ADB shell 执行命令。
    返回 (returncode, stdout, stderr)
    注意：adb shell 的 returncode 不一定可靠，以 stdout 为准。
    """
    host = host or ROBOT_IP
    port = port or YANSHEE_ADB_PORT

    ok, _ = _adb_connect(host, port)
    if not ok:
        return -1, "", "ADB 未连接"

    try:
        result = subprocess.run(
            ["adb", "shell", command],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "ADB shell 超时"
    except FileNotFoundError:
        return -1, "", "ADB 未安装"


def _adb_push(local_path, remote_path, host=None, port=None, timeout=30):
    """
    通过 ADB push 上传文件到机器人。
    返回 (success: bool, message: str)
    """
    host = host or ROBOT_IP
    port = port or YANSHEE_ADB_PORT

    ok, _ = _adb_connect(host, port)
    if not ok:
        return False, "ADB 未连接"

    try:
        result = subprocess.run(
            ["adb", "push", local_path, remote_path],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout
        )
        out = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode == 0 and "error" not in out.lower():
            return True, out
        return False, out
    except subprocess.TimeoutExpired:
        return False, "ADB push 超时"
    except FileNotFoundError:
        return False, "ADB 未安装"


def _adb_enable_ssh(host=None, port=None, quiet=False):
    """
    通过 ADB 在机器人上开启 SSH 服务。
    依次尝试多个启动命令。
    返回 (success, message)
    """
    host = host or ROBOT_IP
    port = port or YANSHEE_ADB_PORT

    if not quiet:
        print(f"[VideoStream] 🔧 ADB 尝试开启 SSH ...")

    ok, _ = _adb_connect(host, port)
    if not ok:
        return False, "ADB 未连接，无法开启 SSH"

    # 先检测 SSH 是否已经在跑
    rc, out, _ = _adb_shell("pgrep -f sshd || pgrep -f dropbear || echo NOT_RUNNING",
                             host, port, timeout=8)
    out = (out or "").strip()
    if rc == 0 and "NOT_RUNNING" not in out and out:
        if not quiet:
            print(f"[VideoStream] SSH 已在运行（PID: {out.strip()}）")
        return True, "SSH 已在运行"

    if not quiet:
        print(f"[VideoStream] SSH 未运行，尝试启动...")

    # 依次尝试启动命令
    for cmd in ADB_SSH_START_CMDS:
        if not quiet:
            print(f"[VideoStream]   → {cmd}")
        rc, out, err = _adb_shell(cmd, host, port, timeout=10)
        out = ((out or "") + (err or "")).strip().lower()
        if "not found" in out or "unknown" in out:
            continue
        # 启动后等一下
        time.sleep(1.5)

    # 验证是否启动成功
    rc, out, _ = _adb_shell("pgrep -f sshd || pgrep -f dropbear || echo NOT_RUNNING",
                             host, port, timeout=8)
    out = (out or "").strip()
    if rc == 0 and "NOT_RUNNING" not in out and out:
        if not quiet:
            print(f"[VideoStream] ✅ SSH 已开启（PID: {out}）")
        return True, "SSH 已开启"

    if not quiet:
        print(f"[VideoStream] ⚠️ ADB 无法开启 SSH（可能系统不支持）")
    return False, "ADB 无法开启 SSH"


def _adb_deploy_and_start(host=None, port=None, stream_port=None, quiet=False):
    """
    通过 ADB 直接部署 mjpeg_server.py 并启动。
    跳过 SSH 环节，ADB push + adb shell nohup。
    返回 (success, message)
    """
    host = host or ROBOT_IP
    port = port or YANSHEE_ADB_PORT
    stream_port = stream_port or VIDEO_STREAM_PORT

    # 确保 ADB 已连接
    ok, msg = _adb_connect(host, port)
    if not ok:
        return False, f"ADB 连接失败: {msg}"

    # 1. 检测远程脚本是否存在
    rc, out, _ = _adb_shell(
        f"test -f {_REMOTE_SCRIPT} && echo 'EXISTS' || echo 'MISSING'",
        host, port, timeout=8
    )
    if "MISSING" in (out or ""):
        if not quiet:
            print(f"[VideoStream] ADB push → {_REMOTE_SCRIPT}")
        ok, msg = _adb_push(_LOCAL_SCRIPT, _REMOTE_SCRIPT, host, port)
        if not ok:
            return False, f"ADB push 失败: {msg}"

    # 2. 杀掉旧进程
    _adb_shell("pkill -f mjpeg_server.py 2>/dev/null; echo done",
               host, port, timeout=5)

    # 3. 启动
    start_cmd = (
        f"nohup python3 {_REMOTE_SCRIPT} "
        f"--port {stream_port} --camera 0 "
        f"> {VIDEO_STREAM_REMOTE_DIR}/mjpeg.log 2>&1 &"
    )
    if not quiet:
        print(f"[VideoStream] ADB 启动 MJPEG 服务...")
    rc, out, err = _adb_shell(start_cmd, host, port, timeout=10)
    if rc != 0 and err.strip():
        return False, f"ADB 启动失败: {err.strip()}"

    if not quiet:
        print(f"[VideoStream] MJPEG 服务已提交启动，等待就绪...")
    return True, "ADB 部署完成"


# ======================== SSH 部署子流程 ========================

def _deploy_via_ssh(host, port, ssh_user, ssh_password, ssh_port, quiet=False):
    """
    通过 SSH 部署 MJPEG 服务。
    返回 (success, message)
    """
    # 检测远程脚本是否存在
    check_cmd = f"test -f {_REMOTE_SCRIPT} && echo 'EXISTS' || echo 'MISSING'"
    rc, out, err = _ssh_cmd(check_cmd, host, ssh_user, ssh_password, ssh_port, timeout=12)
    if rc != 0:
        return False, f"SSH 连接失败: {err.strip()}"

    if "MISSING" in (out or ""):
        if not quiet:
            print(f"[VideoStream] SCP 上传 → {_REMOTE_SCRIPT}")
        rc, out, err = _scp_upload(
            _LOCAL_SCRIPT, _REMOTE_SCRIPT,
            host, ssh_user, ssh_password, ssh_port
        )
        if rc != 0:
            return False, f"SCP 上传失败: {err.strip()}"

    # 杀掉旧进程
    _ssh_cmd("pkill -f mjpeg_server.py 2>/dev/null; echo done",
             host, ssh_user, ssh_password, ssh_port, timeout=8)

    # 启动
    start_cmd = (
        f"nohup python3 {_REMOTE_SCRIPT} "
        f"--port {port} --camera 0 "
        f"> {VIDEO_STREAM_REMOTE_DIR}/mjpeg.log 2>&1 &"
    )
    rc, out, err = _ssh_cmd(start_cmd, host, ssh_user, ssh_password, ssh_port, timeout=12)
    if rc != 0:
        return False, f"SSH 启动失败: {err.strip()}"

    return True, "SSH 部署完成"


# ======================== 等待流就绪 ========================

def _wait_for_stream(host, port, timeout, quiet=False):
    """轮询等待 MJPEG 流就绪。返回 (success, message)"""
    stream_url = f"http://{host}:{port}/stream"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_stream(host, port):
            elapsed = timeout - (deadline - time.time())
            if not quiet:
                print(f"[VideoStream] ✅ 视频流就绪（{elapsed:.1f}s）→ {stream_url}")
            return True, "流已就绪"
        time.sleep(0.8)
    return False, f"视频流启动超时（{timeout}s）"


# ======================== 主入口 ========================

def ensure_video_stream(
    host=None, port=None, timeout=None,
    ssh_user=None, ssh_password=None, ssh_port=None,
    adb_port=None, quiet=False, skip_ssh=False
):
    """
    确保机器人 MJPEG 视频流可用。

    ┌─────────────────────────────────────────────┐
    │ 1. 检测流是否已可用 → ✅ 直接返回             │
    │ 2. 检测端口是否被占用 → ⚠️ 报错               │
    │ 3. 尝试 SSH 部署                              │
    │    ├─ 成功 → 等待流就绪                       │
    │    └─ 失败 → 进入 ADB 恢复流程 ↓              │
    │ 4. ADB 尝试开启 SSH → 重试 SSH 部署           │
    │    ├─ 成功 → 等待流就绪                       │
    │    └─ 失败 → ADB 直接部署 ↓                   │
    │ 5. ADB push + shell 启动 mjpeg_server.py      │
    │    └─ 等待流就绪                              │
    └─────────────────────────────────────────────┘

    返回: (success: bool, message: str)
    """
    host = host or ROBOT_IP
    port = port or VIDEO_STREAM_PORT
    timeout = timeout or VIDEO_STREAM_TIMEOUT
    ssh_user = ssh_user or YANSHEE_SSH_USER
    ssh_password = ssh_password or YANSHEE_SSH_PASSWORD
    ssh_port = ssh_port or YANSHEE_SSH_PORT
    adb_port = adb_port or YANSHEE_ADB_PORT

    stream_url = f"http://{host}:{port}/stream"

    # ── Step 1: 快速检测流是否已可用 ──
    if not quiet:
        print(f"[VideoStream] 检测视频流 {stream_url} ...")
    if _check_stream(host, port):
        if not quiet:
            print(f"[VideoStream] ✅ 视频流已就绪 → {stream_url}")
        return True, "流已就绪"

    if not quiet:
        print(f"[VideoStream] 视频流未就绪，开始自动部署...")

    # ── Step 2: 端口冲突检测 ──
    if _is_port_open(host, port):
        print(f"[VideoStream] ⚠️ 端口 {port} 已开放但不是 MJPEG 流，可能被占用")
        return False, f"端口 {port} 被非 MJPEG 服务占用"

    if skip_ssh:
        return False, "视频流不可用（已设置 skip_ssh）"

    # ── Step 3: 尝试 SSH 部署 ──
    if not quiet:
        print(f"[VideoStream] ── 方式1: SSH 部署 ──")
        print(f"[VideoStream] 尝试 SSH {ssh_user}@{host}:{ssh_port}")

    ok, msg = _deploy_via_ssh(host, port, ssh_user, ssh_password, ssh_port, quiet)
    if ok:
        ok, msg = _wait_for_stream(host, port, timeout, quiet)
        if ok:
            return True, msg
        # 流没起来也不算完全失败，继续走 ADB
        if not quiet:
            print(f"[VideoStream] SSH 部署完成但流未就绪，尝试 ADB 方案...")
    elif not quiet:
        print(f"[VideoStream] SSH 部署失败: {msg}")

    # ── Step 4: ADB 尝试开启 SSH ──
    if not quiet:
        print(f"[VideoStream] ── 方式2: ADB 开启 SSH ──")

    ssh_enabled, _ = _adb_enable_ssh(host, adb_port, quiet)
    if ssh_enabled:
        # 等一下 SSH 服务完全起来
        time.sleep(1.5)
        if _try_ssh(host, ssh_user, ssh_password, ssh_port, quiet=True):
            if not quiet:
                print(f"[VideoStream] SSH 已就绪，重新尝试 SSH 部署...")
            ok, msg = _deploy_via_ssh(host, port, ssh_user, ssh_password, ssh_port, quiet)
            if ok:
                ok, msg = _wait_for_stream(host, port, timeout, quiet)
                if ok:
                    return True, msg
            elif not quiet:
                print(f"[VideoStream] ADB开启SSH后部署仍然失败: {msg}")

    # ── Step 5: ADB 直接部署（最后手段）──
    if not quiet:
        print(f"[VideoStream] ── 方式3: ADB 直接部署 ──")

    ok, msg = _adb_deploy_and_start(host, adb_port, port, quiet)
    if not ok:
        print(f"[VideoStream] ❌ {msg}")
        return False, msg

    ok, msg = _wait_for_stream(host, port, timeout, quiet)
    if not ok:
        # 尝试读日志
        rc, log_out, _ = _adb_shell(
            f"tail -5 {VIDEO_STREAM_REMOTE_DIR}/mjpeg.log 2>/dev/null || echo 'NO_LOG'",
            host, adb_port, timeout=8
        )
        if log_out.strip() and log_out.strip() != "NO_LOG":
            msg += f"\n机器人日志: {log_out.strip()}"
        print(f"[VideoStream] ❌ {msg}")
        return False, msg

    return True, msg


# ========== 便捷函数 ==========

def start_video_stream(**kwargs):
    """ensure_video_stream 的别名，直接集成到启动流程"""
    return ensure_video_stream(**kwargs)


# ========== 命令行入口 ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Yanshee 视频流管理工具")
    parser.add_argument("--host", default=ROBOT_IP, help="机器人 IP")
    parser.add_argument("--port", type=int, default=VIDEO_STREAM_PORT, help="流端口")
    parser.add_argument("--ssh-user", default=YANSHEE_SSH_USER, help="SSH 用户名")
    parser.add_argument("--ssh-password", default=YANSHEE_SSH_PASSWORD, help="SSH 密码")
    parser.add_argument("--ssh-port", type=int, default=YANSHEE_SSH_PORT, help="SSH 端口")
    parser.add_argument("--adb-port", type=int, default=YANSHEE_ADB_PORT, help="ADB 端口")
    parser.add_argument("--timeout", type=int, default=VIDEO_STREAM_TIMEOUT, help="超时秒数")
    parser.add_argument("--skip-ssh", action="store_true", help="仅检测，不 SSH")
    parser.add_argument("--check", action="store_true", help="仅检测流状态")
    parser.add_argument("--enable-ssh", action="store_true", help="仅通过 ADB 开启 SSH")
    parser.add_argument("--setup-key", action="store_true",
                        help="生成 SSH 密钥对并部署到机器人（需提供 --ssh-password）")
    parser.add_argument("--diagnose", action="store_true",
                        help="诊断 SSH/ADB/MJPEG 连接状态")
    args = parser.parse_args()

    if args.diagnose:
        print(f"=== Yanshee 机器人连接诊断 ===")
        print(f"IP: {args.host}")
        print(f"API (9090):    {'✅ 可达' if _is_port_open(args.host, 9090, timeout=3) else '❌ 不可达'}")
        print(f"SSH (22):      {'✅ 可达' if _is_port_open(args.host, args.ssh_port, timeout=3) else '❌ 不可达'}")
        print(f"ADB TCP (5555): {'✅ 可达' if _is_port_open(args.host, args.adb_port, timeout=3) else '❌ 不可达'}")
        print(f"MJPEG (8080):  {'✅ 可达' if _is_port_open(args.host, args.port, timeout=3) else '❌ 不可达'}")
        print(f"流状态:        {'✅ 可用' if _check_stream(args.host, args.port) else '❌ 不可用'}")
        print()
        print(f"SSH 用户: {args.ssh_user or YANSHEE_SSH_USER}")
        print(f"SSH 密码: {'已设置' if (args.ssh_password or YANSHEE_SSH_PASSWORD) else '❌ 未设置（需设置 YANSHEE_SSH_PASSWORD 环境变量）'}")
        print(f"paramiko:  {'✅ 可用' if PARAMIKO_AVAILABLE else '❌ 未安装（pip install paramiko）'}")
        ok = _try_ssh(args.host, args.ssh_user, args.ssh_password, args.ssh_port, quiet=True)
        print(f"SSH 认证:  {'✅ 通过' if ok else '❌ 失败（需设置密码或部署密钥）'}")
        sys.exit(0)

    if args.setup_key:
        print("=== SSH 密钥生成与部署 ===")
        ssh_dir = os.path.expanduser("~/.ssh")
        os.makedirs(ssh_dir, exist_ok=True)
        key_path = os.path.join(ssh_dir, "id_rsa_yanshee")

        if not os.path.exists(key_path):
            print(f"生成密钥对: {key_path}")
            result = subprocess.run(
                ["ssh-keygen", "-t", "rsa", "-b", "2048",
                 "-f", key_path, "-N", "", "-C", "yanshee-auto"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30
            )
            if result.returncode != 0:
                print(f"❌ 密钥生成失败: {result.stderr}")
                sys.exit(1)
            print("✅ 密钥对已生成")
        else:
            print(f"密钥已存在: {key_path}")

        # 读取公钥
        pubkey_path = key_path + ".pub"
        with open(pubkey_path, "r") as f:
            pubkey = f.read().strip()
        print(f"公钥: {pubkey[:60]}...")

        # 部署到机器人
        password = args.ssh_password or YANSHEE_SSH_PASSWORD
        if not password:
            print("❌ 需要提供 --ssh-password 才能自动部署密钥")
            print(f"\n手动操作:")
            print(f"  1. 复制以下公钥:")
            print(f"     {pubkey}")
            print(f"  2. SSH 登录机器人: ssh {args.ssh_user}@{args.host}")
            print(f"  3. 执行: mkdir -p ~/.ssh")
            print(f"  4. 执行: echo '{pubkey}' >> ~/.ssh/authorized_keys")
            print(f"  5. 执行: chmod 600 ~/.ssh/authorized_keys")
            print(f"\n或设置环境变量: set YANSHEE_SSH_PASSWORD=你的密码")
            sys.exit(1)

        print("部署公钥到机器人...")
        deploy_cmd = f"mkdir -p ~/.ssh && echo '{pubkey}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo DONE"
        rc, out, err = _ssh_cmd(
            deploy_cmd, args.host, args.ssh_user, password, args.ssh_port, timeout=15
        )
        if rc == 0 and "DONE" in out:
            print("✅ 密钥部署成功！现在可以用密钥免密登录了")

            # 更新本地 SSH config
            config_path = os.path.join(ssh_dir, "config")
            host_entry = f"\nHost yanshee\n  HostName {args.host}\n  User {args.ssh_user}\n  IdentityFile {key_path}\n  StrictHostKeyChecking no\n"

            existing = ""
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    existing = f.read()
            if "Host yanshee" not in existing:
                with open(config_path, "a") as f:
                    f.write(host_entry)
                print("✅ 已添加到 ~/.ssh/config，现在可以用 'ssh yanshee' 连接")

            # 验证
            rc, out, _ = _ssh_cmd("echo KEY_OK", args.host, args.ssh_user, password, args.ssh_port)
            if rc == 0 and "KEY_OK" in out:
                print("✅ 密钥认证验证通过")
            else:
                print("⚠️ 密钥部署了但验证失败，请手动测试")
        else:
            print(f"❌ 密钥部署失败: {err or out}")
        sys.exit(0 if rc == 0 else 1)

    if args.check:
        ok = _check_stream(args.host, args.port)
        print(f"流状态: {'✅ 可用' if ok else '❌ 不可用'}")
        sys.exit(0 if ok else 1)

    if args.enable_ssh:
        ok, msg = _adb_enable_ssh(args.host, args.adb_port)
        print(msg)
        if ok and _is_port_open(args.host, args.ssh_port or YANSHEE_SSH_PORT):
            print(f"✅ SSH 端口 {args.ssh_port or YANSHEE_SSH_PORT} 已开放")
        sys.exit(0 if ok else 1)

    success, msg = ensure_video_stream(
        host=args.host, port=args.port, timeout=args.timeout,
        ssh_user=args.ssh_user, ssh_password=args.ssh_password,
        ssh_port=args.ssh_port, adb_port=args.adb_port,
        skip_ssh=args.skip_ssh,
    )
    sys.exit(0 if success else 1)
