#!/bin/bash
# 清理旧进程
pkill -f mjpeg_server 2>/dev/null
sleep 2

# 创建新的 MJPEG 服务器脚本
cat > /tmp/mjpeg_server_v2.py << 'EOF'
#!/usr/bin/env python3
"""Yanshee MJPEG Stream Server v2 - Fixed MJPEG format"""
import cv2
import threading
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream' or self.path == '/':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = global_frame
                    if frame is not None:
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        if ret:
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                    time.sleep(0.05)
            except Exception as e:
                print("Client disconnected")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

global_frame = None

def capture_loop():
    global global_frame
    cap = cv2.VideoCapture('/dev/video0')
    
    # 强制使用 MJPEG 格式 - 修复 V4L2 错误
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return
        
    print("Camera opened successfully (MJPEG format)")
    
    while True:
        ret, frame = cap.read()
        if ret:
            global_frame = frame
        else:
            print("Warning: Frame grab failed")
        time.sleep(0.05)
    
    cap.release()

if __name__ == '__main__':
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    port = 8080
    server = StreamingServer(('0.0.0.0', port), StreamingHandler)
    print("MJPEG server v2 started on http://0.0.0.0:" + str(port) + "/stream")
    print("Access URL: http://192.168.3.215:" + str(port) + "/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped")
        server.shutdown()
EOF

# 启动
nohup python3 /tmp/mjpeg_server_v2.py > /tmp/mjpeg_v2.log 2>&1 &
sleep 2

# 检查
echo "=== 进程状态 ==="
ps aux | grep mjpeg_server | grep -v grep
echo ""
echo "=== 日志输出 ==="
cat /tmp/mjpeg_v2.log
