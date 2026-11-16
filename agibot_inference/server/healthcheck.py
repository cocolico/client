"""只查询模型信息，不发送观测或控制机器人。"""
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments/agibot"))
from _ours_protocol import recv_message, send_message
try:
    with socket.create_connection(("127.0.0.1", 19000), timeout=5) as sock:
        send_message(sock, {"cmd": "info"})
        result = recv_message(sock)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if "error" in result:
            sys.exit(1)
except TimeoutError:
    print("状态查询超时：服务可能正被持续连接的 client 占用，请结合 server.log 查看。", file=sys.stderr)
    sys.exit(2)
except OSError as exc:
    print(f"服务尚未就绪：{exc}", file=sys.stderr)
    sys.exit(1)
