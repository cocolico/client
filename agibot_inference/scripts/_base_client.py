import abc
import logging
import socket
from typing import Optional


class BaseClient(abc.ABC):
    """基础 TCP 客户端。"""

    def __init__(self, ip, port):
        self.logger = self._setup_logger()
        self.socket: Optional[socket.socket] = None

        self._connect(ip, port)

    def __delete__(self, instance):
        self._disconnect()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)

        # 避免创建多个实例时重复添加 Handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)

        return logger

    def _connect(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,
    ) -> None:
        """连接 TCP 服务端。"""
        if self.socket is not None:
            raise RuntimeError("客户端已经连接，请先调用 disconnect()")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            self.logger.info("正在连接 %s:%d", host, port)
            sock.connect((host, port))
            self.socket = sock
            self.logger.info("连接成功")
        except Exception:
            sock.close()
            self.logger.exception("连接 %s:%d 失败", host, port)
            raise

    def _disconnect(self) -> None:
        """断开连接；可重复调用。"""
        if self.socket is None:
            return

        sock = self.socket
        self.socket = None

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            # 对端可能已经关闭连接
            pass
        finally:
            sock.close()
            self.logger.info("连接已断开")

    @abc.abstractmethod
    def policy_infer(
        self,
        observation: dict[str, float],
    ) -> dict[str, float]:
        """根据 observation 执行策略推理。"""
        raise NotImplementedError