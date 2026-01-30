#import subprocess
#import time
#import os
#import signal
#import requests
#import logging
#import sys
#import atexit
#
## 配置日志
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger("vllm_manager")
#
#class VLLMServiceManager:
#    _instance = None
#    _process = None
#    
#    # === 配置区域 ===
#    MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model/")
#    HOST = "127.0.0.1"
#    PORT = 9000
#    API_BASE = f"http://{HOST}:{PORT}/v1"
#    
#    # [关键修改] 指定隔离环境的 Python 解释器路径
#    # 请根据您实际上一步获取的路径修改这里！
#    VLLM_PYTHON_EXEC = os.environ.get(
#        "VLLM_PYTHON_EXEC", 
#        "/home/yuhongjie/miniconda3/envs/unsloth/bin/python" 
#    )
#
#    # GPU 配置
#    GPU_DEVICES = "2,3" 
#    TENSOR_PARALLEL_SIZE = 2 
#    
#    def __new__(cls):
#        if cls._instance is None:
#            cls._instance = super(VLLMServiceManager, cls).__new__(cls)
#        return cls._instance
#
#    @classmethod
#    def is_running(cls):
#        """检查服务端口是否通畅"""
#        try:
#            resp = requests.get(f"http://{cls.HOST}:{cls.PORT}/health", timeout=1)
#            return resp.status_code == 200
#        except Exception:
#            return False
#
#    @classmethod
#    def start_service(cls):
#        """启动 vLLM 子进程"""
#        if cls.is_running():
#            return
#
#        # 检查 Python 路径是否存在
#        if not os.path.exists(cls.VLLM_PYTHON_EXEC):
#            logger.error(f"Isolated Python environment not found at: {cls.VLLM_PYTHON_EXEC}")
#            raise FileNotFoundError(f"Python interpreter not found: {cls.VLLM_PYTHON_EXEC}")
#
#        logger.info(f"Starting vLLM service using env: {cls.VLLM_PYTHON_EXEC}")
#        logger.info(f"GPUs: {cls.GPU_DEVICES} | Model: {cls.MODEL_PATH}")
#        
#        # 1. 准备环境变量
#        env = os.environ.copy()
#        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
#        env.pop("PYTHONPATH", None)
#
#        # 2. 构造启动命令
#        cmd = [
#            cls.VLLM_PYTHON_EXEC, 
#            "-m", "vllm.entrypoints.openai.api_server",
#            "--model", cls.MODEL_PATH,
#            "--trust-remote-code",
#            "--host", cls.HOST,
#            "--port", str(cls.PORT),
#            "--tensor-parallel-size", str(cls.TENSOR_PARALLEL_SIZE),
#            "--max-model-len", "4096",
#            
#            # === [修复点] 修改为 JSON 字符串格式 ===
#            "--limit-mm-per-prompt", '{"image": 8}', 
#            
#            "--gpu-memory-utilization", "0.9",
#            "--enforce-eager"
#        ]
#
#        # 3. 启动进程
#        cls._process = subprocess.Popen(
#            cmd,
#            env=env,
#            stdout=open("vllm_stdout.log", "w"),
#            stderr=open("vllm_stderr.log", "w"),
#            preexec_fn=os.setsid 
#        )
#
#        logger.info("Waiting for vLLM to load model...")
#        max_retries = 120 
#        
#        for i in range(max_retries):
#            if cls.is_running():
#                logger.info("vLLM service is READY!")
#                return
#            
#            if i % 5 == 0:
#                logger.info(f"Still loading... ({i * 5}s elapsed)")
#            
#            if cls._process.poll() is not None:
#                raise RuntimeError("vLLM process exited unexpectedly! Check vllm_stderr.log.")
#                
#            time.sleep(5)
#        
#        cls.stop_service()
#        raise TimeoutError("vLLM service failed to start within timeout.")
#
#    @classmethod
#    def stop_service(cls):
#        """停止服务"""
#        if cls._process:
#            logger.warning("Auto-cleanup: Stopping vLLM service...")
#            try:
#                os.killpg(os.getpgid(cls._process.pid), signal.SIGTERM)
#                cls._process.wait(timeout=10)
#                logger.info("vLLM service stopped successfully.")
#            except Exception as e:
#                logger.error(f"Error stopping vLLM: {e}")
#            finally:
#                cls._process = None
#
#vllm_manager = VLLMServiceManager()
#
#@atexit.register
#def _auto_cleanup():
#    vllm_manager.stop_service()
import subprocess
import time
import os
import signal
import requests
import logging
import sys
import atexit

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm_manager")

class VLLMServiceManager:
    _instance = None
    _process = None
    
    # === 配置区域 ===
    MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "/data1/yuhongjie2/sft/qwen3vl_8b_4bit_finetune/merged_model/")
    HOST = "127.0.0.1"
    PORT = 9000
    API_BASE = f"http://{HOST}:{PORT}/v1"
    
    # 指定隔离环境的 Python 解释器路径
    VLLM_PYTHON_EXEC = os.environ.get(
        "VLLM_PYTHON_EXEC", 
        "/home/yuhongjie/miniconda3/envs/unsloth/bin/python" 
    )

    # GPU 配置
    GPU_DEVICES = "2,3" 
    TENSOR_PARALLEL_SIZE = 2 
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLLMServiceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def is_running(cls):
        """检查服务端口是否通畅"""
        try:
            resp = requests.get(f"http://{cls.HOST}:{cls.PORT}/health", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def start_service(cls):
        """启动 vLLM 子进程"""
        if cls.is_running():
            return

        # 检查 Python 路径是否存在
        if not os.path.exists(cls.VLLM_PYTHON_EXEC):
            logger.error(f"Isolated Python environment not found at: {cls.VLLM_PYTHON_EXEC}")
            raise FileNotFoundError(f"Python interpreter not found: {cls.VLLM_PYTHON_EXEC}")

        logger.info(f"Starting vLLM service using env: {cls.VLLM_PYTHON_EXEC}")
        logger.info(f"GPUs: {cls.GPU_DEVICES} | Model: {cls.MODEL_PATH}")
        
        # 1. 准备环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cls.GPU_DEVICES
        env.pop("PYTHONPATH", None)

        # 2. 构造启动命令
        cmd = [
            cls.VLLM_PYTHON_EXEC, 
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", cls.MODEL_PATH,
            "--trust-remote-code",
            "--host", cls.HOST,
            "--port", str(cls.PORT),
            "--tensor-parallel-size", str(cls.TENSOR_PARALLEL_SIZE),
            "--max-model-len", "4096",
            "--limit-mm-per-prompt", '{"image": 8}', 
            "--gpu-memory-utilization", "0.9",
            "--enforce-eager"
        ]

        # 3. 启动进程
        # [核心修改] 去掉了 preexec_fn=os.setsid
        # 这样 vLLM 会在同一个进程组，Ctrl+C 会直接传导给它
        cls._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=open("vllm_stdout.log", "w"),
            stderr=open("vllm_stderr.log", "w")
            # 删除了 preexec_fn=os.setsid
        )

        logger.info("Waiting for vLLM to load model...")
        max_retries = 120 
        
        for i in range(max_retries):
            if cls.is_running():
                logger.info("vLLM service is READY!")
                return
            
            if i % 5 == 0:
                logger.info(f"Still loading... ({i * 5}s elapsed)")
            
            if cls._process.poll() is not None:
                raise RuntimeError("vLLM process exited unexpectedly! Check vllm_stderr.log.")
                
            time.sleep(5)
        
        cls.stop_service()
        raise TimeoutError("vLLM service failed to start within timeout.")

    @classmethod
    def stop_service(cls):
        """停止服务"""
        if cls._process:
            logger.warning("Auto-cleanup: Stopping vLLM service...")
            try:
                # [核心修改] 既然没有 setsid，直接 terminate 子进程对象即可
                # 这种方式更温和，也更不容易出错
                cls._process.terminate()
                
                # 等待进程实际退出，避免僵尸进程
                try:
                    cls._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM didn't exit in time, force killing...")
                    cls._process.kill() # 强制击杀
                    cls._process.wait()
                
                logger.info("vLLM service stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping vLLM: {e}")
            finally:
                cls._process = None

vllm_manager = VLLMServiceManager()

@atexit.register
def _auto_cleanup():
    vllm_manager.stop_service()