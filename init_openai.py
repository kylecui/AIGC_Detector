"""
init_openai.py
全局共享：调用 `from init_openai import client, aclient`
"""
from openai import OpenAI, AsyncOpenAI
import os, httpx
from dotenv import load_dotenv

# AI_Detector API Key
# load api key from .env file via load_dotenv()
load_dotenv()
# 同步 client（保持不变，可选代理）
sync_http_client = httpx.Client(
    # proxies="http://127.0.0.1:7890",   # 没用代理可删
    timeout=30,
)
client = OpenAI(http_client=sync_http_client)

# ★异步 client 需使用 httpx.AsyncClient
async_http_client = httpx.AsyncClient(
    # proxies="http://127.0.0.1:7890",   # 同上
    timeout=30,
)
aclient = AsyncOpenAI(http_client=async_http_client)