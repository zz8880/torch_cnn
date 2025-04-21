import requests
import json

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:8b",  # 你已经 pull 的模型名称
    "prompt": "请解释什么是量子纠缠",
    "stream": False       # 若为 True，则会流式返回
}

response = requests.post(url, json=payload)
print(response.json()["response"])
