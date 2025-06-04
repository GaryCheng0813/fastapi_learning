from fastapi import FastAPI
from fastapi.responses import Response
import requests
import os
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 推荐将 token 放到环境变量里，安全性更高
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class Message(BaseModel):
    role: str = "user"
    content: str

class PromptRequest(BaseModel):
    messages: List[Message]
    model: str = "accounts/fireworks/models/deepseek-r1-0528"

class ResponseMessage(BaseModel):
    content: str

@app.post("/generate", response_model=ResponseMessage)
async def generate_text(payload: PromptRequest):
    prompt = payload.messages[0].content if payload.messages else ""
    if len(prompt) > 200:
        return Response(content="Prompt too long.", status_code=400)
    
    # 直接用 payload.dict() 作为请求体
    response = requests.post(API_URL, headers=headers, json=payload.model_dump())
    if response.status_code != 200:
        return Response(content="Error in response from API.", status_code=response.status_code)
    response_data = response.json()
    # 健壮性检查
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = str(response_data)
    return ResponseMessage(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)