from fastapi import FastAPI, Form
from fastapi.responses import Response
from transformers import pipeline

app = FastAPI()

# 只加载一次模型
generator = pipeline("text-generation", model="gpt2")

@app.post("/generate")
async def generate_text(prompt: str = Form(...)):
    # 限制 prompt 长度，防止滥用
    if len(prompt) > 200:
        return Response(content="Prompt too long.", status_code=400)
    generated_text = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    return Response(content=generated_text, media_type="text/plain")