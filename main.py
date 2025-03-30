from fastapi import FastAPI, HTTPException
import requests
import os

app = FastAPI()

SPHERON_GPU_URL = os.getenv("SPHERON_GPU_URL", "http://provider.spur.gpu3.ai:32333")

@app.get("/")
def home():
    return {"status": "ready", "gpu_service": SPHERON_GPU_URL}

@app.post("/generate")
async def generate_image(prompt: str):
    try:
        response = requests.post(
            f"{SPHERON_GPU_URL}/generate",
            json={"prompt": prompt, "width": 512, "height": 512, "steps": 50},
            timeout=60
        )
        return response.json()
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
