from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import uvicorn
from typing import Optional
from pydantic import BaseModel
import os

app = FastAPI(
    title="Spheron GPU Image Generator",
    description="API wrapper for Spheron GPU service with web interface",
    version="1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spheron GPU Configuration
SPHERON_GPU_URL = os.getenv("SPHERON_GPU_URL", "http://provider.spur.gpu3.ai:32333")
TIMEOUT = int(os.getenv("TIMEOUT", "60"))

class GenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "stable-diffusion-xl"
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = 30
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HTML frontend"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spheron Image Generator</title>
        <style>
            /* Your existing CSS */
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Spheron GPU Image Generator</h2>
            <input type="text" id="prompt" placeholder="Describe your image...">
            <button onclick="generateImage()">Generate</button>
            <div id="result"></div>
            <div id="loading" style="display:none;">Generating...</div>
        </div>
        
        <script>
            async function generateImage() {
                const prompt = document.getElementById('prompt').value;
                if (!prompt) return;
                
                document.getElementById('loading').style.display = 'block';
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt })
                    });
                    
                    const data = await response.json();
                    if (data.image) {
                        document.getElementById('result').innerHTML = `
                            <img src="data:image/png;base64,${data.image}" 
                                 style="max-width:100%; margin-top:20px; border-radius:8px;">
                        `;
                    } else {
                        alert('Error: ' + (data.detail || 'Unknown error'));
                    }
                } catch (error) {
                    alert('Error: ' + error);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate image using Spheron GPU service"""
    payload = {
        "prompt": request.prompt,
        "model": request.model,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "negative_prompt": request.negative_prompt,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed
    }
    
    try:
        response = requests.post(
            f"{SPHERON_GPU_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="GPU service timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"GPU service error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Check service health"""
    try:
        response = requests.get(f"{SPHERON_GPU_URL}/health", timeout=5)
        response.raise_for_status()
        return {"status": "healthy", "gpu_service": response.json()}
    except requests.exceptions.RequestException:
        return {"status": "unhealthy", "gpu_service": "unreachable"}

# Ensure this is at the bottom of your file:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
