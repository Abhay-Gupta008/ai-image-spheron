from fastapi import FastAPI, Form, Response
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionXLPipeline
import torch
import io
import os

app = FastAPI()

# Optimized model loading (critical for Spheron)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,  # FP16 quantization (50% memory reduction)
    variant="fp16",             # Pre-quantized weights
    use_safetensors=True
)

# Memory optimizations (must-have for free tier)
pipe.enable_attention_slicing()  # Processes image in chunks
pipe.enable_model_cpu_offload()  # Swaps unused layers to disk

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body style="font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>Spheron AI Image Generator</h1>
            <form action="/generate" method="post">
                <textarea name="prompt" rows="4" style="width: 100%;" 
                    placeholder="A realistic portrait of a cyberpunk cat..." required></textarea><br>
                <div style="margin: 10px 0;">
                    <label>Steps: <input type="range" name="steps" min="10" max="30" value="20"></label>
                    <span id="steps-value">20</span>
                </div>
                <button type="submit" style="padding: 8px 16px;">Generate</button>
            </form>
            <script>
                document.querySelector('input[name="steps"]').addEventListener('input', (e) => {
                    document.getElementById('steps-value').textContent = e.target.value;
                });
            </script>
        </body>
    </html>
    """

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    steps: int = Form(20)
):
    try:
        # Generate image with safety limits
        image = pipe(
            prompt=prompt,
            num_inference_steps=min(steps, 30),  # Prevent excessive steps
            guidance_scale=7.5,
            height=512,  # Lower res for stability
            width=512
        ).images[0]
        
        # Convert to PNG in memory
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f"filename={prompt[:30]}.png"}
        )
    except Exception as e:
        return Response(
            content=f"Error: {str(e)}",
            status_code=500
        )

# Health check endpoint for Spheron
@app.get("/health")
async def health_check():
    return {"status": "OK", "memory": f"{os.popen('free -m').read()}"}