import os
import io
import torch
from fastapi import FastAPI, Form, Response
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionPipeline

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Server is running!"}

MODEL_CACHE = os.path.join(os.getcwd(), "model_cache")  # Ensures correct path

@app.on_event("startup")
async def load_model():
    global pipe
    
    os.makedirs(MODEL_CACHE, exist_ok=True)
    
    # Load Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe.enable_attention_slicing()

    # Handle CPU/GPU optimization
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention()
        try:
            from accelerate import __version__  # Check if accelerate is installed
            pipe.enable_model_cpu_offload()
        except ImportError:
            print("[WARNING] `accelerate` not installed. Skipping CPU offload.")
    else:
        pipe.enable_sequential_cpu_offload()

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    steps: int = Form(20)
):
    try:
        steps = max(10, min(steps, 30))  # Limit steps between 10-30
        
        # Generate the image
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "filename=generated_image.png"}
        )
    except Exception as e:
        return Response(
            content=f"Error: {str(e)}",
            status_code=500,
            media_type="text/plain"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
