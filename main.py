from fastapi import FastAPI, HTTPException
import requests
import os
from fastapi.responses import StreamingResponse
from io import BytesIO

app = FastAPI()

# Fetch the Spheron GPU URL from environment variable or use default
SPHERON_GPU_URL = os.getenv("SPHERON_GPU_URL", "http://provider.spur.gpu3.ai:32333")

@app.get("/")
def home():
    return {"status": "ready", "gpu_service": SPHERON_GPU_URL}

@app.post("/generate")
async def generate_image(prompt: str):
    try:
        # Request to the Spheron-based AI image generation API
        response = requests.post(
            f"{SPHERON_GPU_URL}/api/generate",  # Correct the URL if needed
            json={"prompt": prompt, "size": "512x512", "steps": 50},  # You may adjust params
            timeout=60
        )
        response.raise_for_status()  # Raise an error if the response was bad

        # Assuming response content is image data (binary)
        image_data = response.content

        # Convert the binary content into a streaming response for the client
        image_stream = BytesIO(image_data)
        return StreamingResponse(image_stream, media_type="image/png")

    except requests.exceptions.RequestException as e:
        # If an error occurs while contacting the Spheron service
        raise HTTPException(status_code=500, detail=f"Error contacting Spheron service: {str(e)}")
    except Exception as e:
        # General error handling
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
