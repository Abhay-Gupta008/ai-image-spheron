from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load model
print("Loading Stable Diffusion model...")
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/generate', methods=['POST'])
def generate():
    # Get the prompt from the request JSON
    data = request.json
    prompt = data.get("prompt", "A beautiful sunset over the mountains")

    # Generate image from the model
    print(f"Generating image for prompt: {prompt}")
    image = model(prompt).images[0]

    # Convert the image to base64 for easy transmission
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({"image": img_str})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Change port to 8001 or any other port
