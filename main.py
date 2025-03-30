from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load AI model
print("Loading Stable Diffusion model...")
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/generate', methods=['POST'])
def generate():
    """Generate an image based on the given text prompt"""
    data = request.json
    prompt = data.get("prompt", "A futuristic city at night")

    print(f"Generating image for prompt: {prompt}")
    image = model(prompt).images[0]

    # Convert image to base64 format for easy transmission
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
