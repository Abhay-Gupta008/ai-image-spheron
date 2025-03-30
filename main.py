from flask import Flask, request, jsonify
import requests
import base64
from io import BytesIO

app = Flask(__name__)

# AI Image Generation Function
def generate_image(prompt):
    URL = "http://provider.spur.gpu3.ai:32333/generate"
    payload = {"prompt": str(prompt)}

    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        data = response.json()
        base64_image = data.get("image", "")
        
        if not base64_image:
            return None
        
        if "," in base64_image:
            base64_image = base64_image.split(",")[-1]

        return base64_image
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    image_base64 = generate_image(prompt)
    if image_base64:
        return jsonify({"image": image_base64})
    else:
        return jsonify({"error": "Failed to generate image"}), 500

if __name__ == '__main__':
    # Set host to '0.0.0.0' to allow access from external devices
    app.run(host='0.0.0.0', port=5000)
