from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import base64
from openai import OpenAI

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Incarcă cheia OpenAI din mediu
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Nu a fost încărcată nicio imagine"}), 400

    # Citește imaginea
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Fallback local OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = {"triangle": 0, "square": 0, "circle": 0}
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            shapes["triangle"] += 1
        elif len(approx) == 4:
            shapes["square"] += 1
        else:
            shapes["circle"] += 1
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)

    save_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(save_path, img)

    # Dacă ai cheie OpenAI, trimite imaginea la GPT-4V
    gpt_result = None
    if client.api_key:
        try:
            with open(save_path, "rb") as f:
                image_bytes = f.read()
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # sau gpt-4v dacă ai acces
                messages=[
                    {"role": "system", "content": "Ești un profesor care recunoaște forme geometrice din imagini."},
                    {"role": "user", "content": "Numără câte triunghiuri, pătrate și cercuri sunt în această imagine.", "image": image_bytes}
                ]
            )
            gpt_result = response.choices[0].message.content
        except Exception as e:
            gpt_result = f"OpenAI fallback error: {str(e)}"

    return jsonify({
        "shapes": shapes,
        "processed_image": save_path,
        "gpt_result": gpt_result
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
