from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ruta principală – pagina HTML
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint pentru upload și procesare imagine
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Nu a fost încărcată nicio imagine"}), 400

    # Citește imaginea
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Conversie în gri + binarizare
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Detectare contururi
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
 
        # Desenează conturul pe imagine
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)

    # Salvează imaginea procesată
    save_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(save_path, img)

    return jsonify({
        "shapes": shapes,
        "processed_image": save_path
    })


if __name__ == "__main__":
    app.run(debug=True, port=8080)
