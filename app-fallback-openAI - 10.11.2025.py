from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI


# Încarcă variabilele de mediu
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# __name__ este o variabilă specială în Python care conține numele modulului curent.
# Dacă fișierul este rulat direct (de exemplu python app-fallback-openAI.py), atunci __name__ == "__main__".
# Flask folosește această informație pentru a ști unde să caute resursele (fișiere statice, șabloane etc.).

app = Flask(__name__)

# Foldere pentru imagini
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Client OpenAI (dacă există cheia)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# Ruta principală – pagina HTML
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint pentru upload și procesare imagine
@app.route("/upload", methods=["POST"])
def upload():
    #  un obiect de tip FileStorage, adică un flux binar — nu o imagine încă!
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Nu a fost încărcată nicio imagine"}), 400

    # file.read() -> Citește tot conținutul fișierului în memorie, sub formă de bytes (un șir binar, adică 0 și 1). file.read()
    # np.frombuffer(..., np.uint8) -> Transformă acei bytes într-un vector NumPy de tip uint8 (numere întregi între 0 și 255). Acum avem o reprezentare numerică a datelor imaginii, dar încă nu este o imagine 2D — doar o listă lungă de numere.
    # cv2.imdecode(..., cv2.IMREAD_COLOR) -> OpenCV ia vectorul de date binare (array-ul de mai sus) și îl decodează într-o imagine color (matrice 3D de pixeli).
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # 10.11.2025 - imbunatatire cod

    # Convertim imaginea în HSV pentru a separa mai bine culorile
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definim gamele de culori pentru roșu, verde, albastru
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1),
                    cv2.inRange(hsv, lower_red2, upper_red2))

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combinăm toate măștile
    mask_total = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_blue))

    # Curățăm zgomotul cu morfologie
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Găsim contururile pe masca totală
    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    # Creează un dicționar Python pentru a număra fiecare tip de formă.
    shapes = {"triangle": 0, "square": 0, "circle": 0}
    
    # contours este lista tuturor formelor detectate în imagine (obținută cu cv2.findContours).
    # Fiecare cnt (contour) este o listă de coordonate (puncte x,y) care descriu marginea unei forme.
    # cv2.approxPolyDP() simplifică forma conturului pentru a afla câte „colțuri” are.
    # cv2.arcLength(cnt, True) – lungimea perimetrului conturului.
    # 0.04 * ... – precizia (toleranța). Cu cât valoarea este mai mică, cu atât forma este mai exactă. 0.04 înseamnă o toleranță de 4% din lungimea totală.
    # True – indică faptul că forma este închisă (nu o linie deschisă).
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            shapes["triangle"] += 1
        elif len(approx) == 4:
            shapes["square"] += 1
        else:
            shapes["circle"] += 1
        # Desenează contururile
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)

    # Salvează imaginea procesată în static/uploads
    save_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(save_path, img)

    # Fallback GPT-4V (doar dacă există client OpenAI)
    gpt_result = None
    if client:
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
        "processed_image": "/static/uploads/result.jpg",
        "gpt_result": gpt_result
    })

if __name__ == "__main__":
    app.run(debug=True, port=9000)
