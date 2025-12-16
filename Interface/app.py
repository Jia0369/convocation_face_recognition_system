import os
import json
import qrcode
from PIL import Image
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, flash, url_for, jsonify
from werkzeug.utils import secure_filename
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = "supersecretkey"

# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = r"C:\Users\USER\Documents\image_assign"
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
STATIC_UPLOAD_FOLDER = "static/uploads"
QR_FOLDER = os.path.join(BASE_DIR, "static", "qr")
JSON_FILE = os.path.join(BASE_DIR, "students.json")
MODEL_PATH = os.path.join(BASE_DIR, "facenet_embed_classifier.h5")
DATABASE_EMBEDDING_FILE = os.path.join(BASE_DIR, "embeddings", "flask_app_embeddings.pkl")
IMG_SIZE = (160, 160)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Known courses for dropdown
KNOWN_COURSES = [
    "BSc (Hons) Computer Science",
    "BSc (Hons) Information Systems",
    "Diploma in Computer Science",
    "BEng (Hons) Electrical & Electronic",
    "BEng (Hons) Mechanical",
    "BBA (Hons)",
    "BCom (Hons) Accounting",
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QR_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Models (detector + embedder + classifier)
# -----------------------------
try:
    detector = MTCNN()
    embedder = FaceNet()
    classifier = load_model(MODEL_PATH)
    print("✅ Models initialized.")
except Exception as e:
    print(f"❌ Error initializing models: {e}")
    raise

# -----------------------------
# Helpers
# -----------------------------
def l2n(v, eps=1e-10):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return v / n

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_students():
    if not os.path.exists(JSON_FILE):
        print(f"ℹ️ No students.json found at {JSON_FILE}. Returning empty list.")
        return []
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"ℹ️ students.json is empty at {JSON_FILE}. Returning empty list.")
                return []
            print(f"📄 students.json content: {content}")
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in students.json: {e}. Content: '{content}'")
        print(f"⚠️ Check for invalid characters, encoding issues, or file corruption. Returning empty list.")
        return []
    except UnicodeDecodeError as e:
        print(f"❌ Encoding error in students.json: {e}. Content may contain invalid characters.")
        print(f"⚠️ Ensure file is saved with UTF-8 encoding. Returning empty list.")
        return []
    except Exception as e:
        print(f"❌ Unexpected error reading students.json: {e}. Returning empty list.")
        return []

def save_students(data):
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved students.json with {len(data)} students.")

# -----------------------------
# Embedding DB I/O (FaceNet)
# -----------------------------
def load_database():
    if not os.path.exists(DATABASE_EMBEDDING_FILE):
        print(f"❌ No database file found at {DATABASE_EMBEDDING_FILE}. Please ensure it exists.")
        raise FileNotFoundError(f"Embedding database {DATABASE_EMBEDDING_FILE} not found.")
    with open(DATABASE_EMBEDDING_FILE, "rb") as f:
        db = pickle.load(f)
    db["embeddings"] = list(db.get("embeddings", []))
    db["labels"] = list(db.get("labels", []))
    db["label_map"] = dict(db.get("label_map", {}))
    db["embeddings"] = [l2n(e) for e in db["embeddings"]]
    return db

DB = load_database()

# -----------------------------
# Face pipeline (MTCNN + FaceNet)
# -----------------------------
def _detect_and_crop_rgb(frame_bgr, min_conf=0.9):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dets = detector.detect_faces(rgb)
    if not dets:
        return None
    det = max(dets, key=lambda d: d.get("confidence", 0))
    if det.get("confidence", 0) < min_conf:
        return None
    x, y, w, h = det["box"]
    x, y = max(0, x), max(0, y)
    crop = rgb[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, IMG_SIZE)
    return crop

def embed_face_rgb(rgb_160):
    emb = embedder.embeddings(np.array([rgb_160]))[0]
    return l2n(emb)

# -----------------------------
# Recognition
# -----------------------------
def recognize_face(frame_bgr, clf_threshold=0.18):
    try:
        face_rgb = _detect_and_crop_rgb(frame_bgr, min_conf=0.9)
        if face_rgb is None:
            return "No face", 0.0

        q = embed_face_rgb(face_rgb)
        preds = classifier.predict(np.array([q]), verbose=0)
        c = int(np.argmax(preds[0]))
        conf = float(preds[0][c])
        if conf >= clf_threshold:
            best_name = DB["label_map"].get(c, f"class_{c}")
            return best_name, conf
        return "Unknown", conf

    except Exception as e:
        print(f"Face recognition error: {str(e)}")
        return "No face", 0.0

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    student_id = request.form["student_id"]
    password = request.form["password"]
    students = load_students()
    print(f"🔍 Attempting login for student_id: {student_id}, password: {password}")
    print(f"📋 Loaded students: {students}")
    student = next((s for s in students
                    if s["student_id"].lower() == student_id.lower()
                    and s["password"] == password), None)
    if student:
        print(f"✅ Login successful for {student_id}. Redirecting to profile.")
        return redirect(url_for("profile", student_id=student_id))
    else:
        print(f"❌ Login failed for {student_id}. Invalid ID or password.")
        flash("Invalid Student ID or Password.")
        return redirect("/")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        photo = request.files.get("photo")
        course = (request.form.get("course") or "").strip()
        course_other = (request.form.get("course_other") or "").strip()

        if not course:
            flash("Please select your course.")
            return redirect(request.url)
        if course == "Other" and not course_other:
            flash("Please specify your course in the text box.")
            return redirect(request.url)

        final_course = course_other if course == "Other" else course

        if not photo or photo.filename == "":
            flash("No photo selected.")
            return redirect(request.url)

        if not allowed_file(photo.filename):
            flash("Invalid file type. Please upload an image.")
            return redirect(request.url)

        students = load_students()
        if students:
            last_id = students[-1]['student_id']
            next_number = int(last_id[1:]) + 1
        else:
            next_number = 1

        student_id = f"S{next_number:03d}"
        filename = secure_filename(f"{student_id}_{photo.filename}")
        photo_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        photo.save(photo_path)

        new_student = {
            "student_id": student_id,
            "name": name,
            "email": email,
            "password": password,
            "course": final_course,
            "photo": os.path.join(STATIC_UPLOAD_FOLDER, filename).replace('\\', '/')
        }
        students.append(new_student)
        save_students(students)
        flash(f"Congratulations {name} for graduating from {final_course}")
        flash("Registration successful! Your Student ID is: " + student_id)
        return render_template("register.html", success=True, student_id=student_id, name=name, course=final_course, KNOWN_COURSES=KNOWN_COURSES)

    return render_template("register.html", KNOWN_COURSES=KNOWN_COURSES)

@app.route("/scan")
def to_scanPage():
    return render_template("scan.html")

@app.route("/face_scan/<student_id>")
def face_scan(student_id):
    students = load_students()
    student = next((s for s in students if s["student_id"].lower() == student_id.lower()), None)
    if not student:
        flash("Student not found.")
        return redirect(url_for("home"))
    return render_template("jiawei_scan.html", student_id=student_id)

@app.route("/profile/<student_id>", methods=["GET", "POST"])
def profile(student_id):
    students = load_students()
    student = next((s for s in students if s["student_id"] == student_id), None)
    if not student:
        return "Student not found", 404

    if request.method == "POST":
        student["name"] = request.form["name"]
        student["email"] = request.form["email"]
        student["password"] = request.form["password"]
        course = (request.form.get("course") or "").strip()
        course_other = (request.form.get("course_other") or "").strip()
        if course == "Other" and course_other:
            student["course"] = course_other
        else:
            if course:
                student["course"] = course

        photo = request.files.get("photo")
        if photo and allowed_file(photo.filename):
            filename = secure_filename(f"{student_id}_{photo.filename}")
            photo_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            photo.save(photo_path)
            student["photo"] = os.path.join(STATIC_UPLOAD_FOLDER, filename).replace('\\', '/')

        save_students(students)
        flash("Profile updated successfully.")
        return redirect(url_for("profile", student_id=student_id))

    qr_data = f"{student['student_id']}|{student['name']}"
    qr = qrcode.QRCode(
        version=5,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_path = os.path.join(QR_FOLDER, f"{student_id}_qr.png")
    qr_img.save(qr_path)
    qr_url = f"qr/{student_id}_qr.png"
    return render_template("profile.html", student=student, qr_code_url=qr_url, KNOWN_COURSES=KNOWN_COURSES)

@app.route("/api/validate_qr", methods=["POST"])
def validate_qr():
    data = request.get_json(silent=True) or {}
    qr_text = (data.get("qr") or "").strip()
    if not qr_text:
        return jsonify({"ok": False, "student": None, "message": "No QR data provided."}), 400

    parts = qr_text.split("|")
    if len(parts) != 2:
        return jsonify({"ok": False, "student": None, "message": "Invalid QR format. Expected 'ID|Name'."}), 400

    qr_id, qr_name = parts[0].strip(), parts[1].strip()
    students = load_students()
    student = next(
        (s for s in students
         if s.get("student_id", "").strip().lower() == qr_id.lower()
         and s.get("name", "").strip().lower() == qr_name.lower()),
        None
    )

    if student:
        safe_student = {
            "student_id": student["student_id"],
            "name": student["name"],
            "email": student.get("email", ""),
            "photo": student.get("photo", ""),
            "course": student.get("course", "")
        }
        return jsonify({"ok": True, "student": safe_student, "message": "QR verified."})
    else:
        return jsonify({"ok": False, "student": None, "message": "Student not found or mismatch."}), 404

@app.route("/api/validate_face/<student_id>", methods=["POST"])
def validate_face(student_id):
    print(f"📥 /api/validate_face/{student_id}")
    students = load_students()
    student = next((s for s in students if s["student_id"].lower() == student_id.lower()), None)
    if not student:
        return jsonify({"ok": False, "name": "Unknown", "message": "Student not found."}), 404

    data = request.get_json(silent=True) or {}
    frame_image_data = data.get("frame_image")
    if not frame_image_data:
        return jsonify({"ok": False, "name": "Unknown", "message": "No frame image provided."}), 400

    try:
        b64 = frame_image_data.split(",", 1)[1]
        frame_image_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(frame_image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_face_{student_id}.jpg")
        cv2.imwrite(debug_path, frame_bgr)

        predicted_id, score = recognize_face(frame_bgr)
        print(f"→ predicted: {predicted_id} (score {score:.3f}) vs student_id: {student_id}")

        ok = (predicted_id.lower() == student_id.lower() and predicted_id != "Unknown")

        if ok:
            display_name = student["name"]
            course = student.get("course", "Unknown")
        else:
            rec = next((s for s in students if s["student_id"].lower() == predicted_id.lower()), None)
            display_name = rec["name"] if rec else (predicted_id if predicted_id != "Unknown" else "Unknown")
            course = rec.get("course", "Unknown") if rec else "Unknown"

        response = {
            "ok": ok,
            "predicted_id": predicted_id,
            "name": display_name,
            "message": ("Face verified." if ok else "Face mismatch."),
            "score": float(score)
        }
        if ok:
            response["course"] = course

        return jsonify(response)
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return jsonify({"ok": False, "name": "Unknown", "message": f"Face recognition failed: {str(e)}", "score": 0.0}), 400

@app.route("/test_routes")
def test_routes():
    return jsonify({"routes": [str(rule) for rule in app.url_map.iter_rules()]})

if __name__ == "__main__":
    app.run(debug=True)