# ============================================
# SKIN CANCER DETECTION APP
# Flask + TensorFlow/Keras + MySQL
# ============================================

import os
import json
import numpy as np
import mysql.connector
from flask import (Flask, render_template, request,
                   redirect, url_for, session, flash)
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ============================================
# CONFIGURATION FLASK
# ============================================

app = Flask(__name__)
app.secret_key = 'skin_cancer_secret_key_2024'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ============================================
# CHARGEMENT DU MODÈLE VGG16
# ============================================

MODEL_PATH = os.path.join('model', 'vgg16_malignant_vs_benign.h5')

print("⏳ Chargement du modèle IA...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé avec succès !")

# ============================================
# CONNEXION MYSQL
# ============================================

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        port=3307,
        user='root',
        password='',
        database='skin_cancer_db'
    )

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_image(img_path):
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(img_path):
    img_array  = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        label      = 'Malignant'
        confidence = round(float(prediction) * 100, 2)
    else:
        label      = 'Benign'
        confidence = round((1 - float(prediction)) * 100, 2)
    return label, confidence


def save_notes(filename, notes_dict):
    """Sauvegarde les notes cliniques dans un fichier JSON."""
    base        = os.path.splitext(filename)[0]
    notes_path  = os.path.join(UPLOAD_FOLDER, base + '_notes.json')
    with open(notes_path, 'w', encoding='utf-8') as f:
        json.dump(notes_dict, f, ensure_ascii=False, indent=2)


def load_notes(filename):
    """Charge les notes cliniques depuis le fichier JSON."""
    base       = os.path.splitext(filename)[0]
    notes_path = os.path.join(UPLOAD_FOLDER, base + '_notes.json')
    if os.path.exists(notes_path):
        with open(notes_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# ROUTES
# ============================================

# ---------- LOGIN ----------
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Veuillez remplir tous les champs.', 'danger')
            return render_template('login.html')

        try:
            conn   = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                'SELECT * FROM users WHERE username = %s AND password = %s',
                (username, password)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                session['user_id']   = user['id']
                session['username']  = user['username']
                session['full_name'] = user['username']
                flash(f"Bienvenue, {user['username']} !", 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Identifiants incorrects.', 'danger')

        except mysql.connector.Error as e:
            flash(f'Erreur DB : {e}', 'danger')

    return render_template('login.html')


# ---------- DASHBOARD ----------
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute('SELECT COUNT(*) AS total FROM patients')
        total = cursor.fetchone()['total']

        cursor.execute(
            "SELECT COUNT(*) AS total FROM patients WHERE result = 'Malignant'"
        )
        malignant = cursor.fetchone()['total']

        cursor.execute(
            "SELECT COUNT(*) AS total FROM patients WHERE result = 'Benign'"
        )
        benign = cursor.fetchone()['total']

        cursor.execute(
            'SELECT * FROM patients ORDER BY created_at DESC LIMIT 5'
        )
        recent = cursor.fetchall()

        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        flash(f'Erreur base de données : {e}', 'danger')
        total = malignant = benign = 0
        recent = []

    return render_template('dashboard.html',
                           total=total,
                           malignant=malignant,
                           benign=benign,
                           recent=recent)


# ---------- PREDICT ----------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':

        # Données patient
        patient_name = request.form.get('patient_name', '').strip()
        age          = request.form.get('age', '').strip()

        # Notes cliniques
        localisation = request.form.get('localisation', '').strip()
        width        = request.form.get('width', '').strip()
        since_when   = request.form.get('since_when', '').strip()
        extra_notes  = request.form.get('extra_notes', '').strip()

        if not patient_name:
            flash('Veuillez entrer le nom du patient.', 'danger')
            return render_template('predict.html')

        if 'image' not in request.files:
            flash('Aucun fichier sélectionné.', 'danger')
            return render_template('predict.html')

        file = request.files['image']

        if file.filename == '':
            flash('Aucun fichier sélectionné.', 'danger')
            return render_template('predict.html')

        if not allowed_file(file.filename):
            flash('Format non autorisé. Utilisez PNG, JPG ou JPEG.', 'danger')
            return render_template('predict.html')

        # Sauvegarde image
        filename  = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Prédiction IA
        try:
            label, confidence = predict_image(save_path)
        except Exception as e:
            flash(f'Erreur lors de la prédiction : {e}', 'danger')
            return render_template('predict.html')

        # Sauvegarde en base de données (avec age)
        try:
            conn   = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO patients
                   (name, age, image_path, result, probability)
                   VALUES (%s, %s, %s, %s, %s)''',
                (patient_name,
                 int(age) if age.isdigit() else None,
                 filename,
                 label,
                 confidence)
            )
            conn.commit()
            cursor.close()
            conn.close()

        except mysql.connector.Error as e:
            flash(f'Erreur sauvegarde : {e}', 'danger')
            return render_template('predict.html')

        # Sauvegarde notes cliniques en JSON
        save_notes(filename, {
            'localisation': localisation,
            'width':        width,
            'since_when':   since_when,
            'extra_notes':  extra_notes
        })

        return redirect(url_for('result',
                                patient_name=patient_name,
                                age=age,
                                label=label,
                                confidence=confidence,
                                image_file=filename,
                                localisation=localisation,
                                width=width,
                                since_when=since_when,
                                extra_notes=extra_notes))

    return render_template('predict.html')


# ---------- RESULT ----------
@app.route('/result')
@login_required
def result():
    patient_name = request.args.get('patient_name', 'Inconnu')
    age          = request.args.get('age', '')
    label        = request.args.get('label', 'N/A')
    confidence   = request.args.get('confidence', '0')
    image_file   = request.args.get('image_file', '')
    localisation = request.args.get('localisation', '')
    width        = request.args.get('width', '')
    since_when   = request.args.get('since_when', '')
    extra_notes  = request.args.get('extra_notes', '')

    return render_template('result.html',
                           patient_name=patient_name,
                           age=age,
                           label=label,
                           confidence=float(confidence),
                           image_file=image_file,
                           localisation=localisation,
                           width=width,
                           since_when=since_when,
                           extra_notes=extra_notes)


# ---------- PATIENTS ----------
@app.route('/patients')
@login_required
def patients():
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT * FROM patients ORDER BY created_at DESC'
        )
        all_patients = cursor.fetchall()
        cursor.close()
        conn.close()

        # Charger les notes cliniques pour chaque patient
        for p in all_patients:
            p['notes'] = load_notes(p['image_path'])

    except mysql.connector.Error as e:
        flash(f'Erreur base de données : {e}', 'danger')
        all_patients = []

    return render_template('patients.html', patients=all_patients)

# ---------- DELETE PATIENT ----------
@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    """Supprime un patient et son image + notes JSON."""
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Récupérer le nom de l'image avant suppression
        cursor.execute(
            'SELECT image_path FROM patients WHERE id = %s', (patient_id,)
        )
        patient = cursor.fetchone()

        if patient:
            # Supprimer l'image uploadée
            img_path = os.path.join(
                app.config['UPLOAD_FOLDER'], patient['image_path']
            )
            if os.path.exists(img_path):
                os.remove(img_path)

            # Supprimer le fichier JSON des notes
            base       = os.path.splitext(patient['image_path'])[0]
            notes_path = os.path.join(
                app.config['UPLOAD_FOLDER'], base + '_notes.json'
            )
            if os.path.exists(notes_path):
                os.remove(notes_path)

            # Supprimer de la base de données
            cursor.execute(
                'DELETE FROM patients WHERE id = %s', (patient_id,)
            )
            conn.commit()
            flash('Patient supprimé avec succès.', 'success')
        else:
            flash('Patient introuvable.', 'danger')

        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        flash(f'Erreur suppression : {e}', 'danger')

    return redirect(url_for('patients'))
# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.clear()
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('login'))


# ============================================
# LANCEMENT
# ============================================

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)