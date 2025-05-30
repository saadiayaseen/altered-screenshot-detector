import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from transformers import pipeline
import torch

# Setup Flask app and config
app = Flask(__name__)
app.config['SECRET_KEY'] = '84280407586740552894079845002389'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Detection History model
class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    diffusion_result = db.Column(db.String(300))
    roberta_result = db.Column(db.String(300))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load RoBERTa sentiment-analysis pipeline once
roberta_classifier = pipeline("sentiment-analysis", model="roberta-base", device=0 if torch.cuda.is_available() else -1)

# AI model functions
def run_diffusion_forensics(image_path):
    try:
        img = Image.open(image_path).convert("L")
        pixels = list(img.getdata())
        variance = torch.tensor(pixels).float().var().item()
        if variance > 1000:
            return "Fake (High variance)"
        else:
            return "Real (Low variance)"
    except Exception as e:
        return f"Error analyzing image: {e}"

def run_roberta_analysis(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            return "No text detected"
        results = roberta_classifier(text[:512])
        label = results[0]['label']
        if label == 'NEGATIVE':
            return "Suspicious Text Detected"
        else:
            return "Clean Text"
    except Exception as e:
        return f"Text analysis error: {e}"

# Routes

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email address already registered')
            return redirect(url_for('register'))
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('upload'))
        else:
            flash('Invalid login credentials')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('welcome'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            diffusion_result = run_diffusion_forensics(save_path)
            roberta_result = run_roberta_analysis(save_path)

            # Save detection history
            new_history = DetectionHistory(user_id=current_user.id, filename=filename,
                                           diffusion_result=diffusion_result, roberta_result=roberta_result)
            db.session.add(new_history)
            db.session.commit()

            return render_template('result.html', diffusion=diffusion_result,
                                   roberta=roberta_result, filename=filename)
    return render_template('upload.html')

@app.route('/history')
@login_required
def history():
    records = DetectionHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', records=records)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    