from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import threading
from werkzeug.utils import secure_filename
from cart2 import video_to_cartoon


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

processing_status = {}


# перевірка дозволених розширень файлів
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# обгортка  з можливістю відстеження прогресу
def video_to_cartoon_wrapper(input_path, output_path, model_path, filename):
    global processing_status
    print(f"Начало обработки {filename}")
    processing_status[filename] = 'processing'
    def update_progress(progress):
        global processing_status
        processing_status[filename] = f'{progress}%'

    video_to_cartoon(input_path, output_path, model_path, update_progress=update_progress)
    processing_status[filename] = 'done'
    print(f"Оюробку {filename} завершено")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files: # чи є файл відео у запиті
        return 'No file part', 400
    file = request.files['video'] #  файл з запиту
    if file.filename == '': # чи було вибрано файл
        return 'No selected file', 400
    if file and allowed_file(file.filename): #чи файл має дозволене розширення
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)# збереження файлу

        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        thread = threading.Thread(target=video_to_cartoon_wrapper, args=(save_path, processed_path, 'saved_models', processed_filename))

        thread.start()
        return 'Відео завантажене і знаходиться в обробці.', 202
    else:
        return 'Тип файла не ідтримується.', 400


@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/status/<filename>')
def check_status(filename):
    status = processing_status.get(filename, 'not_found')
    return {'status': status}


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    app.run(debug=True)
