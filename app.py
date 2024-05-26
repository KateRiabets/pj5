from flask import Flask, request, render_template, send_from_directory
import os
import threading
from werkzeug.utils import secure_filename
from model import extract_frames, process_images, frames_to_video, add_audio_to_video, clean_temp_folders

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Директорія для завантажених файлів
PROCESSED_FOLDER = 'processed'  # Директорія для оброблених файлів
MODEL_PATH = 'saved_models/unet_model.h5'  # Укажіть правильний шлях до моделі
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}  # Допустимі розширення файлів

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

processing_status = {}  # Словник для відстеження статусу обробки файлів


# Перевірка дозволених розширень файлів
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Обгортка з можливістю відстеження прогресу
def video_to_cartoon_wrapper(input_path, output_path, model_path, filename):
    global processing_status
    print(f"Начало обработки {filename}")
    processing_status[filename] = 'processing'

    temp_folder = 'temp_frames'  # Тимчасова директорія для кадрів
    temp_result_folder = 'temp_result'  # Тимчасова директорія для результатів
    duration = 8  # Тривалість обробки в секундах

    # Створення директорії для обробленого файлу, якщо її не існує
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_frames, fps = extract_frames(input_path, temp_folder, duration)
    if total_frames == 0:
        print(f"Ошибка: не удалось извлечь кадры из {filename}")
        processing_status[filename] = 'error'
    else:
        process_images((256, 256), model_path, temp_folder, temp_result_folder)
        frames_to_video(temp_result_folder, output_path, fps)

        # Тимчасовий файл для відео без звуку
        temp_video_no_audio = output_path.replace('.mp4', '_no_audio.mp4')
        os.rename(output_path, temp_video_no_audio)

        add_audio_to_video(temp_video_no_audio, input_path, output_path, duration)
        clean_temp_folders(temp_folder, temp_result_folder)

        # Видалення тимчасового файлу відео без звуку
        os.remove(temp_video_no_audio)

        processing_status[filename] = 'done'
        print(f"Обработка {filename} завершена")


@app.route('/')
def index():
    return render_template('index.html')  # Повертає головну сторінку


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:  # Перевірка, чи є файл відео в запиті
        return 'No file part', 400
    file = request.files['video']  # Файл з запиту
    if file.filename == '':  # Перевірка, чи було вибрано відео
        return 'No selected file', 400
    if file and allowed_file(file.filename):  # Перевірка розширення файлу
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)  # Збереження файлу

        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        thread = threading.Thread(target=video_to_cartoon_wrapper,
                                  args=(save_path, processed_path, MODEL_PATH, processed_filename))
        thread.start()
        return 'Відео завантажено та знаходиться в обробці.', 202
    else:
        return 'Тип файлу не підтримується.', 400


@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)  # Завантаження обробленого файлу


@app.route('/status/<filename>')
def check_status(filename):
    status = processing_status.get(filename, 'not_found')
    return {'status': status}  # Повертає статус обробки файлу


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Створення директорії для завантажених файлів, якщо її не існує
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)  # Створення директорії для оброблених файлів, якщо її не існує
    app.run(debug=True)  # Запуск сервера у режимі налагодження
