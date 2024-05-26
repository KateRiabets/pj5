import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip

def extract_frames(video_path, temp_folder, duration=8):
    os.makedirs(temp_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps * duration)
    frame_number = 0
    while frame_number < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_folder, f"{frame_number:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_number += 1
    cap.release()
    return total_frames, fps

def process_images(image_size, model_path, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def load_images_from_folder(folder):
        images = []
        original_sizes = []
        filenames = sorted([f for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))],
                           key=lambda x: int(''.join(filter(str.isdigit, x))))
        for filename in filenames:
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            original_sizes.append(img.size)
            img_array = img_to_array(img)
            images.append(img_array)
        return images, filenames, original_sizes

    model = tf.keras.models.load_model(model_path)
    test_images, test_filenames, original_sizes = load_images_from_folder(input_folder)

    if len(test_images) == 0:
        print("No images found in the input folder.")
        return

    test_images_resized = [tf.image.resize(img, image_size) / 255.0 for img in test_images]
    test_images_resized = np.array(test_images_resized)

    print("Predicting on test images...")
    predictions_resized = model.predict(test_images_resized)

    predictions = []
    for i, pred in enumerate(predictions_resized):
        original_size = original_sizes[i]
        resized_pred = tf.image.resize(pred, original_size).numpy()
        predictions.append(resized_pred)

    for i, prediction in enumerate(predictions):
        result_path = os.path.join(output_folder, test_filenames[i])
        pred_image = Image.fromarray(np.uint8(prediction * 255))
        original_image = Image.open(os.path.join(input_folder, test_filenames[i]))
        pred_image = pred_image.resize(original_image.size, Image.Resampling.LANCZOS)
        try:
            exif = original_image.info['exif']
            pred_image.save(result_path, exif=exif)
        except KeyError:
            pred_image.save(result_path)

    print(f"Все изображения обработаны и сохранены в папке {output_folder}")

def frames_to_video(temp_folder, output_video_path, fps):
    frame_paths = sorted([os.path.join(temp_folder, f) for f in os.listdir(temp_folder) if f.endswith('.png')])
    frame = cv2.imread(frame_paths[0])
    height, width, _ = frame.shape

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

def add_audio_to_video(video_path, audio_path, output_path, duration):
    video_clip = VideoFileClip(video_path)
    try:
        audio_clip = AudioFileClip(audio_path).subclip(0, duration)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except:
        print("No audio found in the video, skipping audio addition.")
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def clean_temp_folders(*folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            os.remove(file_path)
        os.rmdir(folder)

