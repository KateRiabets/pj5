import os
import cv2
import numpy as np
import tensorflow as tf
import network
import guided_filter
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def resize_crop(image, target_width, target_height):
    h, w, c = np.shape(image) #розміри вхідного зображення
    scale = min(target_height / h, target_width / w) #коефіцієнта масштабування для збереження пропорцій
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) #зміна розміру зображення
    pad_h = (target_height - new_h) // 2
    pad_w = (target_width - new_w) // 2
    image = cv2.copyMakeBorder(image, pad_h, target_height - new_h - pad_h, pad_w, target_width - new_w - pad_w,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image


def cartoonize_image(image, sess, final_out, input_photo, target_width, target_height):
    image = resize_crop(image, target_width, target_height) #підготовка зображення
    # нормалізація зображення
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    # застосування моделі для створення мультяшного ефекту
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def video_to_cartoon(video_path, save_path, model_path, update_progress=None):
    # завантаження моделі TensorFlow
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    #для збереження стану моделі
    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    # конфігурація сесії TensorFlow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # ініціалізація глобальних змінних і завантаження збереженого стану моделі
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    # читання відеофайлу
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # встановлення кодеку та параметрів для запису відео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = 'temp_output_video.mp4'
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    # створення індикатора прогресу
    progress_bar = tqdm(total=total_frames, desc='Processing Frames', unit='frames')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cartoon_frame = cartoonize_image(frame, sess, final_out, input_photo, width, height)
        # запис кадру в вихідний файл
        out.write(cartoon_frame)
        progress_bar.update(1)
        processed_frames += 1
        if update_progress:
            progress = (processed_frames / total_frames) * 100
            update_progress(round(progress))

    # закриття всіх ресурсів
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    progress_bar.close()

    # додавання звуку до відео
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    final_clip = VideoFileClip(temp_output_path).set_audio(audio_clip)
    final_clip.write_videofile(save_path, codec='libx264', audio_codec='aac')
    if update_progress:
        update_progress(100)
    # видалення тимчасових файлів
    os.remove(temp_output_path)


if __name__ == '__main__':
    model_path = 'saved_models'
    video_path = 'test_videos/input_video.mp4'
    save_path = 'cartoonized_videos/output_video.mp4'
    video_to_cartoon(video_path, save_path, model_path)
