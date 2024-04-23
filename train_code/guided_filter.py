import tensorflow as tf
import numpy as np

# функція для створення боксового фільтру з вказаним радіусом r для згладжування зображення
def tf_box_filter(x, r):
    ch = x.get_shape().as_list()[-1]  # отримання кількості каналів вхідного зображення
    weight = 1/((2*r+1)**2)  # вага для кожного пікселя у ядрі фільтру
    box_kernel = weight*np.ones((2*r+1, 2*r+1, ch, 1))  # створення ядра фільтру
    box_kernel = np.array(box_kernel).astype(np.float32)  # перетворення ядра в потрібний тип даних
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')  # застосування фільтру через глибинну конволюцію
    return output

# гідований фільтр
def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)  # отримання розмірів вхідного тензора x

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)  # обчислення розміру області фільтрації

    mean_x = tf_box_filter(x, r) / N  # середнє значення по x
    mean_y = tf_box_filter(y, r) / N  # середнє значення по y
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y  # коваріація між x та y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x  # дисперсія x

    A = cov_xy / (var_x + eps)  # коефіцієнт A для лінійної регресії
    b = mean_y - A * mean_x  # вільний член b

    mean_A = tf_box_filter(A, r) / N  # усереднений коефіцієнт A
    mean_b = tf_box_filter(b, r) / N  # усереднений вільний член b

    output = mean_A * x + mean_b  # кінцевий результат гідованого фільтру

    return output
