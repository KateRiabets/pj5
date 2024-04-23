import numpy as np
import scipy.stats as st
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]  # середні значення для кожного кольорового каналу в BGR

class Vgg19:
    
    def __init__(self, vgg19_npy_path=None):
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')  # завантаження ваг моделі

    def build_conv4_4(self, rgb, include_fc=False):
        rgb_scaled = (rgb+1) * 127.5  # масштабування RGB значень з [-1,1] до [0,255]
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)  # розділення каналів
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])  # конвертація RGB в BGR

        # Послідовність сверткових і пулінг шарів
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        # Подібні шари прослідковуються аж до conv4_4
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = tf.nn.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        return self.conv4_4  # повернення результату свертки до conv4_4

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias  # повернення результату свертки без ReLU активації

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")  # завантаження ваг фільтра

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")  # завантаження зсуву

    def fc_layer(self, bottom, name):
        # Функція для створення повнозв'язного шару
        pass

def vggloss_4_4(image_a, image_b):
    vgg_model = Vgg19('vgg19_no_fc.npy')
    vgg_a = vgg_model.build_conv4_4(image_a)
    vgg_b = vgg_model.build_conv4_4(image_b)
    VGG_loss = tf.losses.absolute_difference(vgg_a, vgg_b)  # обчислення абсолютної різниці між двома зображеннями
    h, w, c= vgg_a.get_shape().as_list()[1:]
    VGG_loss = tf.reduce_mean(VGG_loss)/(h*w*c)  # нормалізація втрати
    return VGG_loss

# Функції для визначення втрат в генеративних змагальних мережах, які включають WGAN, GAN та LSGAN
# Ці функції обчислюють втрату на основі результатів реальних та сфабрикованих зображень

def total_variation_loss(image, k_size=1):
    # Функція для обчислення варіації загальної втрати, що використовується для згладжування зображення
    h, w = image.get_shape().as_list()[1:3]
    tv_h = tf.reduce_mean((image[:, k_size:, :, :] - image[:, :h - k_size, :, :])**2)
    tv_w = tf.reduce_mean((image[:, :, k_size:, :] - image[:, :, :w - k_size, :])**2)
    tv_loss = (tv_h + tv_w)/(3*h*w)
    return tv_loss
