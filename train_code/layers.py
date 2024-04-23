import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# адаптивна нормалізація екземпляра, використовується для адаптації змісту зображення до стилю іншого зображення
def adaptive_instance_norm(content, style, epsilon=1e-5):
    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)  # обчислення середнього та варіансу змісту
    s_mean, s_var = tf.nn.moments(style, axes=[1, 2], keep_dims=True)  # обчислення середнього та варіансу стилю
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)  # стандартні відхилення змісту та стилю

    return s_std * (content - c_mean) / c_std + s_mean  # нормалізоване зображення змісту до стилю

# спектральна нормалізація для стабілізації навчання генеративних мереж
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()  # форма ваги
    w = tf.reshape(w, [-1, w_shape[-1]])  # згортання в матрицю

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # ініціалізація u

    u_hat = u
    v_hat = None
    for i in range(iteration):  # ітерація для покращення оцінки спектральної норми
        v_ = tf.matmul(u_hat, tf.transpose(w))  # оновлення v
        v_hat = tf.nn.l2_normalize(v_)  # нормалізація v

        u_ = tf.matmul(v_hat, w)  # оновлення u
        u_hat = tf.nn.l2_normalize(u_)  # нормалізація u

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))  # обчислення спектральної норми

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma  # нормалізація ваги
        w_norm = tf.reshape(w_norm, w_shape)  # повернення до первісної форми

    return w_norm

# конволюційний шар зі спектральною нормалізацією
def conv_spectral_norm(x, channel, k_size, stride=1, name='conv_snorm'):
    with tf.variable_scope(name):
        w = tf.get_variable("kernel", shape=[k_size[0], k_size[1], x.get_shape()[-1], channel])
        b = tf.get_variable("bias", [channel], initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME') + b

        return x

# блок самоуваги для моделей генеративних змагань
def self_attention(inputs, name='attention', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]  # висота і ширина вхідних даних
        bs, _, _, ch = inputs.get_shape().as_list()  # базова інформація про форму входу
        f = slim.convolution2d(inputs, ch//8, [1, 1], activation_fn=None)  # зменшення розміру каналу
        g = slim.convolution2d(inputs, ch//8, [1, 1], activation_fn=None)  
        s = slim.convolution2d(inputs, 1, [1, 1], activation_fn=None)  
        f_flatten = tf.reshape(f, shape=[f.shape[0], -1, f.shape[-1]])  
        g_flatten = tf.reshape(g, shape=[g.shape[0], -1, g.shape[-id]])  # матриця уваги
        beta = tf.matmul(f_flatten, g_flatten, transpose_b=True)  # вагові коефіцієнти для уваги
        beta = tf.nn.softmax(beta)  # softmax для нормалізації коефіцієнтів уваги
        
        s_flatten = tf.reshape(s, shape=[s.shape[0], -1, s.shape[-1]])  
        att_map = tf.matmul(beta, s_flatten)  # карта уваги
        att_map = tf.reshape(att_map, shape=[bs, h, w, 1])  
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))  
        output = att_map * gamma + inputs  # результуюче зображення з урахуванням уваги
        
        return att_map, output
    

if __name__ == '__main__':
    pass
