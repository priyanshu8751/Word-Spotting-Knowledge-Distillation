import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from itertools import groupby
from dataloader_iam import Batch


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def ctc_decoder(predictions, char_list):
    '''
    input: given batch of predictions from text rec model
    output: return lists of raw extracted text

    '''
    text_list = []

    pred_indcies = np.argmax(predictions, axis=2)
    # print(pred_indcies,predictions)
    # WHAT IS THIS PART DOING
    for i in range(pred_indcies.shape[0]):
        ans = ""

        # merge repeats
        merged_list = [k for k, _ in groupby(pred_indcies[i])]

        # remove blanks
        for p in merged_list:
            if p != len(char_list):
                ans += char_list[int(p)]

        text_list.append(ans)

    return text_list


def encode_to_labels(txt, char_list, max_label_len):
    # encoding each output word into digits
    dig_lst = []

    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return pad_sequences([dig_lst], maxlen=max_label_len, padding='post', value=len(char_list))[0]


# class CTCLayer(layers.Layer):

#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self.loss_fn = tf.keras.backend.ctc_batch_cost

#     def call(self, y_true, y_pred):
#         # Compute the training-time loss value and add it
#         # to the layer using `self.add_loss()`.

#         batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#         input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#         label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

#         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#         label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

#         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
#         self.add_loss(loss)

#         # At test time, just return the computed predictions
#         return y_pred


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def build_model(char_list):
    input = layers.Input(shape=(250, 50, 1), name="input_images")
    conv_1 = layers.Conv2D(32, (3, 3), padding='same')(input)  # conv1
    batch_1 = layers.BatchNormalization()(conv_1)
    relu_1 = layers.ReLU()(batch_1)
    pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_1)
    conv_2 = layers.Conv2D(32, (3, 3), padding='same')(pool_3)  # conv2
    batch_2 = layers.BatchNormalization()(conv_2)
    relu_2 = layers.ReLU()(batch_2)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(relu_2)
    conv_3 = layers.Conv2D(64, (3, 3), padding='same')(pool_1)  # conv3
    batch_3 = layers.BatchNormalization()(conv_3)
    relu_3 = layers.ReLU()(batch_3)
    pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_3)
    conv_4 = layers.Conv2D(64, (3, 3), padding='same')(pool_3)  # conv4
    batch_3 = layers.BatchNormalization()(conv_4)
    relu_3 = layers.ReLU()(batch_3)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(relu_3)
    conv_8 = layers.Conv2D(64, (3, 3), padding='same')(pool_2)  # conv5
    batch_4 = layers.BatchNormalization()(conv_8)
    relu_4 = layers.ReLU()(batch_4)
    pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_4)
    conv_9 = layers.Conv2D(64, (3, 3), padding='same')(pool_3)  # conv6
    batch_4 = layers.BatchNormalization()(conv_9)
    relu_4 = layers.ReLU()(batch_4)
    pool_6 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same')(relu_4)
    rnn_input = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2), name='last_cnn_output')(pool_6)
    bi_1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(rnn_input)
    bi_2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(bi_1)
    softmax_output = layers.Dense(len(char_list) + 1, activation='softmax', name="dense")(bi_2)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model = Model(inputs=input, outputs=softmax_output)
    model.compile(optimizer=opt, loss=CTCLoss)
    return model

def build_model2(char_list):
    input = layers.Input(shape=(250, 50, 1), name="input_images")
    conv_1 = layers.Conv2D(8, (3, 3), padding='same')(input)  # conv1
    batch_1 = layers.BatchNormalization()(conv_1)
    relu_1 = layers.ReLU()(batch_1)
    pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_1)
    conv_2 = layers.Conv2D(8, (3, 3), padding='same')(pool_3)  # conv2
    batch_2 = layers.BatchNormalization()(conv_2)
    relu_2 = layers.ReLU()(batch_2)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(relu_2)
    conv_3 = layers.Conv2D(16, (3, 3), padding='same')(pool_1)  # conv3
    batch_3 = layers.BatchNormalization()(conv_3)
    relu_3 = layers.ReLU()(batch_3)
    pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_3)
    conv_4 = layers.Conv2D(16, (3, 3), padding='same')(pool_3)  # conv4
    batch_3 = layers.BatchNormalization()(conv_4)
    relu_3 = layers.ReLU()(batch_3)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(relu_3)
    conv_8 = layers.Conv2D(16, (3, 3), padding='same')(pool_2)  # conv5
    batch_4 = layers.BatchNormalization()(conv_8)
    relu_4 = layers.ReLU()(batch_4)
    pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_4)
    conv_9 = layers.Conv2D(16, (3, 3), padding='same')(pool_3)  # conv6
    batch_4 = layers.BatchNormalization()(conv_9)
    relu_4 = layers.ReLU()(batch_4)
    pool_6 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same')(relu_4)
    rnn_input = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2), name='last_cnn_output')(pool_6)
    bi_1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(rnn_input)
    bi_2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(bi_1)
    softmax_output = layers.Dense(len(char_list) + 1, activation='softmax', name="dense")(bi_2)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model = Model(inputs=input, outputs=softmax_output)
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


def train_batch(batch: Batch, model, char_list):
    images = batch.imgs
    true_labels = batch.gt_texts
    max_label_len = 14
    padded_image_texts = [encode_to_labels(label, char_list, max_label_len) for label in true_labels]
    print(images[0])
    print(padded_image_texts[0])
    results = model.fit(np.array(images), np.array(padded_image_texts), epochs=1)
    # , batch_size=len(true_labels), verbose=0
    return model


def get_Prediction_Model(model):
    return model


def infer_batch(images, model, char_list):
    predicted = model.predict(np.array(images), verbose=0)
    predicted = ctc_decoder(predicted, char_list)
    return predicted


def get_CNNPrediction_Model(model):
    prediction_model = tf.keras.models.Model(model.get_layer(name="input_images").input,
                                             model.get_layer(name="last_cnn_output").output)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    prediction_model.compile(optimizer=opt)
    # print(prediction_model.summary())
    return prediction_model


def infer_batch_cnn(model, images):
    cnn_output = model.predict(np.array(images), verbose=0)
    return cnn_output


def get_Label_Prediction_Model(model):
    lstm_input = layers.Input(shape=(62, 64))
    x = lstm_input
    all_layers = [layer for layer in model.layers]
    for i in range(26, len(all_layers)):
        x = all_layers[i](x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    prediction_model = Model(inputs=lstm_input, outputs=x)
    prediction_model.compile(optimizer=opt, loss=CTCLoss)
    # print(prediction_model.summary())
    return prediction_model


def infer_batch_labels(model, images, char_list, true_labels, get_labels):
    max_label_len = 14
    padded_image_texts = [encode_to_labels(label, char_list, max_label_len) for label in true_labels]
    predicted = None
    if get_labels:
        labels = model.predict(np.array(images))
        predicted = ctc_decoder(labels, char_list)
    loss = model.evaluate(np.array(images), np.array(padded_image_texts), verbose=0)
    return loss, predicted

# def build_model(char_list):
#     input = layers.Input(shape=(250, 50, 1), name="input_images")
#     # labels = layers.Input(name="label", shape=(None,), dtype="float32")
#     conv_1 = layers.Conv2D(32, (3, 3), padding='same')(input)
#     batch_1 = layers.BatchNormalization()(conv_1)
#     relu_1 = layers.ReLU()(batch_1)
#     conv_2 = layers.Conv2D(32, (3, 3), padding='same')(relu_1)
#     batch_2 = layers.BatchNormalization()(conv_2)
#     relu_2 = layers.ReLU()(batch_2)
#     pool_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(relu_2)
#     conv_3 = layers.Conv2D(64, (3, 3), padding='same')(pool_1)
#     batch_3 = layers.BatchNormalization()(conv_3)
#     relu_3 = layers.ReLU()(batch_3)
#     conv_4 = layers.Conv2D(64, (3, 3), padding='same')(relu_3)
#     batch_3 = layers.BatchNormalization()(conv_4)
#     relu_3 = layers.ReLU()(batch_3)
#     pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(relu_3)
#     conv_8 = layers.Conv2D(96, (3, 3), padding='same')(pool_2)
#     batch_4 = layers.BatchNormalization()(conv_8)
#     relu_4 = layers.ReLU()(batch_4)
#     conv_9 = layers.Conv2D(96, (3, 3), padding='same')(relu_4)
#     batch_4 = layers.BatchNormalization()(conv_9)
#     relu_4 = layers.ReLU()(batch_4)
#     pool_3 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_4)
#     conv_10 = layers.Conv2D(96, (3, 3), padding='same')(pool_3)
#     batch_5 = layers.BatchNormalization()(conv_10)
#     relu_5 = layers.ReLU()(batch_5)
#     conv_11 = layers.Conv2D(128, (3, 3), padding='same')(relu_5)
#     batch_5 = layers.BatchNormalization()(conv_11)
#     relu_5 = layers.ReLU()(batch_5)
#     pool_4 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(relu_5)
#     conv_12 = layers.Conv2D(128, (3, 3), padding='same')(pool_4)
#     batch_5 = layers.BatchNormalization()(conv_12)
#     relu_5 = layers.ReLU()(batch_5)
#     pool_5 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same')(relu_5)
#     conv_13 = layers.Conv2D(128, (3, 3), padding='same')(pool_5)
#     batch_6 = layers.BatchNormalization()(conv_13)
#     relu_6 = layers.ReLU()(batch_6)
#     pool_6 = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same')(relu_6)
#     rnn_input = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2), name='last_cnn_output')(pool_6)
#     bi_1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(rnn_input)
#     bi_2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(bi_1)
#     softmax_output = layers.Dense(len(char_list) + 1, activation='softmax', name="dense")(bi_2)
#     opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
#     model = Model(inputs=input, outputs=softmax_output)
#     model.compile(optimizer=opt, loss=CTCLoss)
#     return model

# MODDLE SUMMARY
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_images (InputLayer)    [(None, 250, 50, 1)]      0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 250, 50, 32)       320
# _________________________________________________________________
# batch_normalization (BatchNo (None, 250, 50, 32)       128
# _________________________________________________________________
# re_lu (ReLU)                 (None, 250, 50, 32)       0
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 250, 25, 32)       0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 250, 25, 32)       9248
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 250, 25, 32)       128
# _________________________________________________________________
# re_lu_1 (ReLU)               (None, 250, 25, 32)       0
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 125, 12, 32)       0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 125, 12, 64)       18496
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 125, 12, 64)       256
# _________________________________________________________________
# re_lu_2 (ReLU)               (None, 125, 12, 64)       0
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 125, 6, 64)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 125, 6, 64)        36928
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 125, 6, 64)        256
# _________________________________________________________________
# re_lu_3 (ReLU)               (None, 125, 6, 64)        0
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 62, 3, 64)         0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 62, 3, 64)         36928
# _________________________________________________________________
# batch_normalization_4 (Batch (None, 62, 3, 64)         256
# _________________________________________________________________
# re_lu_4 (ReLU)               (None, 62, 3, 64)         0
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 62, 1, 64)         0
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 62, 1, 64)         36928
# _________________________________________________________________
# batch_normalization_5 (Batch (None, 62, 1, 64)         256
# _________________________________________________________________
# re_lu_5 (ReLU)               (None, 62, 1, 64)         0
# _________________________________________________________________
# max_pooling2d_5 (MaxPooling2 (None, 62, 1, 64)         0
# _________________________________________________________________
# last_cnn_output (Lambda)     (None, 62, 64)            0
# _________________________________________________________________
# bidirectional (Bidirectional (None, 62, 512)           657408
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 62, 512)           1574912
# _________________________________________________________________
# dense (Dense)                (None, 62, 5)             2565
# =================================================================
# Total params: 2,375,013
# Trainable params: 2,374,373
# Non-trainable params: 640