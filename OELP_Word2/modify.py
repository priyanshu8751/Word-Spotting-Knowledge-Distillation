import tensorflow as tf
import os.path
import numpy as np

import os
import sys
from typing import List, Tuple

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from itertools import groupby
from dataloader_iam import Batch

import argparse
import json
from typing import Tuple, List

import cv2
import pandas as pd
import editdistance
from path import Path
import numpy as np
import tensorflow as tf

from dataloader_iam import DataLoaderIAM, Batch
from model_file import build_model, train_batch, infer_batch, get_Prediction_Model, CTCLoss
from preprocessor import Preprocessor

# set all connected gpu to true so that they can run the process
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# assigning paths to variable names
class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


# fixing all image heights to 50 and returning through function
def get_img_height() -> int:
    """Fixed height for NN."""
    return 50


# image size is returning the dimensions 250*50 which is also fixed
def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    return 250, get_img_height()


# here opening file names and writing the error rates and accuracy in the files
def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def train(batch_size, model, char_list,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        # each iteration increasing number of epocs
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        # WHAT THIS LINE DOES
        loader.train_set()
        # till any training set is remaining
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            model = train_batch(batch, model, char_list)
            # print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(get_Prediction_Model(model), char_list, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        min_delta = 0.001
        if (char_error_rate + min_delta) <= best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save('CTC-model_' + str(batch_size))
        else:
            # WHAT IS := THIS
            print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model, char_list, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    # WHAT THIS LINE DOES
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized = infer_batch(batch.imgs, model, char_list)
        # orig_texts = []
        # for label in batch.gt_texts:
        #     label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        #     orig_texts.append(label)
        # print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            # WE ARE DOING WORD SPOTTING SO HOW ARE WE DOING THIS BY THIS ISN'T IT WORD RECOGNITION
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            # DISTANCE IS CALCULATED BUT HOW
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            # print("True Label : ", batch.gt_texts[i], " Recognised : ", recognized[i])

    # print validation result
    # ACCURACY ATTAINED TILL NOW 
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.', flush=True)
    return char_error_rate, word_accuracy


# def infer(model, fn_img: Path) -> None:
#     """Recognizes text in image provided by file path."""
#     img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
#     assert img is not None
#
#     preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
#     img = preprocessor.process_img(img)
#
#     batch = Batch([img], None, 1)
#     recognized, probability = model.infer_batch(batch, True)
#     # print(f'Recognized: "{recognized[0]}"')
#     # print(f'Probability: {probability[0]}')
#     return recognized[0]


def main():
    """Main function."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=16)
    parser.add_argument('--test_img', help='Testing images ', required=False)
    parser.add_argument('--test_csv', help='Testing map ', required=False)
    parser.add_argument('--validate_img', help='Validation images ', required=False)
    parser.add_argument('--validate_csv', help='Validation map ', required=False)
    parser.add_argument('--train_img', help='Training images ', required=False)
    parser.add_argument('--train_csv', help='Training map ', required=False)
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=10)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
    args = parser.parse_args()

    # train or validate on IAM dataset
    if args.mode == 'train':
        # load training data, create TF model
        loader = DataLoaderIAM(args.train_img, args.train_csv, args.validate_img, args.validate_csv, args.batch_size)
        char_list = loader.char_list
        # print(char_list)
        # when in line mode, take care to have a whitespace in the char list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

        model = build_model(char_list)
        train(args.batch_size, model, char_list, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # test the saved model on a set of images
    elif args.mode == 'test':
        char_list = list(open(FilePaths.fn_char_list).read())
        model = tf.keras.models.load_model('CTC-model_'+str(args.batch_size), custom_objects={'CTCLoss': CTCLoss})
        #model = get_Prediction_Model(model)
        #model=tf.keras.models.load_model('ravi_GW2_phos_64_'+".h5")
        map_path = './dataset/' + args.test_csv
        img_path = './dataset/' + args.test_img
        num_char_err = 0
        num_char_total = 0
        file = pd.read_csv(map_path)
        image_name_list = list(file['Image'])
        gt_list = list(file['Word'])
        total_words = 0
        words_ok = 0
        for i in range(len(image_name_list)):
            total_words = total_words + 1
            img = cv2.imread(img_path + image_name_list[i], cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float)
            img = cv2.transpose(img)
            # WHAT is this
            img = img / 255
            recognised_string = infer_batch([img], model, char_list)
            #print(recognised_string, gt_list[i] )
            if gt_list[i] == recognised_string[0]:
                words_ok = words_ok + 1
            print(gt_list[i] , recognised_string[0])
            dist = editdistance.eval(recognised_string[0], gt_list[i])
            num_char_err += dist
            num_char_total += len(gt_list[i])
        print("Word Accuracy : ", round((words_ok / total_words) * 100, 2))
        print("Character Error rate : ", round((num_char_err / num_char_total) * 100, 2))


if __name__ == '__main__':
    main()

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



def train_batch(batch: Batch, model, char_list):
    images = batch.imgs
    true_labels = batch.gt_texts
    max_label_len = 14
    padded_image_texts = [encode_to_labels(label, char_list, max_label_len) for label in true_labels]
    results = model.fit(np.array(images), np.array(padded_image_texts), epochs=1, batch_size=len(true_labels), verbose=0)
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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# WGAN -GP implementation
def generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=20000))  # layer 1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(units=16000))  # layer 1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(units=12000))  # layer 1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(units=8000))  # layer 1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(units=6000))  # layer 1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(units=3968))  # layer 2
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    return model


def discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=2000, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(units=500, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(units=150, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(units=1))  # linear activation in last layer for WGAN
    return model


class WGAN():
    def __init__(self, generator_lr, discriminator_lr):
        if os.path.isdir("Generator"):
            print("Loading saved generator model", flush=True)
            self.generator = tf.keras.models.load_model('Generator')
        else:
            print("Loading new generator model", flush=True)
            self.generator = generator()  # get generator model

        if os.path.isdir("Discriminator"):
            print("Loading saved discriminator model", flush=True)
            self.discriminator = tf.keras.models.load_model('Discriminator')
        else:
            print("Loading new discriminator model", flush=True)
            self.discriminator = discriminator()  # get discriminator model

        self.LAMBDA = 10
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr, beta_1=0, beta_2=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr,  beta_1=0, beta_2=0.9)
        self.e = 0
        self.update_count_d = 0
        self.max_update_count = 3

    def generator_loss(self, noise, phoc_vectors):
        fake_sample = self.generator(noise, training=True)
        fake_sample = tf.concat((fake_sample, phoc_vectors), axis=1)
        prediction = self.discriminator(fake_sample, training=True)
        # WHAT IS LOSS HERE AND WHAT IS ACTUALLY TRAINED HERE DISCRIMINATOR 
        loss = -tf.reduce_mean(prediction)
        return loss

    def discriminator_loss(self, true_sample, fake_sample, phoc_vectors):
        true_sample = tf.concat((true_sample, phoc_vectors), axis=1)
        fake_sample = tf.concat((fake_sample, phoc_vectors), axis=1)
        true_prediction = self.discriminator(true_sample, training=True)
        fake_prediction = self.discriminator(fake_sample, training=True)
        loss =  tf.reduce_mean(fake_prediction) - tf.reduce_mean(true_prediction)
        return loss

    def save_models(self):
        self.discriminator.save("Discriminator")
        self.generator.save("Generator")
