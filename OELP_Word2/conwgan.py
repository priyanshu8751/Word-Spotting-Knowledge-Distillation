import tensorflow as tf
import os.path
import numpy as np

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
