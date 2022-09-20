#!/bin/env python3.8
# Gavri Kepets
# Help with debugging received by Husam Almanakhly, and help with understanding the bases received from Ali Ghuman

import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))

# global variables
upper_bound = 1
lower_bound = 0
constant_sigma = 0.2


def gaussian(x, mew, sigma):
    return tf.math.exp(-((x - mew) ** 2) / sigma**2)


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    m: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(lower_bound, upper_bound, size=(self.num_samples, 1))
        clean_y = np.sin(2 * np.pi * self.x)  # values of true sampled sin wave
        self.y = rng.normal(
            loc=clean_y, scale=self.sigma
        )  # added noise from normal dist

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


matplotlib.style.use("classic")
matplotlib.rc(
    "font",
    **{
        "size": 10,
    },
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("m", 5, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_bool("debug", False, "Set logging level to debug")


class Model(tf.Module):
    def __init__(self, rng, m):
        self.m = m
        self.w = tf.Variable(rng.normal(shape=[self.m, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        self.mews = tf.Variable(rng.normal(shape=[self.m, 1]), name="means")
        self.sigmas = tf.Variable(rng.normal(shape=[self.m, 1]), name="sigmas")

    def __call__(self, x):
        return tf.squeeze(
            tf.reduce_sum(
                tf.transpose(self.w)
                * gaussian(x, tf.transpose(self.mews), tf.transpose(self.sigmas)),
                1,
            )
            + self.b
        )


def main(a):
    logging.basicConfig()

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    data = Data(np_rng, FLAGS.m, FLAGS.num_samples, 0.1)

    model = Model(tf_rng, FLAGS.m)
    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    bar = trange(FLAGS.num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

    ax[0].set_title("Sine Wave Linear Regression (M=" + str(FLAGS.m) + ")")
    ax[0].set_xlabel("x")
    h = ax[0].set_ylabel("y")
    h.set_rotation(0)

    # plot sample points
    xs = np.linspace(lower_bound, upper_bound, 1000)
    xs = xs[:, np.newaxis]
    yh = model(xs)
    ax[0].plot(np.squeeze(data.x), data.y, "o", color="green")

    # plot model
    ax[0].plot(xs, np.squeeze(yh), "--", color="red")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    # plot true sin wave
    true_sin = np.sin(2 * np.pi * xs)
    ax[0].plot(xs, true_sin, color="blue")

    # plot bases
    xs = np.linspace(
        np.amin(model.mews) - constant_sigma * 3,
        np.amax(model.mews) + constant_sigma * 3,
        1000,
    )
    ax[1].set_xlim(
        np.amin(model.mews) - constant_sigma * 3,
        np.amax(model.mews) + constant_sigma * 3,
    )
    ax[1].set_title("Bases")
    ax[1].set_xlabel("x")
    h = ax[1].set_ylabel("y")
    h.set_rotation(0)

    for i in range(FLAGS.m):
        ax[1].plot(
            xs,
            gaussian(xs, model.mews[i], model.sigmas[i]),
        )

    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")


if __name__ == "__main__":
    app.run(main)
