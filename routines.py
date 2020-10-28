import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Zeros, Ones


class ColoredMNISTEnvironments:

    def __init__(self):
        self.__load_initial_data()
        self.__create_envs()
        self.__create_validation_envs()

    def __load_initial_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # subsample for computational efficiency
        x_train = x_train.reshape((-1, 28, 28))[:, ::2, ::2]
        x_test = x_test.reshape((-1, 28, 28))[:, ::2, ::2]

        # convert to RGB
        x_train = np.stack((x_train,) * 3, axis=-1)
        x_test = np.stack((x_test,) * 3, axis=-1)

        # normalize
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # binary label
        y_train = (y_train < 5).astype('bool')
        y_test = (y_test < 5).astype('bool')

        self.original_data = {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def __create_envs(self):
        k = 10 ** 4
        self.e1 = self.__create_env(self.original_data['x_train'][:k],
                                    self.original_data['y_train'][:k], .1)
        self.e2 = self.__create_env(self.original_data['x_train'][k:2 * k],
                                    self.original_data['y_train'][k:2 * k], .2)
        self.e3 = self.__create_env(self.original_data['x_train'][2 * k:3 * k],
                                    self.original_data['y_train'][2 * k:3 * k], .9)
        x, y = self.__create_env(self.original_data['x_train'][2 * k:3 * k],
                                 self.original_data['y_train'][2 * k:3 * k], .9, return_tf_data=False)
        x[:, :, :, 1] = x[:, :, :, 0]
        x[:, :, :, 2] = x[:, :, :, 0]
        self.e3_greyscale = tf.data.Dataset.from_tensor_slices((x, y))

    def __create_validation_envs(self):
        k = 10 ** 4
        i = 3 * k
        self.e11 = self.__create_env(self.original_data['x_train'][i:i + k],
                                     self.original_data['y_train'][i:i + k], .1)
        self.e22 = self.__create_env(self.original_data['x_train'][i + k:i + 2 * k],
                                     self.original_data['y_train'][i + k:i + 2 * k], .2)
        self.e33 = self.__create_env(self.original_data['x_train'][i + 2 * k:i + 3 * k],
                                     self.original_data['y_train'][i + 2 * k:i + 3 * k], .9)

        x, y = self.__create_env(self.original_data['x_train'][i + 2 * k:i + 3 * k],
                                 self.original_data['y_train'][i + 2 * k:i + 3 * k], .9, return_tf_data=False)
        x[:, :, :, 1] = x[:, :, :, 0]
        x[:, :, :, 2] = x[:, :, :, 0]
        self.e33_greyscale = tf.data.Dataset.from_tensor_slices((x, y))

    @staticmethod
    def __create_env(x, y, e, labelflip_proba=.25, return_tf_data=True):
        x = x.copy()
        y = y.copy()

        y = np.logical_xor(
            y,
            (np.random.random(size=len(y)) < labelflip_proba).astype(bool)
        ).astype(bool)

        color = np.logical_xor(
            y,
            (np.random.random(size=len(y)) < e).astype(bool)
        )

        x[color, :, :, 2] = np.float32(0)
        x[color, :, :, 1] = np.float32(0)
        if return_tf_data:
            return tf.data.Dataset.from_tensor_slices((x, y))
        return x, y


def get_model(args):
    n_final_units = args.nb_units
    mlp = args.mlp
    l2reg = args.l2_reg
    n_layers = args.nb_layers
    dropout = None if args.dropout == 0 else args.dropout

    input_images = Input(shape=(14, 14, 3))

    if not mlp:
        cnn = Conv2D(32, kernel_size=(3, 3),
                     activation='relu')(input_images)
        cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
        if dropout is not None:
            cnn = Dropout(dropout)(cnn)
        cnn = Flatten()(cnn)
        representation = cnn
    else:
        d = Flatten()(input_images)
        for _ in range(n_layers):
            d = Dense(n_final_units, activation='relu')(d)
        representation = d

    if not mlp:
        final = Dense(32, activation='relu')(representation)
        if dropout is not None:
            final = Dropout(dropout)(final)
    else:
        final = representation

    output = Dense(1, kernel_regularizer=l2(l2reg), name='output')(final)
    dummy_weights = np.zeros(n_final_units, dtype=np.float32)
    dummy_weights[0] = 1.
    dummy = Dense(1,
                  kernel_initializer=tf.constant_initializer(dummy_weights),
                  bias_initializer=Zeros(), name='dummy', trainable=True)(final)

    model = Model(
        inputs=[input_images],
        outputs=[output, dummy]
    )
    model.dummy = dummy
    model.dummy_weights = dummy_weights

    return model


class IRMModel(object):

    def __init__(self, model, optimizer=tf.keras.optimizers.Adam(), envs=None):
        self.model = model
        self.optimizer = optimizer
        self.envs = ColoredMNISTEnvironments() if envs is None else envs
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.dummy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.logs = defaultdict(list)
        self.logdir = "/home/e.diemert/tflogs/scalars/" + \
                      datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + \
                      '-%d' % random.randint(0, 1000)
        print("logging to", self.logdir)
        self.file_writer = tf.summary.create_file_writer(self.logdir + "/metrics")
        self.file_writer.set_as_default()

    def evaluate(self, env):
        d = [_ for _ in env]
        x, y = tf.convert_to_tensor([_[0] for _ in d]), tf.convert_to_tensor([_[1] for _ in d])
        accuracy = tf.keras.metrics.Accuracy()
        loss = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
        per_batch_penalties = []
        _, dummy_penalty = self.compute_penalty(x, y, training=False)
        logits, _ = self.model(x, training=False)
        loss.update_state(y, logits)
        accuracy.update_state(y, tf.math.greater(tf.keras.activations.sigmoid(logits), .5))
        return loss.result().numpy(), accuracy.result().numpy(), dummy_penalty.numpy()

    def compute_penalty(self, x, y, training=True):
        with tf.GradientTape() as tape:
            _, dummy_logits = self.model(x, training=training)
            dummy_loss = self.dummy_loss(y, dummy_logits)
        # print("DL", dummy_loss.numpy())
        dummy_grads = tape.gradient(dummy_loss, self.model.trainable_variables)
        dummy_penalty = [
            tf.reduce_sum(g)
            for g, v in zip(dummy_grads, self.model.trainable_variables) if 'dummy/kernel' in v.name
        ]
        assert len(dummy_penalty) == 1
        dummy_penalty = dummy_penalty[0]**2
        # print("DP", dummy_penalty.numpy())
        return dummy_loss, dummy_penalty

    def compute_gradients(self, x, y, lambda_=1, use_loss=True):
        # compute model gradients
        with tf.GradientTape() as tape:
            _, penalty = self.compute_penalty(x, y)
            logits, _ = self.model(x, training=True)
            if use_loss:
                loss = self.loss(y, logits) + lambda_ * penalty
            else:
                loss = lambda_ * penalty
            # loss = (
            #                self.loss(y_s[i], logits) + tf.convert_to_tensor(lambda_,
            #                                                                 dtype=tf.dtypes.float32) * penalty
            #        ) / tf.convert_to_tensor(lambda_, dtype=tf.dtypes.float32)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def do_evaluations(self, epoch, lambda_, print_=True, batch_size=2048):
        if print_:
            print('-' * 80)
            print("epoch:", epoch)
        for env_name, env in (('e10', self.envs.e1),
                              # ('e11', self.envs.e11),
                              ('e20', self.envs.e2),
                              # ('e22', self.envs.e22),
                              # ('e30', self.envs.e3),
                              ('e33', self.envs.e33),
                              ('egs', self.envs.e33_greyscale)):
            env_loss, env_acc, env_penalty = self.evaluate(env)
            self.log_event(env_name + '_loss', env_loss, epoch)
            self.log_event(env_name + '_acc', env_acc, epoch)
            self.log_event(env_name + '_pen', env_penalty, epoch)
            if print_:
                print('%s acc: %.3f loss %.5f pen: %.7f' % (env_name, env_acc, env_loss, lambda_*env_penalty))
        if print_:
            print('-' * 80)

    def log_event(self, event, value, epoch):
        tf.summary.scalar(event, data=value, step=epoch)
        self.logs[event] += [value]

    @staticmethod
    def grad_norm(grad):
        if grad is None:
            return tf.constant(0)
        return tf.math.sqrt(tf.reduce_sum(tf.math.square(grad)))

    def report_grad_norms(self, loss, grads, name, epoch):
        self.log_event(name + 'loss', loss.numpy(), epoch)
        total_grad_norm = tf.reduce_sum([self.grad_norm(grad) for grad in grads if grad is not None])
        self.log_event(name + 'grad', total_grad_norm.numpy(), epoch)
        for layer, grad in enumerate(grads):
            self.log_event(name + 'grad-layer%d' % layer, self.grad_norm(grad).numpy(), epoch)
        for i, layer in enumerate(self.model.layers):
            if not len(layer.get_weights()):
                continue
            self.log_event(name + 'bias-layer%d' % i, self.grad_norm(layer.get_weights()[1]).numpy(), epoch)
            self.log_event(name + 'weight-layer%d' % i, self.grad_norm(layer.get_weights()[0]).numpy(), epoch)

    def train(self, epochs, lambda_, print_=False, eval_every=10):
        d1 = [_ for _ in self.envs.e1]
        d2 = [_ for _ in self.envs.e2]
        x1, y1 = tf.convert_to_tensor([_[0] for _ in d1]), tf.convert_to_tensor([_[1] for _ in d1])
        x2, y2 = tf.convert_to_tensor([_[0] for _ in d2]), tf.convert_to_tensor([_[1] for _ in d2])
        for epoch in range(epochs):
            for envname, x, y in (('e1', x1, y1), ('e2', x2, y2)):
                loss, gradients = self.compute_gradients(x, y, lambda_=lambda_(epoch))#, use_loss=lambda_(epoch)<=1)
                # nullify gradients of dummy variable
                gradients = [grad if 'dummy' not in var.name else 0*grad for grad, var in zip(gradients, self.model.trainable_variables)]
                for grad, var in zip(gradients, self.model.trainable_variables):
                    if 'dummy' in var.name:
                        assert tf.reduce_sum(grad).numpy() == 0
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.report_grad_norms(loss, gradients, envname, epoch)
            if epoch % eval_every == 0:
                self.do_evaluations(epoch, lambda_=lambda_(epoch), print_=print_)
        self.do_evaluations(epochs, lambda_=lambda_(epoch), print_=print_)

    def train_batch(self, epochs, lambda_, batch_size=128, print_=False, eval_every=1):
        for epoch in range(epochs):
            d1 = self.envs.e1.shuffle(2 * batch_size).batch(batch_size).__iter__()
            d2 = self.envs.e2.shuffle(2 * batch_size).batch(batch_size).__iter__()
            batch = 0
            while True:
                try:
                    x1, y1 = d1.next()
                    x2, y2 = d2.next()
                    total_loss, total_grads = self.compute_gradients(
                        [x1, x2], [y1, y2],
                        lambda_=lambda_(epoch), batch=batch, epoch=epoch
                    )
                    for grads in total_grads:
                        self.report_grad_norms(total_loss, grads, 'total-',
                                               batch, epoch, report_weight=True)
                    for grads in total_grads:
                        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    batch += 1
                except StopIteration:
                    break
            if epoch % eval_every == 0:
                self.do_evaluations(epoch, batch_size=batch_size, print_=print_)
        self.do_evaluations(epochs, batch_size=batch_size, print_=print_)
