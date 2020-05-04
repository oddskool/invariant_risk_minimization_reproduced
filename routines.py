import numpy as np
from collections import defaultdict
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Multiply
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPool3D


class ColoredMNISTEnvironments():
    
    def __init__(self):
        
        self.__load_initial_data()
        self.__create_envs()
        self.__create_validation_envs()

    def __load_initial_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # convert to RGB
        x_train = np.stack((x_train,)*3, axis=-1)
        x_test = np.stack((x_test,)*3, axis=-1)

        # normalize
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # binary label
        y_train = (y_train < 5).astype(int)
        y_test = (y_test < 5).astype(int)
        
        self.original_data = {
            'x_train':x_train,
            'x_test':x_test,
            'y_train':y_train,
            'y_test':y_test
        }
        
    def __create_envs(self):
        k=10**4
        self.e1 = self.__create_env(self.original_data['x_train'][:k], 
                                    self.original_data['y_train'][:k], .1)
        self.e2 = self.__create_env(self.original_data['x_train'][k:2*k], 
                                    self.original_data['y_train'][k:2*k], .2)
        self.e3 = self.__create_env(self.original_data['x_train'][2*k:3*k], 
                                    self.original_data['y_train'][2*k:3*k], .9)
        
    def __create_validation_envs(self):
        k=10**4
        i=3*k
        self.e11 = self.__create_env(self.original_data['x_train'][i:i+k], 
                                     self.original_data['y_train'][i:i+k], .1)
        self.e22 = self.__create_env(self.original_data['x_train'][i+k:i+2*k], 
                                     self.original_data['y_train'][i+k:i+2*k], .2)
        self.e33 = self.__create_env(self.original_data['x_train'][i+2*k:i+3*k], 
                                     self.original_data['y_train'][i+2*k:i+3*k], .9)
        
    def __create_env(self, x, y, e, labelflip_proba=.25):
        x = x.copy()
        y = y.copy()

        y = np.logical_xor(
            y,
            (np.random.random(size=len(y)) < labelflip_proba).astype(int)
        ).astype(int)

        color = np.logical_xor(
            y,
            (np.random.random(size=len(y)) < e).astype(int)
        )

        x[color, :, :, 2] = 0
        x[color, :, :, 1] = 0
        return tf.data.Dataset.from_tensor_slices((x, y))

                                                  
def get_model(n_final_units=32, compile=False):
    
    input_images = Input(shape=(28, 28, 3))
    
    cnn = Conv2D(32, kernel_size=(3, 3),
                 activation='relu')(input_images)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = Dropout(0.25)(cnn)
    cnn = Flatten()(cnn)
    
    env1 = Dense(32, activation='relu')(cnn)
    env1 = Dropout(0.5)(env1)
    env1 = Dense(1, name='env1')(env1)
        
    model = Model(
        inputs=[input_images],
        outputs=[env1]
    )
    
    if compile:
        model.compile(
            loss=[
                tf.keras.losses.binary_crossentropy,
            ],
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=['accuracy']
        )
    return model


from collections import defaultdict
from datetime import datetime
import random

class IRMModel(object):
    
    def __init__(self, model = get_model(), optimizer = tf.keras.optimizers.Adam()):
        self.model = model
        self.optimizer = optimizer
        self.envs = ColoredMNISTEnvironments()
        self.dummy = tf.convert_to_tensor([1.])
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.logs = defaultdict(list)
        self.logdir = "/home/e.diemert/tflogs/scalars/" + \
            datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + \
                                    '-%d'%random.randint(0,1000)
        print("logging to", self.logdir)
        self.file_writer = tf.summary.create_file_writer(self.logdir + "/metrics")
        self.file_writer.set_as_default()

        
    def evaluate(self, env):
        accuracy = tf.keras.metrics.Accuracy()
        loss = tf.keras.metrics.BinaryCrossentropy()
        per_batch_penalties = []
        for (batch, (x, y)) in enumerate(env):
            with tf.GradientTape() as tape:
                tape.watch(self.dummy)
                logits = self.model(x, training=False)
                dummy_loss = self.loss(y, logits) * self.dummy
            batch_grads = tape.gradient(dummy_loss, [self.dummy])
            per_batch_penalties += [
                tf.math.square(
                    dummy_loss * batch_grads
                )
            ]
            loss.update_state(y, logits)
            accuracy.update_state(y, tf.math.greater(tf.keras.activations.sigmoid(logits), .5))
        return loss.result().numpy(), accuracy.result().numpy(), tf.reduce_mean(per_batch_penalties)
    
    def compute_penalty(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(self.dummy)
            dummy_logits = self.model(x, training=True)
            dummy_loss = self.loss(y, dummy_logits * self.dummy)
        dummy_grads = tape.gradient(dummy_loss, self.dummy)
        dummy_penalty = dummy_grads ** 2
        return dummy_loss, dummy_penalty
        
    def batch_gradients(self, x, y, penalty):
        with tf.GradientTape() as tape:
            tape.watch(penalty)
            logits = self.model(x, training=True)
            loss_value = self.loss(y, logits) + penalty
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        return loss_value, grads
    
    def do_evaluations(self, epoch, print_=True, batch_size=128):
        if print_:
            print('-'*80)
            print("epoch:", epoch)
        ood_loss, ood_acc, ood_penalty = self.evaluate(self.envs.e3.shuffle(2*batch_size).batch(batch_size))
        self.logs['ood-loss'] += [ood_loss]
        self.logs['ood-acc'] += [ood_acc]
        self.logs['ood-penalty'] += [ood_penalty.numpy()]
        self.log_event('ood_loss', ood_loss, epoch)
        self.log_event('ood_acc', ood_acc, epoch)
        self.log_event('ood_pen', ood_penalty.numpy(), epoch)
        if print_:
            print('ood  loss %.5f acc: %.3f'%(ood_loss, ood_acc))   
        for env_name, env in (('e11',self.envs.e11.shuffle(2*batch_size).batch(batch_size)), 
                              ('e22',self.envs.e22.shuffle(2*batch_size).batch(batch_size)),):
            env_loss, env_acc, env_penalty = self.evaluate(env)
            self.logs[env_name+'-test-loss'] += [env_loss]
            self.logs[env_name+'-test-acc'] += [env_acc]
            self.log_event(env_name+'_loss', env_loss, epoch)
            self.log_event(env_name+'_acc', env_acc, epoch)
            self.log_event(env_name+'_pen', env_penalty.numpy(), epoch)
            if print_:
                print('%s loss %.5f acc: %.3f'%(env_name, env_loss, env_acc))
        if print_:
            print('-'*80)

    def log_event(self, event, value, epoch):
        tf.summary.scalar(event, data=value, step=epoch)
                
    def train(self, epochs, lambda_, batch_size=128, print_=False):
        for epoch in range(epochs):
            d1 = self.envs.e1.shuffle(2*batch_size).batch(batch_size).__iter__()
            d2 = self.envs.e2.shuffle(2*batch_size).batch(batch_size).__iter__()
            batch = 0
            while True:
                try:
                    x1, y1 = d1.next()
                    x2, y2 = d2.next()
                    l1d, pen1 = self.compute_penalty(x1, y1)
                    l2d, pen2 = self.compute_penalty(x2, y2)
                    self.logs['e1'+'-train-penalty'] += pen1.numpy()
                    self.logs['e2'+'-train-penalty'] += pen2.numpy()
                    pen = tf.reduce_mean([pen1, pen2])
                    l1, grads1 = self.batch_gradients(x1, y1, lambda_(epoch) * pen)
                    l2, grads2 = self.batch_gradients(x2, y2, lambda_(epoch) * pen)
                    self.logs['e1'+'-train-loss'] += [l1]
                    self.logs['e2'+'-train-loss'] += [l2]
                    grads = [ tf.reduce_mean([grads1[_], grads2[_]], axis=0) for _ in range(len(grads1)) ]
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    if print_ and not batch % 10:
                        print("%4d"%batch, 
                              "tr-l1: %.5f"%l1d.numpy(), "tr-p1: %.5f"%(pen1.numpy()*lambda_(epoch)), 
                              "tr-l2: %.5f"%l2d.numpy(), "tr-p2: %.5f"%(pen2.numpy()*lambda_(epoch)))
                    batch += 1
                except StopIteration:
                    break
            self.do_evaluations(epoch, batch_size=batch_size, print_=print_)