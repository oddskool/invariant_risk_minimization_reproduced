import numpy as np
from collections import defaultdict
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Multiply
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPool3D
from tensorflow.keras.regularizers import l2


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

                                                  
def get_model(args):
    n_final_units=args.nb_units
    mlp=args.mlp
    l2reg=args.l2_reg
    n_layers=args.nb_layers
    
    input_images = Input(shape=(28, 28, 3))
    
    if not mlp:
        cnn = Conv2D(32, kernel_size=(3, 3),
                     activation='relu')(input_images)
        cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
        cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
        cnn = Dropout(0.25)(cnn)
        cnn = Flatten()(cnn)
        representation = cnn
    else:
        d = Flatten()(input_images)
        for _ in range(n_layers):
            d = Dense(n_final_units, activation='relu')(d)
        representation = d
    
    if not mlp:
        final = Dense(32, activation='relu')(representation)
        final = Dropout(0.5)(final)
    else:
        final = representation
    
    final = Dense(1, kernel_regularizer=l2(l2reg), name='env')(final)
        
    model = Model(
        inputs=[input_images],
        outputs=[final]
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
    
    def __init__(self, model, optimizer = tf.keras.optimizers.Adam()):
        self.model = model
        self.optimizer = optimizer
        self.envs = ColoredMNISTEnvironments()
#         self.dummy = tf.convert_to_tensor([1.])
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
            dummy = tf.convert_to_tensor([1.])
            with tf.GradientTape() as tape:
                tape.watch(dummy)
                logits = self.model(x, training=False)
                dummy_loss = self.loss(y, logits) * dummy
            batch_grads = tape.gradient(dummy_loss, [dummy])
            per_batch_penalties += [
                tf.math.reduce_sum(tf.math.square(
                    dummy_loss * batch_grads
                ))
            ]
            loss.update_state(y, logits)
            accuracy.update_state(y, tf.math.greater(tf.keras.activations.sigmoid(logits), .5))
        return loss.result().numpy(), accuracy.result().numpy(), tf.reduce_mean(per_batch_penalties)
    
    def compute_penalty(self, x, y):
        dummy = tf.convert_to_tensor([1.])
        with tf.GradientTape() as tape:
            tape.watch(dummy)
            dummy_logits = self.model(x, training=True)
            dummy_loss = self.loss(y, dummy_logits * dummy)
        dummy_grads = tape.gradient(dummy_loss, dummy)
        dummy_penalty = tf.math.reduce_sum(dummy_grads ** 2)
        return dummy_loss, dummy_penalty
        
    def batch_gradients(self, x, y, penalty, gradclip=1.0):
        with tf.GradientTape() as tape:
            tape.watch(penalty)
            logits = self.model(x, training=True)
            loss_value = self.loss(y, logits) + penalty
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        #grads = tf.clip_by_value(grads, -1, 1)
        return loss_value, grads
    
    def compute_gradients(self, x_s, y_s, lambda_=1, gradclip=1.0, batch=0, epoch=0):
        # compute penalty
        dummy = tf.convert_to_tensor([1.])
        with tf.GradientTape() as tape:
            tape.watch(dummy)
            dummy_logits = [self.model(x_s[i], training=True) for i in range(len(x_s))]
            dummy_losses = [self.loss(y_s[i], dummy * dummy_logits[i]) for i in range(len(x_s))]
            dummy_loss = tf.reduce_mean(dummy_losses)
        dummy_grads = tape.gradient(dummy_loss, dummy)
        self.report_grad_norms(dummy_grads, 'grad-pen-', batch, epoch)
        penalty = tf.reduce_sum(dummy_grads ** 2)
        # compute model gradients
        with tf.GradientTape() as tape:
            logits = [self.model(x_s[i], training=True) for i in range(len(x_s))]
            losses = [self.loss(y_s[i], logits[i]) + lambda_*penalty for i in range(len(x_s))]
            total_loss = tf.reduce_mean(losses)
        total_grads = tape.gradient(total_loss, self.model.trainable_variables)
#         total_grads = tf.clip_by_global_norm([total_grads], 100)
        return total_loss, total_grads
    
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
            print('ood loss %.5f acc: %.3f'%(ood_loss, ood_acc))   
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
        
    def grad_norm(self, grad):
        return tf.math.sqrt(tf.reduce_sum(tf.math.square(grad)))
    
    def report_grad_norms(self, grads, name, batch, epoch):
        for layer, grad in enumerate(grads):
            self.log_event(name + 'layer%d'%layer, self.grad_norm(grad).numpy(), epoch*10000+batch)
                
    def train(self, epochs, lambda_, batch_size=128, print_=False):
        for epoch in range(epochs):
            d1 = self.envs.e1.shuffle(2*batch_size).batch(batch_size).__iter__()
            d2 = self.envs.e2.shuffle(2*batch_size).batch(batch_size).__iter__()
            batch = 0
            while True:
                try:
                    x1, y1 = d1.next()
                    x2, y2 = d2.next()
                    total_loss, total_grads = self.compute_gradients([x1, x2], [y1, y2], lambda_=lambda_(epoch), 
                                                                    batch=batch, epoch=epoch)
                    if batch == 0:
                        self.report_grad_norms(total_grads, 'grad-', batch, epoch)
                    self.optimizer.apply_gradients(zip(total_grads, self.model.trainable_variables))
                    batch += 1
                except StopIteration:
                    break
            self.do_evaluations(epoch, batch_size=batch_size, print_=print_)
