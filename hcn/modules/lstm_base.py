import os

import tensorflow as tf


class LSTMBase(object):
    MODEL_NAME = 'hcn.ckpt'

    def __init__(self, in_vocab, in_config, in_feature_vector_size, in_action_size, model_folder=None, trainable_vars=[], standalone=True):
        self.config = in_config
        self.feature_vector_size = in_feature_vector_size
        self.max_input_length = in_config['max_input_length']
        self.nb_hidden = self.config['embedding_size']
        self.action_size = in_action_size
        self.batch_size = self.config['batch_size']
        self.max_sequence_length = self.config['max_sequence_length']
        self.model_folder = model_folder
        self.trainable_vars = trainable_vars
        self.vocab = in_vocab
        self.global_step = tf.Variable(0, trainable=False)
        self.sess = tf.Session()  #config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=in_config.get('gpu_memory_fraction', 1.0))))
        # build graph
        self.__graph__()
        # start a session; attach to self
        if standalone:
            self.sess.run(tf.global_variables_initializer())
        if self.model_folder:
            self.restore(self.model_folder)
        # self.sess.run(tf.assign(self.initial_weights, tf.concat([tf.reshape(var.initialized_value(), (-1,)) for var in tf.trainable_variables()], axis=-1)))

    def __graph__(self):
        raise NotImplementedError

    def reset_state(self):
        pass
        # set init state to zeros
        # self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        # self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)

    # save session to checkpoint
    def save(self, in_folder):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(self.sess, os.path.join(in_folder, LSTMBase.MODEL_NAME))
        print('\n:: saved to {}\n'.format(os.path.join(in_folder, LSTMBase.MODEL_NAME)))

    # restore session from checkpoint
    def restore(self, in_folder):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        ckpt = tf.train.get_checkpoint_state(in_folder)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('\n:: <ERR> checkpoint not found! \n')
