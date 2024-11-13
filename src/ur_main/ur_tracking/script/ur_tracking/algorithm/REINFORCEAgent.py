'''
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from sklearn.utils import shuffle

class REINFORCEAgent:
    def __init__(self, obs_dim: int, n_act: int, epochs: int = 10, lr: float = 3e-5, hdim: int = 64, 
                 max_std: float = 1.0, seed: int = 0) -> None:
        """
        初始化REINFORCE Agent.

        :param obs_dim: 观测空间的维度
        :param n_act: 动作空间的维度
        :param epochs: 每次更新的迭代次数
        :param lr: 学习率
        :param hdim: 隐藏层维度
        :param max_std: 最大标准差
        :param seed: 随机种子
        """
        self.seed = seed
        self.obs_dim = obs_dim
        self.n_act = n_act
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self.max_std = max_std

        # Set random seeds
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Build the model
        self._build_model()

    def _build_model(self) -> None:
        """
        使用Keras Sequential API 构建策略网络和优化器。
        """
        # Define the policy network using the Keras Sequential API
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hdim, activation='tanh',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                  input_shape=(self.obs_dim,)),
            tf.keras.layers.Dense(self.hdim, activation='tanh',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed)),
            tf.keras.layers.Dense(self.n_act,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed))
        ])

        # Define logstd as a trainable variable
        self.logstd = tf.Variable(tf.zeros((1, self.n_act)), dtype=tf.float32, name="logstd")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _compute_loss(self, observes: np.ndarray, actions: np.ndarray) -> tf.Tensor:
        """
        计算损失值.

        :param observes: 观测数据，形状为 (batch_size, obs_dim)
        :param actions: 实际采取的动作，形状为 (batch_size, n_act)
        :return: 损失值的张量
        """
        mean = self.model(observes)
        std = tf.exp(self.logstd)
        normalized_actions = (actions - mean) / std
        loss = 0.5 * tf.reduce_sum(tf.square(normalized_actions), axis=1)
        return tf.reduce_mean(loss)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        基于当前策略采样一个动作.

        :param obs: 当前的观测值，形状为 (obs_dim,)
        :return: 采样得到的动作，形状为 (n_act,)
        """
        obs = np.array(obs)
        mean = self.model(obs)
        std = tf.exp(self.logstd)
        sampled_action = mean + std * tf.random.normal(shape=mean.shape)
        return sampled_action.numpy()[0]

    def control(self, obs: np.ndarray) -> int:
        """
        计算在给定观测下，最大概率的动作.

        :param obs: 当前的观测值，形状为 (obs_dim,)
        :return: 最大概率的动作（对应的索引）
        """
        obs = np.array(obs)
        mean = self.model(obs)
        return np.argmax(mean.numpy())

    def update(self, observes: np.ndarray, actions: np.ndarray, scores: np.ndarray, batch_size: int = 128) -> float:
        """
        使用观测、动作和得分进行策略更新.

        :param observes: 观测数据，形状为 (num_samples, obs_dim)
        :param actions: 动作数据，形状为 (num_samples, n_act)
        :param scores: 每个动作的得分，形状为 (num_samples,)
        :param batch_size: 每次更新的样本数量
        :return: 训练后的损失值
        """
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        for epoch in range(self.epochs):
            observes, actions, scores = shuffle(observes, actions, scores, random_state=self.seed)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs_batch = observes[start:end]
                act_batch = actions[start:end]

                with tf.GradientTape() as tape:
                    loss = self._compute_loss(obs_batch, act_batch)

                gradients = tape.gradient(loss, self.model.trainable_variables + [self.logstd])
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables + [self.logstd]))
        
        # Return the loss after updates for logging purposes
        return float(loss.numpy())

    def close_sess(self) -> None:
        """
        关闭会话（在 TensorFlow 2.x 中不需要，但保持接口一致）。
        """
        pass
'''



"""
Old version below

"""
from collections import deque
import numpy as np
#import tensorflow as tf
from sklearn.utils import shuffle
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

class REINFORCEAgent(object):
    def __init__(self, obs_dim, n_act,
                 epochs=10, lr=3e-5, hdim=64, max_std=1.0,
                 seed=0):
        
        self.seed=0
        
        self.obs_dim = obs_dim
        self.n_act = n_act
        
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self.max_std = max_std
        
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._normal_act()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.variables = tf.global_variables()
            
    def _placeholders(self):
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.n_act), 'act') #(None, )
        self.score_ph = tf.placeholder(tf.float32, (None,), 'score')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        
    def _policy_nn(self):        
        hid1_size = self.hdim
        hid2_size = self.hdim
        
        # TWO HIDDEN LAYERS
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h2")
               
        self.logits = tf.layers.dense(out, self.n_act,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="logits")

        # Output fully connected layer 
        self.output_placeholder = tf.layers.dense(out,  self.n_act,
                                             activation=tf.tanh,
                                             use_bias=True,
                                             kernel_initializer=tf.keras.initializers.glorot_uniform(), # tf.contrib.layers.xavier_initializer()
                                             name="output")
        
        self.logstd = tf.Variable(tf.zeros([1, self.n_act]),
                                dtype=tf.float32, name="logstd") 

        self.sample_action = self.output_placeholder + tf.exp(self.logstd) * tf.random_normal(tf.shape(self.output_placeholder)) # shape [2, ]

        # SOFTMAX POLICY
        self.pi = self.output_placeholder
        #self.pi = tf.nn.softmax(self.logits)
        
        # SAMPLE OPERATION
        #categorical = tf.distributions.Categorical(logits=self.logits)
        #self.sample_action = categorical.sample(1,seed=self.seed)
        
    def _normal_act(self):
        # PROBABILITY WITH TRAINING PARAMETER        
        #one_hot_act = tf.one_hot(self.act_ph, self.n_act)
        self.action_normalized = (self.act_ph - self.output_placeholder) / tf.exp(self.logstd) # shape [2, ]
        #self.log_p = - 0.5 * tf.reduce_sum(tf.square(action_normalized), axis=1)
        #self.log_p = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_act, logits=self.logits)
        
    def _loss_train_op(self):
        
        # REINFORCE OBJECTIVE
        #self.loss = -tf.reduce_mean(self.score_ph*self.log_p)
        self.loss = - 0.5 * tf.reduce_sum(tf.square(self.action_normalized), axis=1)

        # OPTIMIZER 
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(self.init)

    def get_action(self, obs): # SAMPLE FROM POLICY
        feed_dict = {self.obs_ph: obs}  
        sampled_action, = self.sess.run(self.sample_action,feed_dict=feed_dict) # shape [2, ]
        #return sampled_action[0]
        return sampled_action
    
    def control(self, obs): # COMPUTE MAX PROB
        feed_dict = {self.obs_ph: obs}
        best_action = np.argmax(self.sess.run(self.pi,feed_dict=feed_dict))
        return best_action        
    
    def update(self, observes, actions, scores, batch_size = 128): # TRAIN POLICY        
        #print ('########### TRAIN POLICY ###############')
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches
        
        for e in range(self.epochs):
            observes, actions, scores = shuffle(observes, actions, scores, random_state=self.seed)
            '''
            print ("######################update######################")
            ('observes: ', (100, 15), <type 'numpy.ndarray'>)
            ('actions: ', (100, 6), <type 'numpy.ndarray'>)
            ('rewards: ', (100,), <type 'numpy.ndarray'>)
            '''
            for j in range(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: observes[start:end,:],
                     self.act_ph: actions[start:end],
                     self.score_ph: scores[start:end],
                     self.lr_ph: self.lr}        
                self.sess.run(self.train_op, feed_dict)
            
        feed_dict = {self.obs_ph: observes,
             self.act_ph: actions,
             self.score_ph: scores,
             self.lr_ph: self.lr}               
        loss  = self.sess.run(self.loss, feed_dict) 
        return np.mean(loss)
    
    def close_sess(self):
        self.sess.close()