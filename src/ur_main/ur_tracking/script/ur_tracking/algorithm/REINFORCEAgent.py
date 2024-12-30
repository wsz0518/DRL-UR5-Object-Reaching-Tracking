import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from sklearn.utils import shuffle

# ==============================
# 迁移过来的 PolicyNetwork (仅用于 REINFORCE)
# ==============================
class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hdim, max_std, seed=0):
        super().__init__()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # 和 PPOAgent 中类似的初始化方式
        self.hid1 = layers.Dense(hdim, activation='tanh',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                 name='policy_h1')
        self.hid2 = layers.Dense(hdim, activation='tanh',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                 name='policy_h2')
        self.mean_layer = layers.Dense(act_dim,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                       name='mean')

        # logstd 用 sigmoid 加上一个缩放，类似 PPOAgent 中的写法
        # 使其可学习，但又不会无限大或负到崩溃
        self.logits_std = tf.Variable(
            initial_value=tf.random.normal(shape=(1,), stddev=0.01, seed=seed),
            trainable=True, name='logits_std'
        )
        self.max_std = max_std

    def call(self, obs):
        """
        前向传播：输出 (mean, std)
        obs: shape = [batch_size, obs_dim]
        """
        x = self.hid1(obs)
        x = self.hid2(x)
        mean = self.mean_layer(x)  # shape=[batch_size, act_dim]

        # std 范围受 sigmoid & max_std 限制
        std = self.max_std * tf.sigmoid(self.logits_std)  # shape=[1,]
        # 为了和 mean 形状匹配，这里做一个 broadcast
        std = tf.ones_like(mean) * std  # shape=[batch_size, act_dim]
        return mean, std


# ==============================
# REINFORCE Agent (迁移 PPO 写法，但核心不变)
# ==============================
class REINFORCEAgent(object):
    def __init__(self, obs_dim, n_act,
                 epochs=10, lr=3e-5, hdim=64, max_std=1.0,
                 seed=0):
        
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.obs_dim = obs_dim
        self.n_act = n_act
        
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self.max_std = max_std

        # ===== 类似 PPOAgent：建立一个 PolicyNetwork =====
        self.policy_network = PolicyNetwork(obs_dim=self.obs_dim,
                                            act_dim=self.n_act,
                                            hdim=self.hdim,
                                            max_std=self.max_std,
                                            seed=self.seed)

        # ===== 优化器也使用 Adam =====
        self.optimizer = optimizers.Adam(learning_rate=self.lr)

    def log_prob(self, act, mean, std):
        """
        计算高斯分布 N(mean, std^2) 的 log probability。
        act, mean, std 均为 shape=[batch_size, action_dim]。
        返回 shape=[batch_size]，对应每条样本的对数概率。
        
        对多维度动作做独立高斯的假设，因此是各维度对数概率之和。
        """
        # shape: (batch_size, action_dim)
        var = tf.square(std)
        log_std = tf.math.log(std)

        # log(N(x; mean, std)) = -0.5 * sum( ((x-mean)/std)^2 ) - sum(log_std) - (D/2)*log(2π)
        # 这里跟 PPOAgent 一样省去常数 or 只做 axis=1 的 reduce_sum
        # 若要更严谨，需包含 (D/2)*log(2π)；对梯度无影响，这里可省略
        logp_per_dim = -0.5 * tf.square((act - mean) / std) - log_std
        logp = tf.reduce_sum(logp_per_dim, axis=1)
        return logp

    @tf.function
    def _compute_loss(self, obs, act, score):
        """
        REINFORCE loss:  - E[ score * log_pi(a|s) ]

        obs:    [batch_size, obs_dim]
        act:    [batch_size, act_dim]
        score:  [batch_size]  (或 [batch_size,1])
        """
        mean, std = self.policy_network(obs)
        logp = self.log_prob(act, mean, std)  # [batch_size]
        # 注意是 - score * logp
        # score 的形状如果是 [batch_size,1], 这里最好先 squeeze 一下
        score = tf.reshape(score, [-1])  # 确保是 [batch_size]
        loss_per_sample = - score * logp
        loss = tf.reduce_mean(loss_per_sample)
        return loss

    def get_action(self, obs):
        """
        采样动作： mean + std * noise
        obs: shape=[obs_dim] or [1, obs_dim]
        返回 shape=[act_dim]
        """
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        if len(obs_tf.shape) == 1:
            obs_tf = tf.expand_dims(obs_tf, axis=0)  # 变成 batch_size=1

        mean, std = self.policy_network(obs_tf)  # shape=[1, act_dim]
        noise = tf.random.normal(tf.shape(mean), seed=self.seed)
        sampled_action = mean*0.0 + std * noise
        return sampled_action[0].numpy()

    def control(self, obs):
        
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        if len(obs_tf.shape) == 1:
            obs_tf = tf.expand_dims(obs_tf, axis=0)

        mean, std = self.policy_network(obs_tf)
        return mean[0].numpy()

    def update(self, observes, actions, scores, batch_size=128):
        """
        REINFORCE 训练循环 (类似之前的写法)
        """
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        for e in range(self.epochs):
            # 打乱数据
            observes, actions, scores = shuffle(observes, actions, scores, random_state=self.seed)
            
            # 分成小批量
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size

                obs_batch = tf.convert_to_tensor(observes[start:end, :], dtype=tf.float32)
                act_batch = tf.convert_to_tensor(actions[start:end], dtype=tf.float32)
                sco_batch = tf.convert_to_tensor(scores[start:end], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    loss = self._compute_loss(obs_batch, act_batch, sco_batch)
                grads = tape.gradient(loss, self.policy_network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

        # 计算最终 loss 仅做返回或记录
        obs_tf_all = tf.convert_to_tensor(observes, dtype=tf.float32)
        act_tf_all = tf.convert_to_tensor(actions, dtype=tf.float32)
        sco_tf_all = tf.convert_to_tensor(scores, dtype=tf.float32)
        final_loss = self._compute_loss(obs_tf_all, act_tf_all, sco_tf_all)
        return final_loss.numpy()

    def save_model(self, policy_weights_path):
        """
        保存当前策略网络
        """
        # 让网络先跑一次 forward，确保 build 完成
        dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
        self.policy_network(dummy_obs)
        self.policy_network.save_weights(policy_weights_path)

    def load_model(self, policy_weights_path):
        """
        加载策略网络
        """
        dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
        self.policy_network(dummy_obs)  # 先 build 一下
        self.policy_network.load_weights(policy_weights_path)






# import numpy as np
# from sklearn.utils import shuffle
# import tensorflow as tf
# from tensorflow.keras import layers, initializers, optimizers

# class REINFORCEAgent(object):
#     def __init__(self, obs_dim, n_act,
#                  epochs=10, lr=3e-5, hdim=64, max_std=1.0,
#                  seed=0):
        
#         self.seed = seed
#         tf.random.set_seed(self.seed)
#         np.random.seed(self.seed)

#         self.obs_dim = obs_dim
#         self.n_act = n_act
        
#         self.epochs = epochs
#         self.lr = lr
#         self.hdim = hdim
#         self.max_std = max_std

#         # 构建策略网络（相当于_ policy_nn部分）
#         self.policy_model = self._build_policy_model()

#         # 定义logstd变量，与原逻辑一致
#         self.logstd = tf.Variable(tf.zeros([1, self.n_act]), dtype=tf.float32, name="logstd")

#         # 定义优化器
#         # self.optimizer = optimizers.Adam(learning_rate=self.lr)
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

#     def _build_policy_model(self):
#         # 与原逻辑保持相同的网络结构和初始化参数
#         initializer = initializers.RandomNormal(stddev=0.01, seed=self.seed)
#         inputs = tf.keras.Input(shape=(self.obs_dim,))
#         x = layers.Dense(self.hdim, activation='tanh', kernel_initializer=initializer, name="h1")(inputs)
#         x = layers.Dense(self.hdim, activation='tanh', kernel_initializer=initializer, name="h2")(x)
#         output = layers.Dense(self.n_act, activation='tanh', use_bias=True,
#                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
#                               name="output")(x)
#         model = tf.keras.Model(inputs=inputs, outputs=output)
#         return model

#     @tf.function
#     def _compute_loss(self, obs, act, score):
#         # 按原逻辑计算loss:
#         # self.action_normalized = (act_ph - output_placeholder) / tf.exp(self.logstd)
#         # self.loss = -0.5 * tf.reduce_sum(tf.square(self.action_normalized), axis=1)
        
#         output = self.policy_model(obs)  # 相当于 self.output_placeholder
#         action_normalized = (act - output) / tf.exp(self.logstd)
#         # loss_per_sample = -0.5 * tf.reduce_sum(tf.square(action_normalized), axis=1)
#         # 原代码最终返回 np.mean(loss) ，这里直接对loss求平均使其成为标量
#         log_prob_per_sample = -0.5 * tf.reduce_sum(tf.square(action_normalized), axis=1) \
#                               - tf.reduce_sum(self.logstd, axis=1)
#         # REINFORCE cost = - mean( score * log_prob )
#         loss_per_sample = - score * log_prob_per_sample
        
#         loss = tf.reduce_mean(loss_per_sample)
#         return loss

#         # # fake REINFORCE
#         # output = self.policy_model(obs)  # 相当于原来的 self.output_placeholder
#         # action_normalized = (act - output) / tf.exp(self.logstd)
        
#         # # 原来的损失：-0.5 * sum( (act - mean)^2 / std^2 ), 没有乘 score
#         # loss_per_sample = -0.5 * tf.reduce_sum(tf.square(action_normalized), axis=1)
        
#         # # 对 batch 求均值，得到一个标量
#         # loss = tf.reduce_mean(loss_per_sample)
#         # return loss

#     def get_action(self, obs):
#         # 在原代码中: sampled_action = output_placeholder + exp(logstd)*tf.random_normal()
#         obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
#         output = self.policy_model(obs_tf)
#         noise = tf.random.normal(tf.shape(output), seed=self.seed)
#         sampled_action = output + tf.exp(self.logstd) * noise
#         print(sampled_action)
#         return sampled_action[0].numpy() * 0.1

#     def control(self, obs):
#         # 计算max prob对应的动作(这里output是均值动作)
#         obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
#         output = self.policy_model(obs_tf)
#         best_action = np.argmax(output.numpy(), axis=1)[0]
#         return best_action

#     def update(self, observes, actions, scores, batch_size=128):
#         num_batches = max(observes.shape[0] // batch_size, 1)
#         batch_size = observes.shape[0] // num_batches

#         for e in range(self.epochs):
#             observes, actions, scores = shuffle(observes, actions, scores, random_state=self.seed)
            
#             for j in range(num_batches):
#                 start = j * batch_size
#                 end = (j + 1) * batch_size

#                 obs_batch = tf.convert_to_tensor(observes[start:end, :], dtype=tf.float32)
#                 act_batch = tf.convert_to_tensor(actions[start:end], dtype=tf.float32)
#                 score_batch = tf.convert_to_tensor(scores[start:end], dtype=tf.float32)
 
#                 with tf.GradientTape() as tape:
#                     loss = self._compute_loss(obs_batch, act_batch, score_batch)
#                 grads = tape.gradient(loss, self.policy_model.trainable_variables + [self.logstd])
#                 self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables + [self.logstd]))

#         # 最后计算整体loss
#         obs_tf_all = tf.convert_to_tensor(observes, dtype=tf.float32)
#         act_tf_all = tf.convert_to_tensor(actions, dtype=tf.float32)
#         sco_tf_all = tf.convert_to_tensor(scores, dtype=tf.float32)
#         final_loss = self._compute_loss(obs_tf_all, act_tf_all, sco_tf_all)
#         return final_loss.numpy()

#     def save_model(self, policy_weights_path):
#         # 保存当前策略网络的参数和logstd变量
#         dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
#         self.policy_model(dummy_obs)  # 触发build
#         self.policy_model.save_weights(policy_weights_path)

#         # 同时保存logstd值到np文件
#         np.save(policy_weights_path + '_logstd.npy', self.logstd.numpy())

#     def load_model(self, policy_weights_path):
#         # 加载策略网络参数和logstd变量
#         dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
#         self.policy_model(dummy_obs)  # 先build一下model
#         self.policy_model.load_weights(policy_weights_path)

#         logstd_vals = np.load(policy_weights_path + '_logstd.npy')
#         self.logstd.assign(logstd_vals)



"""
Old version below


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
"""