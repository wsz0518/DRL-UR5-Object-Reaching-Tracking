import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from sklearn.utils import shuffle

# 和 PPO 类似，先定义一个 PolicyNetwork
class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hdim, max_std, seed=0):
        super().__init__()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # 两层隐藏层
        self.hid1 = layers.Dense(hdim, activation='tanh',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                 name='policy_h1')
        self.hid2 = layers.Dense(hdim, activation='tanh',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                 name='policy_h2')
        # 均值输出层
        self.mean_layer = layers.Dense(act_dim,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                       name='mean')
        # 可训练的对数标准差 logits_std，用 sigmoid 做范围限制
        self.logits_std = tf.Variable(
            initial_value=tf.random.normal(shape=(1,), stddev=0.01, seed=seed),
            trainable=True, name='logits_std'
        )
        self.max_std = max_std

    def call(self, obs):
        """
        前向传播，输出: (mean, std)
        obs: shape=[batch_size, obs_dim]
        """
        x = self.hid1(obs)
        x = self.hid2(x)
        mean = self.mean_layer(x)  # [batch_size, act_dim]
        # 用 sigmoid 做限制，让 std  ∈ (0, max_std]
        std = self.max_std * tf.ones_like(mean) * tf.sigmoid(self.logits_std)
        return mean, std


class REINFORCEAgent(object):
    def __init__(self, obs_dim, act_dim,
                 epochs=10, lr=3e-5, hdim=64, max_std=1.0, seed=0):
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self.max_std = max_std

        # ==============================
        # 构建策略网络 + 优化器
        # ==============================
        self.policy_network = PolicyNetwork(obs_dim=self.obs_dim,
                                            act_dim=self.act_dim,
                                            hdim=self.hdim,
                                            max_std=self.max_std,
                                            seed=self.seed)
        self.policy_optimizer = optimizers.Adam(learning_rate=self.lr)

    def log_prob(self, act, mean, std):
        """
        和 PPO 中一样，计算给定 (mean,std) 下每个样本动作 act 的对数概率。
        act, mean, std 形状都是 [batch_size, act_dim]
        返回 [batch_size]，表示每条样本的 log_prob。
        """
        # 高斯分布 logprob：-0.5*((y-mu)/sigma)^2 - log(sigma) - 0.5*log(2π)
        # 多维动作 => 各维度对数概率之和
        logp_per_dim = -0.5 * tf.square((act - mean) / std) \
                       - tf.math.log(std) \
                       - 0.5 * np.log(2.0 * np.pi)
        logp = tf.reduce_sum(logp_per_dim, axis=1)  # 对动作维度累加
        return logp

    @tf.function
    def train_step(self, obs_batch, act_batch, score_batch):
        """
        类似于 PPO 里的 train_step, 但这里不需要 ratio、clip 等,
        只用 REINFORCE 核心:  loss = - mean( score * logπ(a|s) )
        """
        with tf.GradientTape() as tape:
            mean, std = self.policy_network(obs_batch)
            logp = self.log_prob(act_batch, mean, std)
            # if shape is [batch_size,1], reshape to [batch_size]
            score_batch = tf.reshape(score_batch, [-1])
            loss_per_sample = - score_batch * logp
            loss = tf.reduce_mean(loss_per_sample)
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        return loss

    def compute_losses(self, obs, act, score):
        """
        类似 PPO 的 compute_losses, 但这里只有 policy loss。
        不涉及 advantage/clip/value 等。
        """
        mean, std = self.policy_network(obs)
        logp = self.log_prob(act, mean, std)
        score = tf.reshape(score, [-1])
        loss_per_sample = - score * logp
        policy_loss = tf.reduce_mean(loss_per_sample)
        return policy_loss

    def get_action(self, obs):
        """
        从当前策略中采样动作: a = mean + std*noise
        obs 可以是单步 [obs_dim] 或者 batch [batch_size, obs_dim]
        """
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        if len(obs_tf.shape) == 1:
            obs_tf = tf.expand_dims(obs_tf, axis=0)
        mean, std = self.policy_network(obs_tf)
        noise = tf.random.normal(tf.shape(mean), seed=self.seed)
        sampled_action = mean + std * noise
        print(sampled_action[0].numpy())
        return sampled_action[0].numpy()  # 只取第一个样本

    def control(self, obs):
        """
        确定性动作（直接输出 mean)
        """
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        if len(obs_tf.shape) == 1:
            obs_tf = tf.expand_dims(obs_tf, axis=0)
        mean, std = self.policy_network(obs_tf)
        return mean[0].numpy()

    def update(self, observes, actions, scores, batch_size=128):
        """
        效仿 PPO 里的 update, 使用 tf.data.Dataset + mini-batch + shuffle。
        但核心公式仍是 REINFORCE: - E[score * logπ(a)]
        """
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        # 打乱数据
        observes, actions, scores = shuffle(observes, actions, scores, random_state=self.seed)

        # 建立 dataset
        dataset = tf.data.Dataset.from_tensor_slices((observes, actions, scores))
        dataset = dataset.batch(batch_size)

        # 多个 epoch 训练
        for e in range(self.epochs):
            for batch in dataset:
                obs_batch, act_batch, score_batch = batch
                obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
                act_batch = tf.convert_to_tensor(act_batch, dtype=tf.float32)
                score_batch = tf.convert_to_tensor(score_batch, dtype=tf.float32)
                self.train_step(obs_batch, act_batch, score_batch)

        # 最终计算并返回一个loss供日志记录
        obs_tf = tf.convert_to_tensor(observes, dtype=tf.float32)
        act_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
        sco_tf = tf.convert_to_tensor(scores, dtype=tf.float32)
        final_loss = self.compute_losses(obs_tf, act_tf, sco_tf)
        return final_loss.numpy()

    def save_model(self, policy_weights_path):
        """
        保存策略网络权重 (模仿 PPO 里的写法)
        """
        dummy_obs = tf.zeros(shape=(1, self.obs_dim), dtype=tf.float32)
        self.policy_network(dummy_obs)  # 先forward一次，触发build
        self.policy_network.save_weights(policy_weights_path)

    def load_model(self, policy_weights_path):
        """
        加载策略网络权重
        """
        dummy_obs = tf.zeros(shape=(1, self.obs_dim), dtype=tf.float32)
        self.policy_network(dummy_obs)  # 先build
        self.policy_network.load_weights(policy_weights_path)



# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, optimizers
# from sklearn.utils import shuffle

# # ==============================
# # 迁移过来的 PolicyNetwork (仅用于 REINFORCE)
# # ==============================
# class PolicyNetwork(tf.keras.Model):
#     def __init__(self, obs_dim, act_dim, hdim, max_std, seed=0):
#         super().__init__()
#         tf.random.set_seed(seed)
#         np.random.seed(seed)

#         # 和 PPOAgent 中类似的初始化方式
#         self.hid1 = layers.Dense(hdim, activation='tanh',
#                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
#                                  name='policy_h1')
#         self.hid2 = layers.Dense(hdim, activation='tanh',
#                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
#                                  name='policy_h2')
#         self.mean_layer = layers.Dense(act_dim,
#                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
#                                        name='mean')

#         # logstd 用 sigmoid 加上一个缩放，类似 PPOAgent 中的写法
#         # 使其可学习，但又不会无限大或负到崩溃
#         self.logits_std = tf.Variable(
#             initial_value=tf.random.normal(shape=(1,), stddev=0.01, seed=seed),
#             trainable=True, name='logits_std'
#         )
#         self.max_std = max_std

#     def call(self, obs):
#         """
#         前向传播：输出 (mean, std)
#         obs: shape = [batch_size, obs_dim]
#         """
#         x = self.hid1(obs)
#         x = self.hid2(x)
#         mean = self.mean_layer(x)  # shape=[batch_size, act_dim]

#         # std 范围受 sigmoid & max_std 限制
#         std = self.max_std * tf.ones_like(mean) * tf.sigmoid(self.logits_std)  # shape=[1,]
#         # 为了和 mean 形状匹配，这里做一个 broadcast , shape=[batch_size, act_dim]
#         return mean, std


# # ==============================
# # REINFORCE Agent (迁移 PPO 写法，但核心不变)
# # ==============================
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

#         # ===== 类似 PPOAgent：建立一个 PolicyNetwork =====
#         self.policy_network = PolicyNetwork(obs_dim=self.obs_dim,
#                                             act_dim=self.n_act,
#                                             hdim=self.hdim,
#                                             max_std=self.max_std,
#                                             seed=self.seed)

#         # ===== 优化器也使用 Adam =====
#         self.optimizer = optimizers.Adam(learning_rate=self.lr)

#     # def log_prob(self, act, mean, std):
#     #     """
#     #     计算高斯分布 N(mean, std^2) 的 log probability。
#     #     act, mean, std 均为 shape=[batch_size, action_dim]。
#     #     返回 shape=[batch_size]，对应每条样本的对数概率。
        
#     #     对多维度动作做独立高斯的假设，因此是各维度对数概率之和。
#     #     """
#     #     # shape: (batch_size, action_dim)
#     #     var = tf.square(std)
#     #     log_std = tf.math.log(std)

#     #     # log(N(x; mean, std)) = -0.5 * sum( ((x-mean)/std)^2 ) - sum(log_std) - (D/2)*log(2π)
#     #     # 这里跟 PPOAgent 一样省去常数 or 只做 axis=1 的 reduce_sum
#     #     # 若要更严谨，需包含 (D/2)*log(2π)；对梯度无影响，这里可省略
#     #     logp_per_dim = -0.5 * tf.square((act - mean) / std) - log_std
#     #     logp = tf.reduce_sum(logp_per_dim, axis=1)
#     #     return logp

#     def log_prob(self, y, mu, sigma):
#         # 计算给定均值和标准差的高斯分布的对数概率
#         return tf.reduce_sum(-0.5 * tf.square((y - mu) / sigma) -
#                              tf.math.log(sigma) - 0.5 * np.log(2.0 * np.pi), axis=1)

#     @tf.function
#     def _compute_loss(self, obs, act, score):
#         """
#         REINFORCE loss:  - E[ score * log_pi(a|s) ]

#         obs:    [batch_size, obs_dim]
#         act:    [batch_size, act_dim]
#         score:  [batch_size]  (或 [batch_size,1])
#         """
#         mean, std = self.policy_network(obs)
#         logp = self.log_prob(act, mean, std)  # [batch_size]
#         # 注意是 - score * logp
#         # score 的形状如果是 [batch_size,1], 这里最好先 squeeze 一下
#         score = tf.reshape(score, [-1])  # 确保是 [batch_size]
#         loss_per_sample = - score * logp
#         loss = tf.reduce_mean(loss_per_sample)
#         return loss

#     def get_action(self, obs):
#         """
#         采样动作： mean + std * noise
#         obs: shape=[obs_dim] or [1, obs_dim]
#         返回 shape=[act_dim]
#         """
#         obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
#         if len(obs_tf.shape) == 1:
#             obs_tf = tf.expand_dims(obs_tf, axis=0)  # 变成 batch_size=1

#         mean, std = self.policy_network(obs_tf)  # shape=[1, act_dim]
#         noise = tf.random.normal(tf.shape(mean), seed=self.seed)
#         sampled_action = std * noise + mean*0.01
#         # print("std: ", std)
#         # print("noise:", noise)
#         # print("mean:" , mean)
#         return sampled_action[0].numpy()

#     def control(self, obs):
        
#         obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
#         if len(obs_tf.shape) == 1:
#             obs_tf = tf.expand_dims(obs_tf, axis=0)

#         mean, std = self.policy_network(obs_tf)
#         return mean[0].numpy()

#     def update(self, observes, actions, scores, batch_size=128):
#         """
#         REINFORCE 训练循环 (类似之前的写法)
#         """
#         num_batches = max(observes.shape[0] // batch_size, 1)
#         batch_size = observes.shape[0] // num_batches

#         for e in range(self.epochs):
#             # 打乱数据
#             observes, actions, scores = shuffle(observes, actions, scores, random_state=self.seed)
            
#             # 分成小批量
#             for j in range(num_batches):
#                 start = j * batch_size
#                 end = (j + 1) * batch_size

#                 obs_batch = tf.convert_to_tensor(observes[start:end, :], dtype=tf.float32)
#                 act_batch = tf.convert_to_tensor(actions[start:end], dtype=tf.float32)
#                 sco_batch = tf.convert_to_tensor(scores[start:end], dtype=tf.float32)

#                 with tf.GradientTape() as tape:
#                     loss = self._compute_loss(obs_batch, act_batch, sco_batch)
#                 grads = tape.gradient(loss, self.policy_network.trainable_variables)
#                 self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

#         # 计算最终 loss 仅做返回或记录
#         obs_tf_all = tf.convert_to_tensor(observes, dtype=tf.float32)
#         act_tf_all = tf.convert_to_tensor(actions, dtype=tf.float32)
#         sco_tf_all = tf.convert_to_tensor(scores, dtype=tf.float32)
#         final_loss = self._compute_loss(obs_tf_all, act_tf_all, sco_tf_all)
#         return final_loss.numpy()

#     def save_model(self, policy_weights_path):
#         """
#         保存当前策略网络
#         """
#         # 让网络先跑一次 forward，确保 build 完成
#         dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
#         self.policy_network(dummy_obs)
#         self.policy_network.save_weights(policy_weights_path)

#     def load_model(self, policy_weights_path):
#         """
#         加载策略网络
#         """
#         dummy_obs = tf.zeros((1, self.obs_dim), dtype=tf.float32)
#         self.policy_network(dummy_obs)  # 先 build 一下
#         self.policy_network.load_weights(policy_weights_path)






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