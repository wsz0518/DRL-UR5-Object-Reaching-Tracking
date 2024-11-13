import tensorflow as tf
from tensorflow.keras import optimizers, layers
import numpy as np
from sklearn.utils import shuffle


class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hdim, max_std, seed):
        super(PolicyNetwork, self).__init__()
        tf.random.set_seed(seed)
        self.hid1 = tf.keras.layers.Dense(hdim, activation='tanh',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                          name='policy_h1')
        self.hid2 = tf.keras.layers.Dense(hdim, activation='tanh',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                          name='policy_h2')
        self.mean_layer = tf.keras.layers.Dense(act_dim,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                                name='mean')
        # 重要部分：定义标准差的变量，并使其可训练
        self.logits_std = tf.Variable(initial_value=tf.random.normal(shape=(1,), stddev=0.01, seed=seed),
                                      trainable=True, name='logits_std')
        self.max_std = max_std

    def call(self, obs):
        # 前向传播策略网络
        x = self.hid1(obs)
        x = self.hid2(x)
        mean = self.mean_layer(x)
        # 重要部分：使用 sigmoid 函数确保标准差为正数并受限
        std = self.max_std * tf.ones_like(mean) * tf.sigmoid(self.logits_std)
        return mean, std

class ValueNetwork(tf.keras.Model):
    def __init__(self, obs_dim, hdim, seed):
        super(ValueNetwork, self).__init__()
        tf.random.set_seed(seed)
        self.hid1 = tf.keras.layers.Dense(hdim, activation='tanh',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                          name='value_h1')
        self.hid2 = tf.keras.layers.Dense(hdim, activation='tanh',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                          name='value_h2')
        self.output_layer = tf.keras.layers.Dense(1,
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                                  name='value_output')

    def call(self, obs):
        # 前向传播价值网络
        x = self.hid1(obs)
        x = self.hid2(x)
        value = self.output_layer(x)
        value = tf.squeeze(value, axis=-1)  # 去除多余的维度
        return value

class PPOGAEAgent(object): 
    def __init__(self, obs_dim, n_act, clip_range=0.2, epochs=10, policy_lr=3e-3, value_lr=7e-4, hdim=64, max_std=1.0, seed=0):
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.obs_dim = obs_dim
        self.act_dim = n_act

        self.clip_range = clip_range

        self.epochs = epochs
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.hdim = hdim
        self.max_std = max_std

        # 构建策略和价值网络
        self.policy_network = PolicyNetwork(self.obs_dim, self.act_dim, self.hdim, self.max_std, self.seed)
        self.value_network = ValueNetwork(self.obs_dim, self.hdim, self.seed)

        # 定义策略和价值网络的优化器
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.policy_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=self.value_lr)

    def log_prob(self, y, mu, sigma):
        # 计算给定均值和标准差的高斯分布的对数概率
        return tf.reduce_sum(-0.5 * tf.square((y - mu) / sigma) -
                             tf.math.log(sigma) - 0.5 * np.log(2.0 * np.pi), axis=1)

    @tf.function
    def train_step(self, obs_batch, act_batch, adv_batch, ret_batch, old_mean_batch, old_std_batch):
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播策略和价值网络
            mean, std = self.policy_network(obs_batch)
            value = self.value_network(obs_batch)

            # 计算新旧策略的对数概率
            logp = self.log_prob(act_batch, mean, std)
            logp_old = self.log_prob(act_batch, old_mean_batch, old_std_batch)

            # 计算 PPO 损失中的比率
            ratio = tf.exp(logp - logp_old)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            # 计算策略损失
            policy_loss = -tf.reduce_mean(tf.minimum(adv_batch * ratio, adv_batch * clipped_ratio))

            # 计算价值损失
            value_loss = tf.reduce_mean(0.5 * tf.square(value - ret_batch))

        # 计算梯度并更新策略网络
        policy_gradients = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_network.trainable_variables))

        # 计算梯度并更新价值网络
        value_gradients = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradients, self.value_network.trainable_variables))

        del tape  # 释放资源

        # 返回当前的策略损失和价值损失
        return policy_loss, value_loss

    def compute_kl(self, mean, std, old_mean, old_std):
        # 计算新旧策略之间的 KL 散度
        log_std_old = tf.math.log(old_std)
        log_std_new = tf.math.log(std)
        frac_std_old_new = old_std / std
        kl = tf.reduce_sum(log_std_new - log_std_old + 0.5 * tf.square(frac_std_old_new) +
                           0.5 * tf.square((mean - old_mean) / std) - 0.5, axis=1)
        return tf.reduce_mean(kl)

    def compute_entropy(self, std):
        # 计算高斯分布的熵
        log_std_new = tf.math.log(std)
        entropy = tf.reduce_sum(log_std_new + 0.5 + 0.5 * np.log(2 * np.pi), axis=1)
        return tf.reduce_mean(entropy)

    def get_value(self, obs):
        # 估计给定观测的价值函数
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        value = self.value_network(obs)
        return value.numpy()

    def get_action(self, obs):
        # 从策略中采样动作
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        mean, std = self.policy_network(obs)
        sampled_action = mean + tf.random.normal(tf.shape(mean), seed=self.seed) * std
        return sampled_action[0].numpy()

    def control(self, obs):
        # 计算确定性策略的均值动作
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        mean, std = self.policy_network(obs)
        # sampled_action = mean + tf.random.normal(tf.shape(mean), seed=self.seed) * std
        # return sampled_action[0].numpy()
        return mean[0].numpy()

    def update(self, observes, actions, advantages, returns, batch_size=128):
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        # 计算旧的均值和标准差
        observes_tf = tf.convert_to_tensor(observes, dtype=tf.float32)
        old_means_tf, old_std_tf = self.policy_network(observes_tf)
        old_means_np = old_means_tf.numpy()
        old_std_np = old_std_tf.numpy()

        # 数据打乱
        observes, actions, advantages, returns, old_means_np, old_std_np = shuffle(
            observes, actions, advantages, returns, old_means_np, old_std_np, random_state=self.seed)
        
        # observes = observes.astype(np.float32)
        # actions = actions.astype(np.float32)
        advantages = advantages.astype(np.float32)
        # returns = returns.astype(np.float32)
        # old_means_np = old_means_np.astype(np.float32)
        # old_std_np = old_std_np.astype(np.float32)

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((observes, actions, advantages, returns, old_means_np, old_std_np))
        dataset = dataset.batch(batch_size)

        for e in range(self.epochs):
            for batch in dataset:
                obs_batch, act_batch, adv_batch, ret_batch, old_mean_batch, old_std_batch = batch
                # 确保张量的类型正确
                obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
                act_batch = tf.convert_to_tensor(act_batch, dtype=tf.float32)
                adv_batch = tf.convert_to_tensor(adv_batch, dtype=tf.float32)
                ret_batch = tf.convert_to_tensor(ret_batch, dtype=tf.float32)
                old_mean_batch = tf.convert_to_tensor(old_mean_batch, dtype=tf.float32)
                old_std_batch = tf.convert_to_tensor(old_std_batch, dtype=tf.float32)

                self.train_step(obs_batch, act_batch, adv_batch, ret_batch, old_mean_batch, old_std_batch)

        # 计算损失用于日志记录
        policy_loss, value_loss, kl, entropy = self.compute_losses(observes_tf, actions, advantages, returns, old_means_tf, old_std_tf)
        return policy_loss.numpy(), value_loss.numpy(), kl.numpy(), entropy.numpy()

    def compute_losses(self, observes, actions, advantages, returns, old_means_tf, old_std_tf):
        mean, std = self.policy_network(observes)
        value = self.value_network(observes)

        # 计算对数概率
        logp = self.log_prob(actions, mean, std)
        logp_old = self.log_prob(actions, old_means_tf, old_std_tf)

        # 计算比率
        ratio = tf.exp(logp - logp_old)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
        # 计算策略损失
        policy_loss = -tf.reduce_mean(tf.minimum(advantages * ratio, advantages * clipped_ratio))

        # 计算价值损失
        value_loss = tf.reduce_mean(0.5 * tf.square(value - returns))

        # 计算 KL 散度和熵
        kl = self.compute_kl(mean, std, old_means_tf, old_std_tf)
        entropy = self.compute_entropy(std)

        return policy_loss, value_loss, kl, entropy
    
    # def save_model(self, policy_path='./models/policy_network.h5', value_path='./models/value_network.h5'):
    #     self.policy_network.save(policy_path)
    #     self.value_network.save(value_path)

    # def load_model(self, policy_path='./models/policy_network.h5', value_path='./models/value_network.h5'):
    #     self.policy_network = tf.keras.models.load_model(policy_path)
    #     self.value_network = tf.keras.models.load_model(value_path)

    def save_model(self, policy_weights_path, value_weights_path):
        self.policy_network.save_weights(policy_weights_path)
        self.value_network.save_weights(value_weights_path)

    def load_model(self, policy_weights_path, value_weights_path):
        dummy_input = tf.zeros(shape=(1, self.obs_dim), dtype=tf.float32)
        self.policy_network(dummy_input)
        self.value_network(dummy_input)
        self.policy_network.load_weights(policy_weights_path)
        self.value_network.load_weights(value_weights_path)


'''
class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hdim, max_std, seed):
        super(PolicyNetwork, self).__init__()
        tf.random.set_seed(seed)
        self.hid1 = layers.Dense(hdim, activation='tanh',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                        name='policy_h1')
        self.hid2 = layers.Dense(hdim, activation='tanh',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                        name='policy_h2')
        self.mean_layer = layers.Dense(act_dim,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                        name='mean')
        # 重要部分：定义标准差的变量，并使其可训练
        self.logits_std = tf.Variable(initial_value=tf.random.normal(shape=(1,), stddev=0.01, seed=seed),
                                      trainable=True, name='logits_std')
        self.max_std = max_std

    def call(self, obs):
        # 前向传播策略网络
        x = self.hid1(obs)
        x = self.hid2(x)
        mean = self.mean_layer(x)
        # 重要部分：使用 sigmoid 函数确保标准差为正数并受限
        std = self.max_std * tf.ones_like(mean) * tf.sigmoid(self.logits_std)
        return mean, std

class ValueNetwork(tf.keras.Model):
    def __init__(self, obs_dim, hdim, seed):
        super(ValueNetwork, self).__init__()
        tf.random.set_seed(seed)
        self.hid1 = layers.Dense(hdim, activation='tanh',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                         name='value_h1')
        self.hid2 = layers.Dense(hdim, activation='tanh',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                         name='value_h2')
        self.output_layer = layers.Dense(1,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
                                         name='value_output')

    def call(self, obs):
        # 前向传播价值网络
        x = self.hid1(obs)
        x = self.hid2(x)
        value = self.output_layer(x)
        value = tf.squeeze(value, axis=-1)  # 去除多余的维度
        return value

class PPOGAEAgent(object): 
    def __init__(self, obs_dim, n_act, clip_range=0.2, epochs=10, policy_lr=3e-3, value_lr=7e-4, hdim=64, max_std=1.0, seed=0):
        self.seed = seed
        # tf.random.set_seed(self.seed)
        # np.random.seed(self.seed)

        self.obs_dim = obs_dim
        self.act_dim = n_act

        self.clip_range = clip_range

        self.epochs = epochs
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.hdim = hdim
        self.max_std = max_std

        # 构建策略和价值网络
        self.policy_network = PolicyNetwork(self.obs_dim, self.act_dim, self.hdim, self.max_std, self.seed)
        self.value_network = ValueNetwork(self.obs_dim, self.hdim, self.seed)

        # 定义策略和价值网络的优化器
        self.policy_optimizer = optimizers.Adam(learning_rate=self.policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=self.value_lr)

    def log_prob(self, y, mu, sigma):
        # 计算给定均值和标准差的高斯分布的对数概率
        return tf.reduce_sum(-0.5 * tf.square((y - mu) / sigma) - tf.math.log(sigma) - 0.5 * np.log(2.0 * np.pi), axis=1)

    @tf.function
    def train_step(self, obs_batch, act_batch, adv_batch, ret_batch, old_mean_batch, old_std_batch):
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播策略和价值网络
            mean, std = self.policy_network(obs_batch)
            value = self.value_network(obs_batch)

            # 计算新旧策略的对数概率
            logp = self.log_prob(act_batch, mean, std)
            logp_old = self.log_prob(act_batch, old_mean_batch, old_std_batch)

            # 计算 PPO 损失中的比率
            ratio = tf.exp(logp - logp_old)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            # 计算策略损失
            policy_loss = -tf.reduce_mean(tf.minimum(adv_batch * ratio, adv_batch * clipped_ratio))

            # 计算价值损失
            value_loss = tf.reduce_mean(0.5 * tf.square(value - ret_batch))

        # 计算梯度并更新策略网络
        policy_gradients = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_network.trainable_variables))

        # 计算梯度并更新价值网络
        value_gradients = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradients, self.value_network.trainable_variables))

        del tape  # 释放资源

    def compute_kl(self, mean, std, old_mean, old_std):
        # 计算新旧策略之间的 KL 散度
        log_std_old = tf.math.log(old_std)
        log_std_new = tf.math.log(std)
        frac_std_old_new = old_std / std
        kl = tf.reduce_sum(log_std_new - log_std_old + 0.5 * tf.square(frac_std_old_new) +
                           0.5 * tf.square((mean - old_mean) / std) - 0.5, axis=1)
        return tf.reduce_mean(kl)

    def compute_entropy(self, std):
        # 计算高斯分布的熵
        log_std_new = tf.math.log(std)
        entropy = tf.reduce_sum(log_std_new + 0.5 + 0.5 * np.log(2 * np.pi), axis=1)
        return tf.reduce_mean(entropy)

    def compute_losses(self, observes, actions, advantages, returns, old_means_tf, old_std_tf):
        mean, std = self.policy_network(observes)
        value = self.value_network(observes)

        # 计算对数概率
        logp = self.log_prob(actions, mean, std)
        logp_old = self.log_prob(actions, old_means_tf, old_std_tf)

        # 计算比率
        ratio = tf.exp(logp - logp_old)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
        # 计算策略损失
        policy_loss = -tf.reduce_mean(tf.minimum(advantages * ratio, advantages * clipped_ratio))

        # 计算价值损失
        value_loss = tf.reduce_mean(0.5 * tf.square(value - returns))

        # 计算 KL 散度和熵
        kl = self.compute_kl(mean, std, old_means_tf, old_std_tf)
        entropy = self.compute_entropy(std)

        return policy_loss, value_loss, kl, entropy

    def get_value(self, obs):
        # 估计给定观测的价值函数
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        value = self.value_network(obs[None, :])  # 添加批次维度
        return value[0].numpy()

    def get_action(self, obs):
        # 从策略中采样动作
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        mean, std = self.policy_network(obs[None, :])  # 添加批次维度
        sampled_action = mean + tf.random.normal(tf.shape(mean), seed=self.seed) * std
        ret = sampled_action[0].numpy()[0]
        print(ret)
        return ret

    def control(self, obs):
        # 计算确定性策略的均值动作
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        mean, _ = self.policy_network(obs[None, :])  # 添加批次维度
        return mean[0].numpy()

    def update(self, observes, actions, advantages, returns, batch_size=128):
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        # 计算旧的均值和标准差
        observes_tf = tf.convert_to_tensor(observes, dtype=tf.float32)
        old_means_tf, old_std_tf = self.policy_network(observes_tf)
        old_means_np = old_means_tf.numpy()
        old_std_np = old_std_tf.numpy()

        # 数据打乱
        observes, actions, advantages, returns, old_means_np, old_std_np = shuffle(
            observes, actions, advantages, returns, old_means_np, old_std_np, random_state=self.seed)

        # **重要修改**：确保所有 numpy 数组都是 float32 类型
        observes = observes.astype(np.float32)
        actions = actions.astype(np.float32)
        advantages = advantages.astype(np.float32)
        returns = returns.astype(np.float32)
        old_means_np = old_means_np.astype(np.float32)
        old_std_np = old_std_np.astype(np.float32)

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((observes, actions, advantages, returns, old_means_np, old_std_np))
        dataset = dataset.batch(batch_size)

        for e in range(self.epochs):
            for batch in dataset:
                obs_batch, act_batch, adv_batch, ret_batch, old_mean_batch, old_std_batch = batch
                self.train_step(obs_batch, act_batch, adv_batch, ret_batch, old_mean_batch, old_std_batch)

        # 计算损失用于日志记录
        policy_loss, value_loss, kl, entropy = self.compute_losses(observes_tf, actions, advantages, returns, old_means_tf, old_std_tf)
        return policy_loss.numpy(), value_loss.numpy(), kl.numpy(), entropy.numpy()
'''