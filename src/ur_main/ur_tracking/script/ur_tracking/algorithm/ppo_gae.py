"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from sklearn.utils import shuffle


class PPOGAEAgent:
    def __init__(self, obs_space, act_space, clip_range=0.2, epochs=10, policy_lr=3e-3, value_lr=7e-4,
                 hdim=64, max_std=1.0, seed=0):
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        
        self.obs_space = obs_space
        self.act_space = act_space
        
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.shape[0]
        self.clip_range = clip_range
        self.epochs = epochs
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.hdim = hdim
        self.max_std = max_std
        
        self._build_models()

    def _build_models(self):
        # Build policy network
        self.policy_model = self._build_policy_network()
        # Build value network
        self.value_model = self._build_value_network()
        # Define optimizers
        self.policy_optimizer = optimizers.Adam(learning_rate=self.policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=self.value_lr)
        # Initialize the log_std as a trainable variable
        self.logstd = tf.Variable(tf.zeros((self.act_dim,)), dtype=tf.float32, trainable=True)

    def _build_policy_network(self):
        model = Sequential([
            layers.InputLayer(input_shape=(self.obs_dim,)),
            layers.Dense(self.hdim, activation='tanh', kernel_initializer='glorot_uniform'),
            layers.Dense(self.hdim, activation='tanh', kernel_initializer='glorot_uniform'),
            layers.Dense(self.act_dim, activation=None)
        ])
        return model

    def _build_value_network(self):
        model = Sequential([
            layers.InputLayer(input_shape=(self.obs_dim,)),
            layers.Dense(self.hdim, activation='tanh', kernel_initializer='glorot_uniform'),
            layers.Dense(self.hdim, activation='tanh', kernel_initializer='glorot_uniform'),
            layers.Dense(1, activation=None)
        ])
        return model

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)  # Ensure obs is 2D
        mean = self.policy_model(obs)
        std = tf.exp(self.logstd)
        sampled_action = mean + std * tf.random.normal(shape=mean.shape)
        # return sampled_action.numpy()[0]
        # 将动作裁剪到 action_space 的范围内
        clipped_action = tf.clip_by_value(sampled_action, self.act_space.low, self.act_space.high)
        return clipped_action.numpy()[0]

    def get_value(self, obs: np.ndarray) -> float:
        obs = np.atleast_2d(obs)  # Ensure obs is 2D
        return self.value_model(obs).numpy()[0]

    def update(self, observes, actions, advantages, returns, batch_size=128):
        # Ensure all inputs are float32 to match TensorFlow's default type
        observes = observes.astype(np.float32)
        actions = actions.astype(np.float32)
        advantages = advantages.astype(np.float32)
        returns = returns.astype(np.float32)

        old_means = self.policy_model(observes)
        old_stds = tf.exp(self.logstd)

        old_means_np = old_means.numpy()
        old_stds_np = np.tile(old_stds.numpy(), (len(observes), 1))

        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        # Initialize variables to store losses
        policy_loss = 0.0
        value_loss = 0.0
        kl = 0.0
        entropy = 0.0

        for epoch in range(self.epochs):
            observes, actions, advantages, returns, old_means_np, old_stds_np = shuffle(
                observes, actions, advantages, returns, old_means_np, old_stds_np, random_state=self.seed
            )

            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size

                pl, vl, k, ent = self._update_step(
                    observes[start:end],
                    actions[start:end],
                    advantages[start:end],
                    returns[start:end],
                    old_means_np[start:end],
                    old_stds_np[start:end]
                )

                # Accumulate losses for reporting
                policy_loss += pl
                value_loss += vl
                kl += k
                entropy += ent

        # Average the accumulated losses over the number of batches
        policy_loss /= (self.epochs * num_batches)
        value_loss /= (self.epochs * num_batches)
        kl /= (self.epochs * num_batches)
        entropy /= (self.epochs * num_batches)

        return policy_loss, value_loss, kl, entropy

    @tf.function
    def _update_step(self, observes, actions, advantages, returns, old_means, old_stds):
        advantages = tf.cast(advantages, tf.float32)
        returns = tf.cast(returns, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            new_means = self.policy_model(observes)
            new_stds = tf.exp(self.logstd)
            log_probs = self._compute_log_prob(actions, new_means, new_stds)
            old_log_probs = self._compute_log_prob(actions, old_means, old_stds)
            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            values = self.value_model(observes)
            value_loss = tf.reduce_mean((returns - tf.squeeze(values)) ** 2)

            # Calculate KL and entropy for monitoring
            kl = tf.reduce_mean(log_probs - old_log_probs)
            entropy = tf.reduce_mean(-log_probs)

        policy_grads = tape.gradient(policy_loss, self.policy_model.trainable_variables + [self.logstd])
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables + [self.logstd]))

        value_grads = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_model.trainable_variables))
        del tape
        
        return policy_loss, value_loss, kl, entropy

    def _compute_log_prob(self, actions, means, stds):
        var = stds ** 2
        log_prob = -0.5 * tf.reduce_sum(((actions - means) ** 2) / var + 2 * tf.math.log(stds) + np.log(2 * np.pi), axis=1)
        return log_prob

    def get_action_mean(self, obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        return self.policy_model(obs).numpy()

    def close(self):
        pass  # No session to close in TensorFlow 2.x
"""

"""
old version
"""

from collections import deque
import numpy as np
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.utils import shuffle


class PPOGAEAgent(object): 
    def __init__(self, obs_dim, n_act, clip_range=0.2, epochs=10, policy_lr=3e-3, value_lr=7e-4, hdim=64, max_std=1.0, seed=0):
        
        self.seed=seed
        
        self.obs_dim = obs_dim
        self.act_dim = n_act
        
        self.clip_range = clip_range
        
        self.epochs = epochs
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.hdim = hdim
        self.max_std = max_std
        
        self._build_graph()
        self._init_session()

        # load the parameters 
        #self.saver.restore(self.sess, './results/ppo_with_gae_model-20')

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._value_nn()
            self._logprob()
            self._loss_train_op()
            self._kl_entropy()
            self.init = tf.global_variables_initializer()
            self.variables = tf.global_variables()  
            # Create a saver object which will save all the variables
            #self.saver = tf.train.Saver()
           
    def _placeholders(self):
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.adv_ph = tf.placeholder(tf.float32, (None,), 'adv')
        self.ret_ph = tf.placeholder(tf.float32, (None,), 'ret')

        # learning rate:
        self.policy_lr_ph = tf.placeholder(tf.float32, (), 'policy_lr')
        self.value_lr_ph = tf.placeholder(tf.float32, (), 'value_lr')
        
        # place holder for old parameters
        self.old_std_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_std')
        self.old_mean_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
        
    def _policy_nn(self):        
        hid1_size = self.hdim
        hid2_size = self.hdim
        with tf.variable_scope("policy"):
            # TWO HIDDEN LAYERS
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                #   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h2")

            # MEAN FUNCTION
            self.mean = tf.layers.dense(out, self.act_dim,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), 
                                    name="mean")
            # UNI-VARIATE
            self.logits_std = tf.get_variable("logits_std",shape=(1,),initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed))
            self.std = self.max_std*tf.ones_like(self.mean)*tf.sigmoid(self.logits_std) # IMPORTANT TRICK

            # SAMPLE OPERATION
            self.sample_action = self.mean + tf.random_normal(tf.shape(self.mean),seed=self.seed)*self.std
    
    def _value_nn(self):
        hid1_size = self.hdim 
        hid2_size = self.hdim
        with tf.variable_scope("value"):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.01,seed=self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.01,seed=self.seed), name="h2")
            value = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.01,seed=self.seed), name='output')
            self.value = tf.squeeze(value)
            
    def _logprob(self):
        # PROBABILITY WITH TRAINING PARAMETER
        y = self.act_ph 
        mu = self.mean
        sigma = self.std
        
        self.logp = tf.reduce_sum(-0.5*tf.square((y-mu)/sigma)-tf.log(sigma)- 0.5*np.log(2.*np.pi),axis=1)

        # PROBABILITY WITH OLD (PREVIOUS) PARAMETER
        old_mu_ph = self.old_mean_ph
        old_sigma_ph = self.old_std_ph
                
        self.logp_old = tf.reduce_sum(-0.5*tf.square((y-old_mu_ph)/old_sigma_ph)-tf.log(old_sigma_ph)- 0.5*np.log(2.*np.pi),axis=1)
        
    def _kl_entropy(self):

        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean_ph, self.old_std_ph
 
        log_std_old = tf.log(old_std)
        log_std_new = tf.log(std)
        frac_std_old_new = old_std/std

        # KL DIVERGENCE BETWEEN TWO GAUSSIAN
        kl = tf.reduce_sum(log_std_new - log_std_old + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((mean - old_mean)/std)- 0.5,axis=1)
        self.kl = tf.reduce_mean(kl)
        
        # ENTROPY OF GAUSSIAN
        entropy = tf.reduce_sum(log_std_new + 0.5 + 0.5*np.log(2*np.pi),axis=1)
        self.entropy = tf.reduce_mean(entropy)
            
    def _loss_train_op(self):
        
        # REINFORCE OBJECTIVE
        ratio = tf.exp(self.logp - self.logp_old)
        cliped_ratio = tf.clip_by_value(ratio,clip_value_min=1-self.clip_range,clip_value_max=1+self.clip_range)
        self.policy_loss = -tf.reduce_mean(tf.minimum(self.adv_ph*ratio,self.adv_ph*cliped_ratio))
        
        # POLICY OPTIMIZER
        self.pol_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy")
        optimizer = tf.train.AdamOptimizer(self.policy_lr_ph)
        self.train_policy = optimizer.minimize(self.policy_loss, var_list=self.pol_var_list)
            
        # L2 LOSS
        self.value_loss = tf.reduce_mean(0.5*tf.square(self.value - self.ret_ph))
            
        # VALUE OPTIMIZER 
        self.val_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="value")
        optimizer = tf.train.AdamOptimizer(self.value_lr_ph)
        self.train_value = optimizer.minimize(self.value_loss, var_list=self.val_var_list)
        
    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)
        
    
    def get_value(self, obs):
        feed_dict = {self.obs_ph: obs}
        value = self.sess.run(self.value, feed_dict=feed_dict)
        return value
    
    def get_action(self, obs): # SAMPLE FROM POLICY
        feed_dict = {self.obs_ph: obs}
        sampled_action = self.sess.run(self.sample_action,feed_dict=feed_dict)
        return sampled_action[0]
    
    def control(self, obs): # COMPUTE MEAN
        feed_dict = {self.obs_ph: obs}
        best_action = self.sess.run(self.mean,feed_dict=feed_dict)
        return best_action    
    
    def update(self, observes, actions, advantages, returns, batch_size=128): # TRAIN POLICY
        
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches
        
        old_means_np, old_std_np = self.sess.run([self.mean, self.std],{self.obs_ph: observes}) # COMPUTE OLD PARAMTER
        for e in range(self.epochs):
            observes, actions, advantages, returns, old_means_np, old_std_np = shuffle(observes, actions, advantages, returns, old_means_np, old_std_np, random_state=self.seed)
            for j in range(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: observes[start:end,:],
                     self.act_ph: actions[start:end],
                     self.adv_ph: advantages[start:end],
                     self.ret_ph: returns[start:end],
                     self.old_std_ph: old_std_np[start:end,:],
                     self.old_mean_ph: old_means_np[start:end,:],
                     self.policy_lr_ph: self.policy_lr,
                     self.value_lr_ph: self.value_lr}        
                self.sess.run([self.train_policy,self.train_value], feed_dict)
            
        feed_dict = {self.obs_ph: observes,
             self.act_ph: actions,
             self.adv_ph: advantages,
             self.ret_ph: returns,
             self.old_std_ph: old_std_np,
             self.old_mean_ph: old_means_np,
             self.policy_lr_ph: self.policy_lr,
             self.value_lr_ph: self.value_lr}               
        policy_loss, value_loss, kl, entropy  = self.sess.run([self.policy_loss, self.value_loss, self.kl, self.entropy], feed_dict)
        
        # save the parameters
        #self.saver.save(self.sess, './results/ppo_with_gae_model', global_step=20)
        return policy_loss, value_loss, kl, entropy
    
    def close_sess(self):
        self.sess.close()