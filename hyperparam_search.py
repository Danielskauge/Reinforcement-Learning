import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from param_noise import *
from action_noise import *
from math import sqrt
import hyperopt
from hyperopt import fmin, tpe, hp


#Pendulum-v1, MountainCarContinuous-v0, LunarLanderContinuous-v2
problem = "Pendulum-v1"
batch_norm = False
use_param_noise = False
use_action_noise = True
epsilon_decay = True
gradient_clipping = False

#TD3 features
target_policy_smoothing = False
delayed_policy_updates = False
#clipped_double_q_learning = False -> might be included later

#interval for policy updates if delayed_policy_updates is enabled
td3_update_interval = 2
episodes = 100
epsilon = 1
gradient_clip_value = 1

if problem == 'Pendulum-v1':
    tau = 0.005
    gamma = 0.99
    actor_lr = 0.002
    critic_lr = 0.001
    buffer_size = 50000
    batch_size = 64
    noise_stddev = 0.2
    target_noise_stddev = 0.1
    hidden_layers_shape = (400,300)
    epsilon_decay = 0.95
    min_epsilon = 0.01 
elif problem == 'MountainCarContinuous-v0':
    tau = 0.001
    gamma = 0.99
    actor_lr = 0.00002
    critic_lr = 0.0002
    buffer_size = 50000
    batch_size = 256
    noise_stddev = 0.2
    target_noise_stddev = 0.1
    hidden_layers_shape = (400,300)
    epsilon_decay = 0.99
    min_epsilon = 0.01
else:
    tau = 0.001
    gamma = 0.99
    actor_lr = 0.0001
    critic_lr = 0.001
    buffer_size = 10000
    batch_size = 40
    noise_stddev = 0.2
    target_noise_stddev = 0.1
    hidden_layers_shape = (400,300)
    epsilon_decay = 0.99
    min_epsilon = 0.01


def hyperparam_search():
    print("Starting hyperparam-optimization")

    params = {'tau': hp.uniform('tau', tau/5, tau*5),
              'critic_lr': hp.uniform('critic_lr', critic_lr/10, critic_lr*10),
              'actor_lr': hp.uniform('actor_lr', actor_lr/10, actor_lr*10),
              'gamma': hp.uniform('gamma', 0.9, 0.999),
              'noise_stddev': hp.uniform('noise_stddev', noise_stddev/2, noise_stddev*2), }

    best = fmin(train, params,
                algo=hyperopt.rand.suggest, max_evals=100)

    print(best)
    print(hyperopt.space_eval(params, best))


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.index = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        self.index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[self.index] = obs_tuple[0]
        self.action_buffer[self.index] = obs_tuple[1]
        self.reward_buffer[self.index] = obs_tuple[2]
        self.next_state_buffer[self.index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, gamma
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)

            #adds noise to targets if target_policy_smoothing is enabled
            if target_policy_smoothing:
                #target_actions = target_actions.numpy()
                noise = tf.cast(tf.convert_to_tensor(
                    target_noise()), tf.float32)
                target_actions = tf.clip_by_value(tf.math.add(
                    noise, target_actions), lower_bound, upper_bound)

            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic(
                [state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

        #if clipped_double_q_learning enabled, only
        if (not delayed_policy_updates) or (delayed_policy_updates and (episode_step_counter % td3_update_interval == 0)):
            #print("training actor network")
            with tf.GradientTape() as tape:
                actions = actor(state_batch, training=True)
                critic_value = critic(
                    [state_batch, actions], training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(
                actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, actor.trainable_variables)
            )

    # We compute the loss and update parameters

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        #print("episode step counter", episode_step_counter)

        self.update(state_batch, action_batch,
                    reward_batch, next_state_batch, gamma)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(hidden_layers_shape):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))

    out = layers.Dense(hidden_layers_shape[0])(inputs)
    if batch_norm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    out = layers.Dense(hidden_layers_shape[1])(out)
    if batch_norm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    outputs = layers.Dense(num_actions, kernel_initializer=last_init)(out)
    if batch_norm:
        outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Activation('tanh')(outputs)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(hidden_layers_shape):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16)(state_input)
    if batch_norm:
        state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Activation('relu')(state_out)
    state_out = layers.Dense(32)(state_out)
    if batch_norm:
        state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Activation('relu')(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32)(action_input)
    if batch_norm:
        action_out = layers.BatchNormalization()(action_out)
    action_out = layers.Activation('relu')(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(hidden_layers_shape[0])(concat)
    if batch_norm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    out = layers.Dense(hidden_layers_shape[1])(out)
    if batch_norm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, use_param_noise):

    #gets action from actor network, and applies either param or action noise
    if use_param_noise:
        action = tf.squeeze(perturbed_actor(state))
    else:
        action = tf.squeeze(actor(state)) + action_noise()

    # We make sure action is within bounds
    legal_action = np.clip(action.numpy(), lower_bound, upper_bound)

    #neccesary for lunar lander
    if problem == 'LunarLanderContinuous-v0':
        return np.squeeze(legal_action)
    return [np.squeeze(legal_action)]

def ddpg_distance_metric(actions1, actions2):
    #Compute "distance" between actions taken by two policies at the same states
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist

def perturb_actor_parameters():
    #Apply parameter noise to actor model, for exploration
    global perturbed_actor
    perturbed_actor.set_weights(actor.get_weights())

    layers_params_array = perturbed_actor.trainable_variables

    random = 0
    for params_array in layers_params_array:
        random = tf.random.normal(params_array.shape)
        params_array.assign(params_array + random * param_noise.current_stddev)


env = gym.make(problem)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

actor = get_actor(hidden_layers_shape)
critic = get_critic(hidden_layers_shape)

target_actor = get_actor(hidden_layers_shape)
target_critic = get_critic(hidden_layers_shape)

def train(params):
    global tau
    global actor_lr
    global critic_lr
    global gamma
    global noise_stddev
    global epsilon
    global actor
    global critic
    global target_actor
    global target_critic

    if params != None:
        tau = params['tau']
        actor_lr = params['actor_lr']
        critic_lr = params['critic_lr']
        gamma = params['gamma']
        noise_stddev = params['noise_stddev']

    buffer = Buffer(buffer_size, batch_size)

    actor = get_actor(hidden_layers_shape)
    critic = get_critic(hidden_layers_shape)

    target_actor = get_actor(hidden_layers_shape)
    target_critic = get_critic(hidden_layers_shape)

    # Making the weights equal initially
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    #actor to be perturbed with parameter noise
    if use_param_noise:
        perturbed_actor = get_actor(hidden_layers_shape)
        perturbed_actor.set_weights(actor.get_weights())

        #parameter noise object
        param_noise = AdaptiveParamNoiseSpec(
            initial_stddev=0.1, desired_action_stddev=noise_stddev, adaptation_coefficient=1.05)

    if use_action_noise:
        #action noise object
        action_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_stddev) * np.ones(1))

    if target_policy_smoothing:
        target_noise = OUActionNoise(mean=np.zeros(
            1), std_deviation=float(target_noise_stddev) * np.ones(1))

    if gradient_clipping:
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipvalue = gradient_clip_value)
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipvalue = gradient_clip_value)
    else:
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)


    episode_step_counter = 0

    #To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    for ep in range(episodes):

        
        prev_state = env.reset()
        episodic_reward = 0
        episode_step_counter = 0

        if use_param_noise: perturb_actor_parameters()

        if epsilon_decay:
            epsilon *= epsilon_decay
            epsilon = np.maximum(min_epsilon,epsilon)
            if use_param_noise:
                param_noise.set_desired_action_stddev(noise_stddev*epsilon)

        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, use_param_noise)

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            episode_step_counter += 1

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn()

            if (not delayed_policy_updates) or (delayed_policy_updates and (episode_step_counter % td3_update_interval == 0)):
                #print("updating target networks")
                update_target(target_actor.variables, actor.variables, tau)
                update_target(target_critic.variables, critic.variables, tau)

            if done:
                break

            prev_state = state
            # End this episode when `done` is True

        if buffer.index - episode_step_counter > 0:
            episode_perturbed_actions = buffer.action_buffer[buffer.index-episode_step_counter:buffer.index]
            episode_states = buffer.state_buffer[buffer.index - episode_step_counter:buffer.index]
        else:
            episode_perturbed_actions = buffer.action_buffer[buffer.index-episode_step_counter + buffer.buffer_capacity : buffer.buffer_capacity] 
            + buffer.action_buffer[0:buffer.index]
            episode_states = buffer.state_buffer[buffer.index-episode_step_counter + buffer.buffer_capacity : buffer.buffer_capacity] 
            + buffer.state_buffer[0:buffer.index]

        if use_param_noise:
            episode_unperturbed_actions = policy(episode_states, use_param_noise)

            #find distance between perturbed and unperturbed actions
            distance = ddpg_distance_metric(episode_perturbed_actions, episode_unperturbed_actions)

            #adjust param noise std_dev
            param_noise.adapt(distance)
        
        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-10:])
        print("Episode * {} * Episodic Reward is ==> {}".format(ep, episodic_reward))
        avg_reward_list.append(avg_reward)

    # Plotting and saving graph
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.show()

    return avg_reward_list[-1]

train(None)
