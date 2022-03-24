# 기본 패키지
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt 

# 강화학습 환경 패키지
import gym

# 인공지능 패키지: 텐서플로, 케라스 
# 호환성을 위해 텐스플로에 포함된 케라스를 불러옴 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def create_q_model(num_states, num_actions):
    inputs = Input(shape=(num_states,))
    layer = Dense(32, activation="relu")(inputs)
    layer = Dense(16, activation="relu")(layer)
    action = Dense(num_actions, activation="linear")(layer)
    return Model(inputs=inputs, outputs=action)

def list_rotate(l):
    return list(zip(*l))

class WorldFull():
    def __init__(self):
        self.get_env_model() #? 
        
        self.memory = deque(maxlen=2000)
        self.N_batch = 64
        self.t_model = create_q_model(self.num_states, self.num_actions)
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.optimizer = Adam(lr=self.learning_rate)
        
        self.epsilon = 0.2
        
    def get_env_model(self):
        self.env = gym.make('CartPole-v1')
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.model = create_q_model(self.num_states, self.num_actions)
    
    def update_t_model(self):
        self.t_model.set_weights(self.model.get_weights())

    def best_action(self, s):
        if random.random() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            s_array = np.array(s).reshape((1,-1))
            Qsa = self.model.predict(s_array)[0]
            return np.argmax(Qsa)

    def train_memory(self):
        if len(self.memory) >= self.N_batch:
            memory_batch = random.sample(self.memory, self.N_batch)
            s_l,a_l,r_l,next_s_l,done_l = [np.array(x) for x in list_rotate(memory_batch)]
            model_w = self.model.trainable_variables
            with tf.GradientTape() as tape:
                Qsa_pred_l = self.model(s_l.astype(np.float32))
                a_l_onehot = tf.one_hot(a_l, self.num_actions)
                Qs_a_pred_l = tf.reduce_sum(a_l_onehot * Qsa_pred_l, 
                                            axis=1)    

                Qsa_tpred_l = self.t_model(next_s_l.astype(np.float32)) 
                Qsa_tpred_l = tf.stop_gradient(Qsa_tpred_l)

                max_Q_next_s_a_l = np.amax(Qsa_tpred_l, axis=-1)
                Qs_a_l = r_l + (1 - done_l) * self.discount_factor * max_Q_next_s_a_l
                loss = tf.reduce_mean(tf.square(Qs_a_l - Qs_a_pred_l))
                grads = tape.gradient(loss, model_w)
                self.optimizer.apply_gradients(zip(grads, model_w))        
        
    def trials(self, n_episodes=100, flag_render=False):
        memory = self.memory
        env = self.env
        model = self.model
        score_l = []
        for e in range(n_episodes):
            done = False
            score = 0
            s = env.reset()
            while not done:                
                a = self.best_action(s)
                next_s, r, done, _ = env.step(a)
                if flag_render:
                    env.render()
                score += r
                memory.append([s,a,r,next_s,done])
                # self.train_memory()     
                s = next_s
                self.train_memory()                 
            self.update_t_model()
            print(f'Episode: {e:5d} -->  Score: {score:3.1f}') 
            score_l.append(score)            
        return score_l

new_world = WorldFull()
score_l = new_world.trials(n_episodes=100)
new_world.env.close()
np.save('score_l.npy', score_l)
print('Job completed!')

plt.plot(score_l)
plt.title("Deep Q-Learning for Cartpole")
plt.xlabel("Episode")
plt.ylabel("Score")