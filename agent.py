import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CartPoleV0:
    def __init__(self, n_iterations, n_episodes_per_update, n_max_steps, discount_rate):
        self.n_iterations = n_iterations
        self.n_episodes_per_update = n_episodes_per_update
        self.n_max_steps = n_max_steps
        self.discount_rate = discount_rate
        
    def update_scene(self, num, frames, patch):
        patch.set_data(frames[num])
        return patch

    def plot_animation(self, frames, repeat = False, interval = 40):
        fig = plt.figure()
        patch = plt.imshow(frames[0])
        plt.axis("off")
        anim = animation.FuncAnimation(
            fig, self.update_scene, fargs = (frames, patch),
            frames = len(frames), repeat = repeat, interval = interval)
        plt.close()
        return anim

    def get_frames(self, model, n_max_steps = 200, seed = 42):
        frames = []
        env = gym.make("CartPole-v0")
        env.seed(seed)
        np.random.seed(seed)
        obs = env.reset()
        for step in range(self.n_max_steps):
            frames.append(env.render(mode = "rgb_array"))
            left_prob = model.predict(obs.reshape(1, -1))
            action = int(np.random.rand() > left_prob)
            obs, reward, done, info = env.step(action)
            if done:
                break
        env.close()
        return frames   

    def one_step(self, env, obs, model, loss_fn):
        with tf.GradientTape() as tape:
            left_prob = model(obs[np.newaxis])
            action = (tf.random.uniform([1, 1]) > left_prob)
            target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(loss_fn(target, left_prob))
        grads = tape.gradient(loss, model.trainable_variables)
        obs, reward, done, info = env.step(int(action[0, 0].numpy()))
        return obs, reward, done, grads

    def multiple_episodes(self, env, n_episodes, n_max_steps, model, loss_fn):
        all_rewards, all_grads = [], []
        for episode in range(n_episodes):
            cur_rewards, cur_grads = [], []
            obs = env.reset()
            for step in range(n_max_steps):
                obs, reward, done, grads = self.one_step(env, obs, model, loss_fn)
                cur_rewards.append(reward)
                cur_grads.append(grads)
                if done:
                    break
            all_rewards.append(cur_rewards)
            all_grads.append(cur_grads)
        return all_rewards, all_grads

    def discount_rewards(self, rewards, rate):
        dis_rewards = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            dis_rewards[step] += dis_rewards[step + 1] * rate
        return dis_rewards

    def discount_and_normalize_rewards(self, all_rewards, rate):
        all_dis_rewards = [self.discount_rewards(rewards, self.discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_dis_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(dis_rewards - reward_mean) / reward_std
                for dis_rewards in all_dis_rewards]

    def build_model(self):
        tf.keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        model = Sequential([
            Dense(5, activation = "elu", input_shape = [4]),
            Dense(1, activation = "sigmoid")
        ])
        return model

    def train(self):
        env = gym.make("CartPole-v0")
        env.seed(42)
        model = self.build_model()
        opt = Adam(lr = 0.01)
        loss_fn = binary_crossentropy
        for iteration in range(self.n_iterations):
            all_rewards, all_grads = self.multiple_episodes(
                env, self.n_episodes_per_update, self.n_max_steps, model, loss_fn)
            total_rewards = sum(map(sum, all_rewards))                   
            print("\rIteration: {} -> Mean Rewards: {:.1f}".format(         
                iteration + 1, total_rewards / self.n_episodes_per_update), end = "") 
            all_final_rewards = self.discount_and_normalize_rewards(all_rewards, self.discount_rate)
            all_mean_grads = []
            for var_index in range(len(model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_grads[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                        for step, final_reward in enumerate(final_rewards)], axis = 0)
                all_mean_grads.append(mean_grads)
            opt.apply_gradients(zip(all_mean_grads, model.trainable_variables))
        env.close()
        return model

    def train_and_plot(self):
        model = self.train()
        frames = self.get_frames(model)
        self.plot_animation(frames)

def main:
    agent = CartPoleV0(n_iterations = 150, n_episodes_per_update = 10, n_max_steps = 200, discount_rate = 0.95)
    agent.train_and_plot()
    
if __name__ == "__main__":
    main()
