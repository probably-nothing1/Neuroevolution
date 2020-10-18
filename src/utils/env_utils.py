import collections
import os

import cv2
import gin
import gym
import numpy as np
from gym import wrappers


def get_video_filepath(env):
    assert isinstance(env, wrappers.Monitor)

    def pad_episode_number(id):
        return str(1000000 + id)[1:]

    episode_number = pad_episode_number(env.episode_id - 2)
    filepath = f"{env.file_prefix}.video.{env.file_infix}.video{episode_number}.mp4"
    return os.path.join(env.directory, filepath)


class FrameBuffer(gym.ObservationWrapper):
    def __init__(self, env, k_frames):
        super().__init__(env)
        old_space = self.observation_space
        new_low = old_space.low.repeat(k_frames, axis=0)
        new_high = old_space.high.repeat(k_frames, axis=0)
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class BlackAndWhiteImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 80, 80), dtype=np.float32)
        self.previous_image = np.zeros((84, 84))

    def observation(self, img):
        img = img[34:194]  # crop
        img = img[::2, ::2, 0]  # downsample by factor of 2
        img[img == 144] = 0  # erase background (background type 1)
        img[img == 109] = 0  # erase background (background type 2)
        img[img != 0] = 1  # everything else (paddles, ball) just set to 1
        img = img.astype(np.float32)
        return img[None]  # reshape to (1, 80, 80)


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, img):
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.597 + img[:, :, 2] * 0.114
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        img = img[18:102, :]
        return np.reshape(img, [84, 84, 1])


class ScaleImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0


class ImageToPytorchChannelOrdering(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], *old_shape[:-1])
        self.observation_space = gym.spaces.Box(0.0, 1.0, new_shape)

    def observation(self, observation):
        return np.einsum("hwc->chw", observation)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=4):
        super().__init__(env)
        self.skip_frames = skip_frames
        self.frame_buffer = collections.deque(maxlen=2)

    def reset(self):
        self.frame_buffer.clear()
        observation = self.env.reset()
        self.frame_buffer.append(observation)
        return observation

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip_frames):
            o, r, done, info = self.env.step(action)
            total_reward += r
            self.frame_buffer.append(o)
            if done:
                break

        max_frame = np.max(np.stack(self.frame_buffer), axis=0)
        return max_frame, total_reward, done, info


class EvalMonitor(wrappers.Monitor):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.mode = "training"

    def _set_mode(self, mode):
        assert mode in ["evaluation", "training"], "Invalid mode"
        self.mode = mode
        self.stats_recorder.type = "t" if mode == "training" else "e"

    def _video_enabled(self):
        return self.video_callable(self.mode)


class PongRewardModifier(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = reward != 0
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class CartPoleRewardModifier(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0

    def modify_reward(self, reward, done):
        if done:
            if self.steps < 30:
                reward -= 10
            else:
                reward = -1
        if self.steps > 100:
            reward += 1
        if self.steps > 200:
            reward += 1
        if self.steps > 300:
            reward += 1

        return reward

    def step(self, action):
        self.steps += 1
        observation, reward, done, info = self.env.step(action)
        reward = self.modify_reward(reward, done)
        return observation, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()


@gin.configurable
def create_environment(name, gym_make_kwargs=dict(), save_videos=False, wrapper_kwargs=dict()):
    env = gym.make(name, **gym_make_kwargs)
    if name.startswith("CartPole"):
        env = wrap_cart_pole_environment(env)
    elif name.startswith("Pong"):
        env = wrap_pong_environment(env)
    else:
        raise NotImplementedError(f"{name} environment is not currently supported.")

    if save_videos:
        env = EvalMonitor(env, video_callable=lambda mode: mode == "evaluation", **wrapper_kwargs)
    return env


def wrap_cart_pole_environment(env):
    # env = wrappers.TimeLimit(env.unwrapped, max_episode_steps=1000)
    # env = CartPoleRewardModifier(env)
    return env


def wrap_pong_environment(env):
    env = MaxAndSkipEnv(env, skip_frames=2)
    # env = ProcessFrame(env)
    # env = ScaleImage(env)
    # env = ImageToPytorchChannelOrdering(env)
    env = BlackAndWhiteImage(env)
    env = FrameBuffer(env, k_frames=4)
    env = PongRewardModifier(env)
    return env
