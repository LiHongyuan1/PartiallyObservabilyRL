# make_video.py
#author: <lihongyuaj>
import os
import argparse
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

import tensorflow as tf
import ncps.tf

# ———— 1. PartialObservation & CustomRNN ————
class PartialObservation(gymnasium.ObservationWrapper):
    def __init__(self, env, idx):
        super().__init__(env)
        self.idx = idx
        low, high = env.observation_space.low[idx], env.observation_space.high[idx]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def observation(self, obs):
        return obs[self.idx].astype(np.float32)

def make_env(render_mode="rgb_array"):
    base = gymnasium.make("HalfCheetah-v4", render_mode=render_mode)
    keep = [0,1,2,3,8,9,10,11,12]
    return PartialObservation(base, keep)

class CustomRNN(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_out, cfg, name):
        super().__init__(obs_space, action_space, num_out, cfg, name)
        cell = cfg.get("custom_model_config", {}).get("cell_size", 64)
        self.cell_size = cell

        in_obs = tf.keras.Input((None, obs_space.shape[0]), name="in_obs")
        in_h   = tf.keras.Input((cell,), name="h")
        in_seq = tf.keras.Input((), dtype=tf.int32, name="seq")

        td = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation="silu"),
                tf.keras.layers.Dense(256, activation="silu"),
            ])
        )(in_obs)

        rnn_out, h_out = ncps.tf.CfC(
            cell, return_sequences=True, return_state=True
        )(td, mask=tf.sequence_mask(in_seq), initial_state=[in_h])

        logits = tf.keras.layers.Dense(num_out)(rnn_out)
        values = tf.keras.layers.Dense(1)(rnn_out)
        self.net = tf.keras.Model([in_obs, in_seq, in_h], [logits, values, h_out])

    @override(RecurrentNetwork)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32)]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        logits, vals, h = self.net([inputs, seq_lens] + state)
        self._value_out = tf.reshape(vals, [-1])
        return logits, [h]

    @override(ModelV2)
    def value_function(self):
        return self._value_out

# ———— 2. 主函数，解析参数 ————
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "-c", type=str, required=True,
                   help="路径指向 RLlib checkpoint 目录（例如 ./ckpt）")
    p.add_argument("--outdir", "-o", type=str, default="./videos",
                   help="输出视频文件夹")
    args = p.parse_args()

    # 2.1 重新注册环境 & 模型（和训练时一致）
    register_env("POMDPHalfCheetah", lambda cfg: make_env(render_mode="rgb_array"))
    ModelCatalog.register_custom_model("cfc_rnn", CustomRNN)

    # 2.2 初始化 Ray 并恢复算法
    ray.init(ignore_reinit_error=True)
    algo = PPO.from_checkpoint(args.checkpoint)

    # 2.3 包装 RecordVideo
    os.makedirs(args.outdir, exist_ok=True)
    env = RecordVideo(
        make_env(render_mode="rgb_array"),
        video_folder=args.outdir,
        name_prefix="cheetah_cfc",
        episode_trigger=lambda ep: True
    )

    state = algo.get_policy().get_initial_state()
    obs, _ = env.reset()
    done = False
    while not done:
        action, state, _ = algo.compute_single_action(obs, state=state, explore=False)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc

    env.close()    # 确保视频 flush
    ray.shutdown()

    print("录制完成，视频保存在：", args.outdir)
    for f in os.listdir(args.outdir):
        print("  ", f)

