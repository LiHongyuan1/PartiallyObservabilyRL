# ----------------- train.py -----------------
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # 关键修复

import time, pathlib, numpy as np, matplotlib.pyplot as plt
import gymnasium
from gymnasium import spaces
import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
import tensorflow as tf
import ncps.tf
from datetime import datetime


print("TF version:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
ray.init(ignore_reinit_error=True)
print("Ray resources:", ray.cluster_resources())

# -------- 1. 修复的环境封装 --------
class PartialObservation(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_indices):
        super().__init__(env)
        self.obs_indices = obs_indices
        obsspace = env.observation_space
        self.observation_space = spaces.Box(
            low=np.array([obsspace.low[i] for i in obs_indices]),
            high=np.array([obsspace.high[i] for i in obs_indices]),
            dtype=np.float32
        )
    
    def observation(self, obs):
        return np.array([obs[i] for i in self.obs_indices], dtype=np.float32)

def make_env(render_mode=None):
    base = gymnasium.make("HalfCheetah-v4", render_mode=render_mode)
    keep = [0, 1, 2, 3, 8, 9, 10, 11, 12]
    return PartialObservation(base, keep)

# -------- 2. 修复的CfC-RNN模型 --------
class CustomRNN(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_out, cfg, name, cell_size=64):
        super().__init__(obs_space, action_space, num_out, cfg, name)
        self.cell_size = cell_size
        
        # 输入层
        in_obs = tf.keras.Input(shape=(None, obs_space.shape[0]), name="in_obs")
        in_h = tf.keras.Input(shape=(cell_size,), name="h")
        in_seq = tf.keras.Input(shape=(), dtype=tf.int32, name="seq")
        
        # 修复的前处理层（与作者一致）
        self.preprocess_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="silu"),
            tf.keras.layers.Dense(256, activation="silu")
        ])
        self.td_preprocess = tf.keras.layers.TimeDistributed(self.preprocess_layers)
        dense1 = self.td_preprocess(in_obs)
        
        # 修复的CfC调用
        rnn_layer = ncps.tf.CfC(cell_size, 
                                return_sequences=True,
                                return_state=True)
        rnn_out, *state_out = rnn_layer(dense1, mask=tf.sequence_mask(in_seq), initial_state=[in_h])
        state_h = state_out[0]
        
        logits = tf.keras.layers.Dense(num_out)(rnn_out)
        values = tf.keras.layers.Dense(1)(rnn_out)
        self.model = tf.keras.Model([in_obs, in_seq, in_h], [logits, values, state_h])

    @override(RecurrentNetwork)
    def get_initial_state(self): 
        return [np.zeros(self.cell_size, np.float32)]
    
    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        logits, vals, h = self.model([inputs, seq_lens] + state)
        self._value_out = tf.reshape(vals, [-1])
        return logits, [h]
    
    @override(ModelV2)
    def value_function(self): 
        return self._value_out

# -------- 3. 修复的评估函数 --------
def evaluate(algo, cell, noise=0.0, episodes=5, apply_filter=True):
    env = make_env()
    
    # 获取观测过滤器
    if apply_filter and hasattr(algo.workers, 'local_worker'):
        filter = algo.workers.local_worker().filters.get("default_policy", None)
    else:
        filter = None
    
    state = [np.zeros(cell, np.float32)] if cell else None
    scores = []
    obs, _ = env.reset()
    ep_r = 0
    
    while len(scores) < episodes:
        # 应用噪声（与作者一致）
        if noise > 0:
            obs_noisy = obs + np.random.default_rng().normal(0, noise, obs.shape)
        else:
            obs_noisy = obs
        
        # 应用过滤器（与作者一致）
        if filter:
            obs_filtered = filter(obs_noisy, update=False)
        else:
            obs_filtered = obs_noisy
        
        if cell:
            action, state, _ = algo.compute_single_action(obs_filtered, state=state, explore=False)
        else:
            action = algo.compute_single_action(obs_filtered, explore=False)
        
        obs, r, term, trun, _ = env.step(action)
        ep_r += r
        
        if term or trun:
            scores.append(ep_r)
            obs, _ = env.reset()
            ep_r = 0
            if cell: 
                state = [np.zeros(cell, np.float32)]
    
    return np.mean(scores)

# -------- 4. 训练函数 --------
def run(model_name, iters=500):
    cfg = {
        "env": "POMDPHalfCheetah",
        "framework": "tf",
        "num_gpus": 1, 
        "num_workers": 8, 
        "num_envs_per_worker": 4,
        "train_batch_size": 65536, 
        "sgd_minibatch_size": 4096,
        "num_sgd_iter": 64,
        "lr": 5e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.1,
        "observation_filter": "MeanStdFilter",
        "vf_loss_coeff": 0.5,  # 添加作者使用的参数
        "grad_clip": 0.5        # 添加作者使用的参数
    }
    
    cell = None
    if model_name == "cfc":
        cell = 64
        cfg["model"] = {
            "vf_share_layers": True,
            "custom_model": "cfc_rnn",
            "custom_model_config": {"cell_size": cell}
        }
    
    ModelCatalog.register_custom_model("cfc_rnn", CustomRNN)
    register_env("POMDPHalfCheetah", lambda _: make_env())
    
    algo = PPO(config=cfg)
    hist = {"it": [], "r": [], "rn": []}
    
    for i in range(1, iters + 1):
        algo.train()
        if i % 10 == 0 or i == 1:
            r = evaluate(algo, cell, noise=0.0, episodes=10)  # 使用10个episode
            rn = evaluate(algo, cell, noise=0.1, episodes=10)  # 使用10个episode
            print(f"{model_name} iter {i:04d}: {r:.1f} / noise {rn:.1f}")
            hist["it"].append(i)
            hist["r"].append(r)
            hist["rn"].append(rn)
    
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
	algo.save(f"ckpts/cfc_{ts}")
    print("saved to", path)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(hist["it"], hist["r"], label=f"{model_name}", color="tab:blue")
    plt.plot(hist["it"], hist["rn"], '--', label=f"{model_name}-noise", color="tab:blue")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("return")
    plt.savefig(f"{model_name}.png")
    return path

# ---------------- MAIN ----------------
if __name__ == "__main__":
    ray.init()
    run("cfc", 30)
    ray.shutdown()
