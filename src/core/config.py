import os
import torch
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CONFIG = {
    "model_path": os.getenv(
        "RWG_MODEL_PATH",
        "Qwen/Qwen2-1.5B"
    ),
    "dtype": torch.float16,
    "device_map": "auto",
    "temperature": 0.7,
    "max_new_tokens": 1400,
    "top_p": 0.9,
    "do_sample": True
}

MODEL_ALIASES = {
    "1": "gpt-4.1-mini",
    "2": "gemini:gemini-2.0-flash",
    "3": "claude:claude-3-5-haiku-latest",
    "qwen": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    "gpt": "openai:gpt-4o-mini",
    # Qwen 72B — Novita format (OPENAI_BASE_URL=https://api.novita.ai/v3/openai)
    "qwen72b": "qwen/qwen2.5-72b-instruct",
    # Qwen 72B — OpenRouter format (OPENAI_BASE_URL=https://openrouter.ai/api/v1)
    "qwen72b-or": "qwen/qwen-2.5-72b-instruct",
}

GEMINI_CONFIG = {
    "model_name": "gemini-2.5-flash",

    "api_key_env": "GOOGLE_API_KEY",

    "temperature": 0.3,
    "max_output_tokens": 1400,
    "top_p": 0.9,
    "top_k": 40,
}

OPENAI_CONFIG = {
    "model_name": "openrouter:qwen/qwen-2.5-72b-instruct",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.3,
    "max_output_tokens": 1400,
    "top_p": 0.9,
    "top_k": 40,
}

AGENT_CONFIG = {
    "default_token_budget": 8000,
    "context_capacity": 32000,
    "cost_per_1k_output_tokens": 0.01,
    "cost_per_1k_input_tokens": 0.01,
    "alpha": 1.0,
    "beta": 1.0,
    "epsilon": 0.05,
    "confidence_threshold": 0.6,
    "gen_rate_hint": 180,
}

NUM_AGENTS = 5
MAX_ROUNDS = 3

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Number of possible actions for agents
NUM_ACTIONS = 5  # PLAN, SEARCH, WRITE, REVIEW, WAIT

IPPO_HYPERPARAMS = {
    "lr": 1e-4,                 # Learning rate
    "gamma": 0.99,              # Discount factor
    "lambda": 1.0,              # GAE parameter
    "clip_param": 0.2,          # PPO clip parameter
    "vf_clip_param": 10.0,      # Value function clip
    "entropy_coeff": 0.01,      # Hệ số entropy (khuyến khích khám phá)
    "num_sgd_iter": 10,         # Số lần lặp lại SGD
    "sgd_minibatch_size": 128,  # Kích thước minibatch
    "train_batch_size": 2000,   # Kích thước batch thu thập từ các worker
}

task = """
Write the "Related Work" section for the given paper.
The section should:
- Synthesize existing studies relevant to the paper’s topic.
- Group prior work into clear research themes rather than listing papers individually.
- Summarize the main ideas and approaches of each group.
- Compare and contrast different research directions.
- Identify limitations or gaps in previous work that motivate the current study.
- Maintain a formal, objective academic tone.
"""

example = """
Actor-critic algorithms have been shown to benefit from learning centralized joint critics alongside
decentralised policies [10676556]. As the critics are not needed during execution, this approach is
inherently decentralizable. COMA [1704341] extends this approach with a counterfactual multi-agent critic
baseline based on temporal difference errors [4419546] in order to facilitate multi-agent credit assignment.
Joint Q-learning can also be made decentralizable. Value Decomposition Networks [4691862]
decompose joint state-action value functions into sums of decentralised utility functions that can be
used during greedy execution. QMIX [7141190] extends this additive decomposition to arbitrary centralized
monotonic mixing networks. Both centralized joint critics and factored joint value functions can reap
some benefits of centralized joint learning, while bypassing the joint action space explosion, imposing
an effective prior on multi-agent credit assignment, and mitigating practical learning pathologies
associated with centralized joint learning on popular benchmark environment StarCraft II [2028251].
Value factorisation has also been successfully transferred to continuous action spaces and combined
with actor critic approaches [5714663].

Independent learning (IL) dates back to the early days of multi-agent reinforcement learning [1704341]. 
While initially tabular, IL algorithms using neural networks as function approximators were
subsequently developed [10676556, 4419546]. The question of whether independent agents are
better at learning cooperative behaviour than multiple complete observing agents [4691862] was first
comprehensively investigated by Tan [7141190], who concludes that while sharing policies or experience
was generally advantageous, sharing extra sensory information could also negatively interfere with
learning, e.g., by expanding the state space.

Trust region optimisation for reinforcement learning was popularised by TRPO [2028251] which implements
iterative guaranteed monotonic improvements. PPO [5714663] preserves many of TRPO’s empirical benefits
while trading in theoretical guarantees for computational speed. PPO’s policy update regularisation
using clipped probability ratios has been subject to much scrutiny in single-agent settings. Truly
PPO [1704341] suggests modifications in order to ensure guaranteed monotonic improvements with little
computational overhead. Recent work confirms that PPO’s performance may crucially depend on the
choice of policy surrogate objective [10676556]. In addition, code-level design choices have been shown to
significantly impact PPO performance in practice [4419546].

While popular for single agent tasks, PPO has only recently been applied to decentralised cooperative
multi-agent tasks. Concurrent work proposes MAPPO [4691862], an actor-critic multi-agent algorithm based
on PPO. Like IPPO, MAPPO employs weight sharing between each agent’s critic. In contrast to
IPPO, MAPPO uses a centralized value function that conditions on the full state. As such, MAPPO is not an independent learning
algorithm. We show that IPPO substantially outperforms MAPPO on a variety of hard SMAC maps
and even beats other state-of-the-art algorithms, such as QMIX, on some of these, with only modest
hyperparameter tuning.
"""



