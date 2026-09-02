from typing import Dict

TASKS = {
    "wsn": {
        "name": "Wireless Sensor Networks",
        "paper_title": "RCAMP: A Resilient Communication-Aware Motion Planner for Mobile Robots with Autonomous Repair of Wireless Connectivity",
        "abstract": "Mobile robots, be it autonomous or teleoperated, require stable communication with the base station to exchange valuable information. Given the stochastic elements in radio signal propagation, such as shadowing and fading, and the possibilities of unpredictable events or hardware failures, communication loss often presents a significant mission risk, both in terms of probability and impact, especially in Urban Search and Rescue (USAR) operations. Depending on the circumstances, disconnected robots are either abandoned, or attempt to autonomously back-trace their way to the base station. Although recent results in Communication-Aware Motion Planning can be used to effectively manage connectivity with robots, there are no results focusing on autonomously re-establishing the wireless connectivity of a mobile robot without"
    },
    "nlp": {
        "name": "Natural Language Processing",
        "paper_title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
    },
    "rl": {
        "name": "Reinforcement Learning",
        "paper_title": "Playing Atari with Deep Reinforcement Learning",
        "abstract": "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them."
    }
}

def get_task(name: str) -> Dict:
    """Get task details by name."""
    return TASKS.get(name.lower(), TASKS["wsn"])

def list_tasks():
    """Print available tasks."""
    print("Available tasks:")
    for key, val in TASKS.items():
        print(f" - {key}: {val['name']}")
