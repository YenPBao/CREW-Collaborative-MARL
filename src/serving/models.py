"""
Model Registry - Extensible configuration for all supported local models.

To add a new model:
1. Add an entry to MODEL_REGISTRY with the model configuration
2. Optionally add shortcuts to MODEL_SHORTCUTS

Example:
    MODEL_REGISTRY["my-new-model"] = {
        "name": "My New Model",
        "hf_name": "organization/model-name",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
        "description": "Description of the model"
    }
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str                           # Display name
    hf_name: str                        # HuggingFace model name
    max_model_len: int = 8192           # Max context length
    gpu_memory_utilization: float = 0.9 # GPU memory usage
    tensor_parallel_size: int = 1       # Number of GPUs for tensor parallelism
    description: str = ""               # Model description
    quantization: Optional[str] = None  # Quantization method (e.g., "awq", "gptq")
    trust_remote_code: bool = True      # Trust remote code for custom models


# ============================================================================
# MODEL REGISTRY - Add new models here
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Qwen Models
    "qwen3-30b": ModelConfig(
        name="Qwen3-30B-A3B-Instruct",
        hf_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=2,
        description="Qwen3 30B instruction-tuned model (A3B variant)"
    ),
    "qwen2.5-72b": ModelConfig(
        name="Qwen2.5-72B-Instruct",
        hf_name="Qwen/Qwen2.5-72B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=2,
        description="Qwen2.5 72B instruction-tuned model (requires 2+ GPUs)"
    ),
    "qwen2.5-32b": ModelConfig(
        name="Qwen2.5-32B-Instruct",
        hf_name="Qwen/Qwen2.5-32B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="Qwen2.5 32B instruction-tuned model"
    ),
    "qwen2.5-14b": ModelConfig(
        name="Qwen2.5-14B-Instruct",
        hf_name="Qwen/Qwen2.5-14B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="Qwen2.5 14B instruction-tuned model"
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B-Instruct",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="Qwen2.5 7B instruction-tuned model"
    ),
    
    # Llama Models
    "llama3.1-70b": ModelConfig(
        name="Llama-3.1-70B-Instruct",
        hf_name="meta-llama/Llama-3.1-70B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=2,
        description="Meta Llama 3.1 70B instruction-tuned (requires 2+ GPUs)"
    ),
    "llama3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="Meta Llama 3.1 8B instruction-tuned"
    ),
    
    # Mistral Models
    "mistral-7b": ModelConfig(
        name="Mistral-7B-Instruct",
        hf_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="Mistral 7B instruction-tuned v0.3"
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral-8x7B-Instruct",
        hf_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=2,
        description="Mixtral 8x7B MoE instruction-tuned (requires 2+ GPUs)"
    ),
    
    # DeepSeek Models
    "deepseek-v2": ModelConfig(
        name="DeepSeek-V2-Chat",
        hf_name="deepseek-ai/DeepSeek-V2-Chat",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=2,
        description="DeepSeek V2 Chat model"
    ),
    "deepseek-coder": ModelConfig(
        name="DeepSeek-Coder-33B-Instruct",
        hf_name="deepseek-ai/deepseek-coder-33b-instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="DeepSeek Coder 33B for code generation"
    ),
    
    # =========================================================================
    # LIGHTWEIGHT TEST MODELS - For quick testing (small, fast to load)
    # =========================================================================
    "qwen2.5-0.5b": ModelConfig(
        name="Qwen2.5-0.5B-Instruct",
        hf_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        description="Ultra-light Qwen (0.5B) - Best for testing (~1GB)"
    ),
    "qwen2.5-1.5b": ModelConfig(
        name="Qwen2.5-1.5B-Instruct",
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        description="Light Qwen (1.5B) - Good for testing (~3GB)"
    ),
    "qwen2.5-3b": ModelConfig(
        name="Qwen2.5-3B-Instruct",
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        description="Small Qwen (3B) - Balance of speed and quality (~6GB)"
    ),
    "tinyllama": ModelConfig(
        name="TinyLlama-1.1B-Chat",
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        description="TinyLlama (1.1B) - Super fast testing (~2GB)"
    ),
}


# ============================================================================
# MODEL SHORTCUTS - Aliases for common models
# ============================================================================

MODEL_SHORTCUTS: Dict[str, str] = {
    # Qwen shortcuts
    "qwen": "qwen3-30b",
    "qwen3": "qwen3-30b",
    
    # Llama shortcuts
    "llama": "llama3.1-8b",
    "llama3": "llama3.1-8b",
    "llama70b": "llama3.1-70b",
    
    # Mistral shortcuts
    "mistral": "mistral-7b",
    "mixtral": "mixtral-8x7b",
    
    # DeepSeek shortcuts
    "deepseek": "deepseek-v2",
    
    # Test model shortcuts (lightweight)
    "test": "qwen2.5-0.5b",       # Ultra-light for testing
    "tiny": "tinyllama",           # TinyLlama
    "small": "qwen2.5-3b",         # Small but capable
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration by name or shortcut.
    
    Args:
        model_name: Model name, shortcut, or HuggingFace model path
        
    Returns:
        ModelConfig for the specified model
        
    Raises:
        ValueError: If model is not found in registry
    """
    # Check shortcuts first
    if model_name.lower() in MODEL_SHORTCUTS:
        model_name = MODEL_SHORTCUTS[model_name.lower()]
    
    # Check registry
    if model_name.lower() in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name.lower()]
    
    # Check if it's a HuggingFace path (contains /)
    if "/" in model_name:
        # Create a dynamic config for custom HF models
        return ModelConfig(
            name=model_name.split("/")[-1],
            hf_name=model_name,
            description="Custom HuggingFace model"
        )
    
    # Not found
    available = list(MODEL_REGISTRY.keys()) + list(MODEL_SHORTCUTS.keys())
    raise ValueError(
        f"Model '{model_name}' not found. "
        f"Available models: {', '.join(sorted(set(available)))}"
    )


def list_models(verbose: bool = True) -> Dict[str, ModelConfig]:
    """
    List all available models.
    
    Args:
        verbose: If True, print model information
        
    Returns:
        Dictionary of model configurations
    """
    if verbose:
        print("\n" + "="*70)
        print(" AVAILABLE MODELS")
        print("="*70)
        
        # Group by family
        families = {}
        for key, config in MODEL_REGISTRY.items():
            family = key.split("-")[0].split(".")[0]
            if family not in families:
                families[family] = []
            families[family].append((key, config))
        
        for family, models in sorted(families.items()):
            print(f"\n- {family.upper()} Family:")
            for key, config in models:
                shortcuts = [k for k, v in MODEL_SHORTCUTS.items() if v == key]
                shortcut_str = f" (aliases: {', '.join(shortcuts)})" if shortcuts else ""
                gpu_info = f", {config.tensor_parallel_size} GPUs" if config.tensor_parallel_size > 1 else ""
                print(f"   * {key}{shortcut_str}")
                print(f"     - {config.hf_name}")
                print(f"       {config.description}{gpu_info}")
        
        print("\n" + "="*70)
        print("Usage: python -m src.serve --model <model_name>")
        print("   Example: python -m src.serve --model qwen")
        print("="*70 + "\n")
    
    return MODEL_REGISTRY


if __name__ == "__main__":
    list_models()
