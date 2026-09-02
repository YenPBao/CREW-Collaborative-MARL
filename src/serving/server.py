"""
vLLM Server Launcher - Start OpenAI-compatible API server for local models.

This module provides functionality to start a vLLM server that serves
models with an OpenAI-compatible API interface.
"""

import subprocess
import sys
import os
from typing import Optional

from src.serving.models import get_model_config, ModelConfig


def start_server(
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    quantization: Optional[str] = None,
    dry_run: bool = False
) -> None:
    """
    Start vLLM server for the specified model.
    
    Args:
        model_name: Name of the model (from registry) or HuggingFace path
        host: Host to bind the server
        port: Port to serve on
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        max_model_len: Maximum model context length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        quantization: Quantization method (awq, gptq, etc.)
        dry_run: If True, only print the command without executing
    """
    # Get model configuration
    config = get_model_config(model_name)
    
    # Override with provided values
    gpu_mem = gpu_memory_utilization or config.gpu_memory_utilization
    max_len = max_model_len or config.max_model_len
    tp_size = tensor_parallel_size or config.tensor_parallel_size
    quant = quantization or config.quantization
    
    # Print banner
    print(f"""
------------------------------------------------------------------------
                      VLLM MODEL SERVER                            
------------------------------------------------------------------------
  Model:      {config.name:<55} 
  HF Path:    {config.hf_name:<55} 
  Host:       {host:<55} 
  Port:       {port:<55} 
  GPU Memory: {gpu_mem:<55} 
  Max Length: {max_len:<55} 
  TP Size:    {tp_size:<55} 
------------------------------------------------------------------------
    """)
    
    if config.description:
        print(f"Description: {config.description}\n")
    
    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.hf_name,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(max_len),
    ]
    
    # Add tensor parallelism if > 1
    if tp_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tp_size)])
    
    # Add quantization if specified
    if quant:
        cmd.extend(["--quantization", quant])
    
    # Add trust remote code if needed
    if config.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    print(f"Command: {' '.join(cmd)}\n")
    print(f"API endpoint: http://{host}:{port}/v1")
    print(f"   Compatible with OpenAI API format")
    print()
    print("="*72)
    print()
    
    if dry_run:
        print("Dry run mode - command not executed")
        return
    
    # Check if vllm is installed
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("vLLM is not installed!")
        print("   Install with: pip install vllm")
        sys.exit(1)
    
    # Start server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting vLLM server: {e}")
        sys.exit(1)


def check_gpu_availability() -> dict:
    """Check available GPUs and memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}
        
        count = torch.cuda.device_count()
        devices = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "memory_gb": props.total_memory / (1024**3),
            })
        
        return {"available": True, "count": count, "devices": devices}
    except Exception as e:
        return {"available": False, "count": 0, "error": str(e)}


def get_cache_dir() -> str:
    """Get the HuggingFace cache directory."""
    return os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))


def list_cached_models() -> list:
    """List all cached models."""
    import shutil
    
    cache_dir = get_cache_dir()
    hub_cache = os.path.join(cache_dir, "hub")
    
    if not os.path.exists(hub_cache):
        return []
    
    models = []
    for item in os.listdir(hub_cache):
        if item.startswith("models--"):
            model_path = os.path.join(hub_cache, item)
            # Parse model name from directory name
            # Format: models--org--model-name
            parts = item.replace("models--", "").split("--")
            model_name = "/".join(parts)
            
            # Get size
            size_bytes = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, files in os.walk(model_path)
                for f in files
            )
            size_gb = size_bytes / (1024**3)
            
            models.append({
                "name": model_name,
                "path": model_path,
                "size_gb": size_gb
            })
    
    return sorted(models, key=lambda x: x["size_gb"], reverse=True)


def show_cache_info() -> dict:
    """Show cache information."""
    cache_dir = get_cache_dir()
    models = list_cached_models()
    total_size = sum(m["size_gb"] for m in models)
    
    print("\n" + "="*60)
    print(" HUGGINGFACE MODEL CACHE")
    print("="*60)
    print(f"   Cache directory: {cache_dir}")
    print(f"   Total models: {len(models)}")
    print(f"   Total size: {total_size:.2f} GB")
    print()
    
    if models:
        print("   Cached models:")
        for i, m in enumerate(models, 1):
            print(f"   [{i}] {m['name']}: {m['size_gb']:.2f} GB")
    else:
        print("   (No models cached)")
    
    print("="*60 + "\n")
    
    return {"cache_dir": cache_dir, "models": models, "total_size_gb": total_size}


def clean_model_cache(model_name: str = None, all_models: bool = False) -> bool:
    """
    Clean cached models.
    
    Args:
        model_name: Specific model to remove (e.g., "Qwen/Qwen3-30B")
        all_models: If True, remove all cached models
        
    Returns:
        True if successful
    """
    import shutil
    
    cache_dir = get_cache_dir()
    hub_cache = os.path.join(cache_dir, "hub")
    
    if not os.path.exists(hub_cache):
        print("   Cache directory doesn't exist")
        return False
    
    if all_models:
        print(f"\nWARNING: This will delete ALL cached models!")
        confirm = input("   Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("   Cancelled.")
            return False
        
        models = list_cached_models()
        total_freed = 0
        for m in models:
            try:
                shutil.rmtree(m["path"])
                print(f"   Deleted: {m['name']} ({m['size_gb']:.2f} GB)")
                total_freed += m["size_gb"]
            except Exception as e:
                print(f"   Failed to delete {m['name']}: {e}")
        
        print(f"\n   Freed {total_freed:.2f} GB")
        return True
    
    elif model_name:
        # Find the model
        models = list_cached_models()
        target = None
        for m in models:
            if model_name.lower() in m["name"].lower():
                target = m
                break
        
        if not target:
            print(f"   Model '{model_name}' not found in cache")
            print("   Use --cache-info to see cached models")
            return False
        
        print(f"\nWill delete: {target['name']} ({target['size_gb']:.2f} GB)")
        confirm = input("   Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("   Cancelled.")
            return False
        
        try:
            shutil.rmtree(target["path"])
            print(f"   Deleted: {target['name']}")
            print(f"   Freed {target['size_gb']:.2f} GB")
            return True
        except Exception as e:
            print(f"   Failed: {e}")
            return False
    
    else:
        print("   Specify --clean <model_name> or --clean-all")
        return False


if __name__ == "__main__":
    # Quick test
    gpu_info = check_gpu_availability()
    print("GPU Status:", gpu_info)
    show_cache_info()
