#!/usr/bin/env python3
"""
Model Server Entry Point - Host local LLMs with vLLM.

Usage:
    python -m src.serve --model qwen                    # Start Qwen server
    python -m src.serve --model llama --port 8001       # Start Llama on port 8001
    python -m src.serve --list                          # List available models
    python -m src.serve --cache-info                    # Show cached models
    python -m src.serve --clean qwen                    # Delete cached Qwen model
    python -m src.serve --clean-all                     # Delete ALL cached models

Examples:
    # Host Qwen3-30B
    python -m src.serve --model qwen
    
    # Host with custom settings
    python -m src.serve --model qwen --port 8000 --gpu-memory 0.85
    
    # Check available models
    python -m src.serve --list
    
    # Check GPU status
    python -m src.serve --gpu-info
    
    # Clean up models
    python -m src.serve --cache-info       # See what's cached
    python -m src.serve --clean qwen       # Delete specific model
    python -m src.serve --clean-all        # Delete all models
"""

import argparse
import sys
import os

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.serving.models import list_models, get_model_config
from src.serving.server import (
    start_server, check_gpu_availability, 
    show_cache_info, clean_model_cache
)


def main():
    parser = argparse.ArgumentParser(
        description="Host local LLMs with vLLM (OpenAI-compatible API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.serve --model qwen              # Host Qwen3-30B
  python -m src.serve --model llama --port 8001 # Host Llama on port 8001
  python -m src.serve --list                    # List available models
  python -m src.serve --cache-info              # Show cached models
  python -m src.serve --clean qwen              # Delete specific model
  python -m src.serve --clean-all               # Delete ALL models
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (from registry) or HuggingFace path"
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)"
    )
    
    # Model settings
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=None,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Max model context length"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size (number of GPUs)"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method"
    )
    
    # Utility options
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show GPU information"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )
    
    # Cache management
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cached models and their sizes"
    )
    parser.add_argument(
        "--clean",
        type=str,
        metavar="MODEL",
        help="Delete specific model from cache (e.g., --clean qwen)"
    )
    parser.add_argument(
        "--clean-all",
        action="store_true",
        help="Delete ALL cached models (will ask for confirmation)"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list:
        list_models(verbose=True)
        return
    
    if args.gpu_info:
        info = check_gpu_availability()
        print("\n" + "="*50)
        print("🖥️  GPU INFORMATION")
        print("="*50)
        if info["available"]:
            print(f"   Available: Yes ({info['count']} GPU(s))")
            for dev in info.get("devices", []):
                print(f"   [{dev['index']}] {dev['name']}: {dev['memory_gb']:.1f} GB")
        else:
            print(f"   Available: No")
            if "error" in info:
                print(f"   Error: {info['error']}")
        print("="*50 + "\n")
        return
    
    if args.cache_info:
        show_cache_info()
        return
    
    if args.clean:
        clean_model_cache(model_name=args.clean)
        return
    
    if args.clean_all:
        clean_model_cache(all_models=True)
        return
    
    # Require model for serving
    if not args.model:
        parser.print_help()
        print("\n❌ Error: --model is required to start the server")
        print("   Use --list to see available models")
        sys.exit(1)
    
    # Start server
    start_server(
        model_name=args.model,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_len,
        tensor_parallel_size=args.tp_size,
        quantization=args.quantization,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

