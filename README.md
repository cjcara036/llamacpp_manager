# llama.cpp Manager

The ultimate local AI runner for Windows. This PowerShell script simplifies running llama.cpp by automatically managing backends, downloading models from Hugging Face, generating optimized configurations, and providing both CLI and server modes with full vision/multimodal support.

## Features

- 🚀 **Zero-Setup Experience** - Just run the script, no manual installation required
- 🎯 **Hardware-Aware** - Auto-detects your GPU/CPU and recommends the optimal backend
- 📥 **Auto-Downloader** - Downloads llama.cpp backends and GGUF models with resume support
- ⚙️ **Smart Configs** - Auto-generates JSON configs with 40+ tunable parameters
- 🖼️ **Vision Support** - Full multimodal model support (LLaVA, Ministral, Qwen2-VL)
- 🔌 **Dual Modes** - Interactive CLI for testing + HTTP Server for apps
- 📚 **Built-in Help** - Auto-generated reference guides for every config parameter
- 🔄 **Resume Support** - Interrupted downloads can be resumed automatically

## Requirements

- **Windows 10/11**
- **PowerShell 5.1+** (included with Windows)
- **curl** (included with Windows 10+)
- Internet connection for initial downloads

## Quick Start

1. **Run the script:**
   ```powershell
   .\llamacpp-manager.ps1
   ```

2. **Select a backend** (the script will recommend the best one for your system):
   - Option 1: CUDA (NVIDIA GPUs - Fastest)
   - Option 2: Vulkan (AMD/Intel GPUs - Good compatibility)
   - Option 3: CPU (No GPU / Basic fallback)
   - Option 4: ARM64 (Native Snapdragon / ARM)

3. **Download a model** (or select an existing one):
   - Enter a Hugging Face repo ID (e.g., `unsloth/Ministral-3-8B-Instruct-2512-GGUF`)
   - Select the quantization file you want
   - Optionally download a vision projector file for multimodal models

4. **Choose a run mode:**
   - **CLI Mode** - Interactive terminal for quick testing
   - **Server Mode** - HTTP API endpoint (OpenAI-compatible)

That's it! The script handles everything else.

## Usage Guide

### Step 1: Backend Selection

The script will scan your hardware and recommend the best backend:

- **CUDA** - Best for NVIDIA GPUs (fastest performance)
- **Vulkan** - Works with AMD, Intel, and other GPUs
- **CPU** - Fallback for systems without GPUs
- **ARM64** - Native execution on Snapdragon/ARM Windows devices

The backend is downloaded from the official [llama.cpp GitHub releases](https://github.com/ggerganov/llama.cpp/releases).

### Step 2: Model Download

Download any GGUF model from Hugging Face:

```powershell
Enter the Repo ID: unsloth/Ministral-3-8B-Instruct-2512-GGUF
```

The script will:
- List all available GGUF files in the repo
- Let you select the quantization (Q4_K_XL, Q5_K_M, Q8_0, etc.)
- Offer to download vision projector files for multimodal models
- Auto-generate a JSON config file with optimal defaults
- Create a comprehensive `.help.txt` reference guide

**Private Repos:** If the model requires a Hugging Face token, the script will prompt you to enter your `HF_TOKEN`.

### Step 3: Run Mode Selection

#### CLI Mode (Interactive Terminal)

Perfect for quick testing and experimentation:

- Type your questions directly
- For vision models: Type `/image C:\path\to\photo.jpg` then your question on the next line
- Press `Ctrl+C` to exit

#### Server Mode (HTTP API)

Run a local OpenAI-compatible API server:

- Default endpoint: `http://127.0.0.1:8080/v1`
- Compatible with chat UIs (Open WebUI, text-generation-webui, etc.)
- Supports streaming responses
- Press `Ctrl+C` to stop

## Configuration

Each model gets its own JSON configuration file in the `Configs/` directory with 40+ tunable parameters:

```json
{
  "model_name": "ministral-3-8b-instruct-2512-ud-q4_k_xl",
  "gguf_path": "C:\\path\\to\\model.gguf",
  "mmproj_path": "C:\\path\\to\\mmproj-f16.gguf",
  "n_ctx": 8192,
  "n_gpu_layers": 999,
  "flash_attn": "auto",
  "cache_type_k": "q8_0",
  "cache_type_v": "q8_0",
  "host": "127.0.0.1",
  "port": 8080,
  ...
}
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_ctx` | Context window size (tokens) | 8192, 16384, 32768 |
| `n_gpu_layers` | Layers offloaded to GPU (999 = all) | 0, 33, 999 |
| `flash_attn` | Reduce KV cache VRAM usage | "on", "off", "auto" |
| `cache_type_k` | Key cache quantization | "f16", "q8_0", "q4_0" |
| `cache_type_v` | Value cache quantization | "f16", "q8_0", "q4_0" |
| `temp` | Sampling temperature | 0.0 - 2.0 (lower = more deterministic) |
| `top_p` | Nucleus sampling | 0.9, 0.95 |
| `n_threads` | CPU threads (physical cores) | Your CPU core count |

### Reference Guide

Every config file has a matching `.help.txt` file with comprehensive documentation for all parameters. Open it to learn about:

- Context and batching options
- GPU offloading and multi-GPU setups
- Flash attention and KV cache quantization
- RoPE scaling for context extension
- Group attention and YaRN methods
- Sampling and generation parameters
- Server-specific settings

## Supported Backends

### CUDA (NVIDIA GPUs)
- **Best for:** NVIDIA GeForce, RTX, Quadro, Tesla GPUs
- **Performance:** Fastest available
- **Features:** Full flash attention support, CUDA graphs

### Vulkan (AMD/Intel/Other GPUs)
- **Best for:** AMD Radeon, Intel Arc, integrated GPUs
- **Performance:** Good GPU acceleration
- **Features:** Wide hardware compatibility

### CPU
- **Best for:** Systems without GPUs or for testing
- **Performance:** Slower but functional
- **Note:** Flash attention not supported on CPU backend

### ARM64
- **Best for:** Snapdragon/ARM Windows devices
- **Performance:** Native ARM execution

## Vision/Multimodal Support

The script fully supports vision language models (VLMs):

### Supported Vision Models
- Ministral 3 8B Instruct
- LLaVA (all variants)
- Qwen2-VL
- MiniCPM-V
- And any other llama.cpp-compatible vision model

### How Vision Works

1. **Download both files:**
   - Main model file (e.g., `Ministral-3-8B-Instruct-2512-GGUF`)
   - Vision projector file (e.g., `mmproj-f16.gguf`)

2. **In CLI Mode:**
   ```
   /image C:\path\to\photo.jpg
   What do you see in this image?
   ```

3. **In Server Mode:**
   The API accepts image payloads - send base64-encoded images with your prompts.

## Directory Structure

```
llama_cpp/
├── llamacpp-manager.ps1           # Main script
├── README.md                       # This file
├── llamacpp_bin/                  # Backend binaries
│   ├── cuda/                       # CUDA backend
│   ├── vulkan/                     # Vulkan backend
│   ├── cpu/                        # CPU backend
│   └── arm64/                      # ARM64 backend
├── Models/                         # Downloaded GGUF files
│   ├── Ministral-3-8B-Instruct-2512-UD-Q4_K_XL.gguf
│   └── mmproj-F16.gguf
└── Configs/                        # Model configurations
    ├── ministral-3-8b-instruct-2512-ud-q4_k_xl.json
    └── ministral-3-8b-instruct-2512-ud-q4_k_xl.help.txt
```

## Troubleshooting

### "Model not found" error
Check the `gguf_path` in your JSON config file. The path must match where the model file is located.

### Backend download fails
- Check your internet connection
- Try running the script again (it will offer to update the backend)
- Verify GitHub releases are accessible

### Hugging Face download fails
- Some models require a Hugging Face account and token
- Create a token at https://huggingface.co/settings/tokens
- Re-run the script and enter the token when prompted

### Out of memory errors
- Reduce `n_ctx` (context size)
- Reduce `n_gpu_layers` (partial CPU offloading)
- Enable `cache_type_k` and `cache_type_v` quantization
- Enable `flash_attn` = "on"

### Slow performance
- For NVIDIA: Use CUDA backend with `n_gpu_layers = 999`
- Enable `flash_attn = "auto"` or `"on"`
- Set `n_threads` to your physical CPU core count
- Consider using a smaller quantization (Q4_K_M instead of Q8_0)

## FAQ

**Q: Do I need to install anything?**  
A: No! Just run the PowerShell script. It downloads everything automatically.

**Q: Can I run multiple models at once?**  
A: Yes, if running in Server Mode, set `parallel` in your config to enable multiple concurrent inference slots.

**Q: How do I update to the latest llama.cpp version?**  
A: Re-run the script and select "y" when it asks to pull the latest update for your backend.

**Q: Can I use this with chat UIs?**  
A: Yes! Run in Server Mode and connect any OpenAI-compatible chat UI to `http://127.0.0.1:8080/v1`.

**Q: What's the difference between llama-cli and llama-mtmd-cli?**  
A: `llama-mtmd-cli` is the newer vision-capable CLI. The script automatically uses it for vision models.

**Q: How do I know if my model supports vision?**  
A: Check if the Hugging Face repo has a `*mmproj*.gguf` file. The script will detect and offer to download it.

**Q: Can I use models from sources other than Hugging Face?**  
A: Yes! Manually place GGUF files in the `Models/` directory and create/edit a JSON config file in `Configs/`.

**Q: What's the recommended quantization?**  
A: For most users, Q4_K_M or Q4_K_XL provides a good balance of quality and size. Use Q8_0 for higher quality if you have enough RAM/VRAM.

## License

This script is provided as-is for managing llama.cpp. The llama.cpp project is licensed under the MIT License.

## Related Links

- [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
- [llama.cpp Documentation](https://llama-cpp-python.readthedocs.io/)
- [Hugging Face GGUF Models](https://huggingface.co/models?library=gguf)
- [Open WebUI](https://openwebui.com/) - A great chat UI compatible with this server

## Contributing

This is a standalone management script. For improvements to the underlying llama.cpp, please contribute to the official repository.

---

**Enjoy running local AI! 🚀**