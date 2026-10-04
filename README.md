# llama.cpp Manager

A single, self-contained Bash script that downloads, configures, and runs **llama.cpp** models — no manual compilation, no manual flag-hunting.

```
llamacpp-manager.sh
├── Downloads the right pre-built binary for your hardware
├── Pulls GGUF models straight from Hugging Face
├── Manages per-model JSON configs (auto-synced to the binary's real flags)
├── Runs in Server, CLI, or interactive terminal mode
└── SWEEP mode: auto-benchmarks your setup for optimal tok/s
```

---

## Requirements

| Dependency | Why |
|---|---|
| `bash` 4+ | Associative arrays, `readarray`, etc. |
| `curl` | Downloads |
| `jq` | JSON config parsing & writing |
| `tar` / `unzip` | Extract binaries & archives |
| `lspci` (pciutils) | GPU detection |
| `grep`, `sed`, `awk` | Text processing |

> **Auto-installer:** The script checks for missing dependencies on launch and attempts to install them via `apt`, `dnf`, `pacman`, or `zypper` automatically.

---

## Quick Start

```bash
chmod +x llamacpp-manager.sh
./llamacpp-manager.sh
```

That's it. The script walks you through everything.

---

## What Happens Step-by-Step

### 1. Hardware Detection

The script scans your CPU architecture and GPU(s), then **recommends** the best backend:

| Backend | Use when |
|---|---|
| **CUDA** | NVIDIA GPU (fastest) |
| **Vulkan** | Intel / AMD GPU (good compatibility) |
| **CPU / AVX2** | No GPU, or you just want simplicity |
| **ARM64** | Linux on ARM (native) |
| **ROCm** | AMD GPU (native accelerated) |

Press **Enter** to accept the recommendation, or type the number to choose manually.

### 2. Binary Download

- Fetches the latest release from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases) via GitHub API
- Supports **resume** on interrupted downloads (`curl -C -`)
- Extracts to `./llamacpp_bin/<backend>/`
- If a backend folder already exists, it asks whether to **update** or **reuse**

### 3. Model Download (Hugging Face)

- Paste a **Repo ID** or full **Hugging Face URL** (e.g. `unsloth/Qwen3.8-27B-GGUF`)
- Lists all `.gguf` files in the repo
- Optionally downloads an **mmproj** vision projector file
- Supports **gated repos** — prompts for an `HF_TOKEN` on the first failed attempt
- All model files land in `./Models/<repo_name>/`

### 4. Auto-Generated JSON Config

Every downloaded model gets a config file in `./Configs/`:

```json
{
  "model_name": "qwen3-8b-instruct",
  "gguf_path": "./Models/unsloth/Qwen3.8-27B-GGUF/Qwen3-8B-Instruct-Q4_K_M.gguf",
  "mmproj_path": null,
  "n_ctx": 8192,
  "batch_size": 512,
  "n_gpu_layers": 33,
  "flash_attn": "auto",
  "perf": true,
  "host": "127.0.0.1",
  "port": 8080,
  "cont_batching": true
}
```

### 5. Flag Auto-Discovery (the magic part)

The script runs `llama-server --help` and **parses every flag the binary actually supports**. Then:

- **Resolves** your config keys to the correct CLI flag name (handles renames across versions, e.g. `-m` → `--model`)
- **Adds** any new flags the binary exposes that your config doesn't have yet (as `null`)
- **Skips** keys the binary doesn't understand (with a warning)

This means your JSON config stays compatible even as llama.cpp evolves its CLI.

### 6. Interactive Settings Editor

Press **`e`** in the model menu to open the editor. For every setting you get:

- The **current value**
- The **matched CLI flag**
- The **binary's help text** for that flag
- A **value hint** (e.g. `Boolean: [ true | false | null ]`)

Type a new value to change it. Type `null` or leave blank to reset.

### 7. Run Modes

| Mode | Binary | What it does |
|---|---|---|
| **Server** | `llama-server` | HTTP API server for OpenAI-compatible endpoints, chat UIs, apps |
| **CLI** | `llama-cli` / `llama-run` / `llama-mtmd-cli` | Interactive terminal chat |
| **SWEEP** | `llama-server` | Benchmark & auto-optimize (see below) |

### 8. SWEEP Mode 🏎️

Automatically benchmarks your setup by sweeping through parameter combinations in phases:

| Phase | Parameter | Values Tested |
|---|---|---|
| 0 | `batch_size` | 128, 256, 512, 1024, 2048 |
| 1 | `parallel` | 1, 2, 4, 8, 16 |
| 2 | `cont_batching` | true / false |
| 3 | `use_mmap` | true / false |
| 4 | `use_mlock` | false / true *(root only)* |
| 5 | `numa` | null / distribute / isolate *(multi-NUMA only)* |

Each phase tests the best result from the previous phase, so it converges quickly. At the end you get a **results table** with tok/s for every combination so you can pick the winner.

---

## Directory Structure

```
./
├── llamacpp-manager.sh          # The script
├── llamacpp_bin/
│   ├── cuda/                    # or vulkan/, cpu/, arm64/, rocm/
│   │   ├── llama-server
│   │   ├── llama-cli
│   │   ├── llama-run
│   │   ├── llama-mtmd-cli
│   │   └── ...
├── Models/
│   ├── unsloth/                 # One folder per HF repo
│   │   ├── *.gguf
│   │   └── mmproj*.gguf        # (optional vision projector)
└── Configs/
    ├── qwen3-8b-instruct.json   # One config per model
    └── llama-3-8b.json
```

---

## Typical Workflow

```bash
./llamacpp-manager.sh
#  → Pick backend (or Enter for auto-detect)
#  → 0. Download New Model → paste HF repo
#  → Select .gguf file
#  → e. Edit Settings (tune n_ctx, n_gpu_layers, etc.)
#  → 1. Server Mode  (or 2. CLI, or 0. SWEEP)
#  → Server starts → Ctrl+C to stop
```

---

## Tips

- **GPU layers (`n_gpu_layers`):** Set to `-1` to offload all layers to the GPU. Adjust down if you hit OOM.
- **Vision models:** The mmproj file enables image understanding. If it's in the repo, the script offers to download it.
- **LAN access:** Change `"host": "127.0.0.1"` to `"0.0.0.0"` in the config to let other devices on your network connect to the server.
- **Resuming:** Both binary and model downloads support resume. If interrupted, just re-run the script — it picks up where it left off.
- **Updating binaries:** Run the script again and choose your backend — it detects the existing install and asks whether to pull the latest release.

---

## License

This script is provided as-is. The llama.cpp binaries are under their respective upstream licenses.
