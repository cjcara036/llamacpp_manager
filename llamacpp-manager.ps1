<#
.SYNOPSIS
    The ultimate local AI runner. Manages backends (x64 & ARM64), downloads models
    from Hugging Face, auto-generates configs, and runs llama.cpp cleanly.
    Supports text-only and vision/multimodal (VLM) models.
#>

$BaseInstallDir = "$PSScriptRoot\llamacpp_bin"
$ModelsDir      = "$PSScriptRoot\Models"
$ConfigsDir     = "$PSScriptRoot\Configs"

if (-not (Test-Path $ModelsDir))  { New-Item -ItemType Directory -Path $ModelsDir  | Out-Null }
if (-not (Test-Path $ConfigsDir)) { New-Item -ItemType Directory -Path $ConfigsDir | Out-Null }

# =============================================================================
# HELPERS
# =============================================================================
function Add-Arg {
    param([System.Collections.Generic.List[string]]$list, [string]$flag, $value)
    if ($null -ne $value -and "$value" -ne "") {
        $list.Add($flag)
        $list.Add("$value")
    }
}

  function Add-Switch {
    param([System.Collections.Generic.List[string]]$list, [string]$flag, $value)
    if ($value -eq $true) { $list.Add($flag) }
}

function Get-FlashAttnValue {
    param([object]$value)
    if ($value -eq $true) { return "on" }
    elseif ($value -eq $false) { return "off" }
    else { return "auto" }
}

# =============================================================================
# DEFAULT CONFIG TEMPLATE
# =============================================================================
function New-DefaultConfig {
    param([string]$ModelName, [string]$GgufPath, [string]$MmprojPath = $null)
    return [ordered]@{
        "model_name"          = $ModelName
        "gguf_path"           = $GgufPath
        "mmproj_path"         = $MmprojPath
        "n_ctx"               = 8192
        "batch_size"          = 512
        "ubatch_size"         = $null
        "n_gpu_layers"        = 33
        "split_mode"          = $null
        "main_gpu"            = $null
        "tensor_split"        = $null
        "n_threads"           = $null
        "n_threads_batch"     = $null
        "flash_attn"          = "auto"   # "on", "off", or "auto" (auto = on if supported, off otherwise)
        "cache_type_k"        = $null
        "cache_type_v"        = $null
        "use_mmap"            = $true    # keep true — mmap is correct for Windows; false+mlock causes VirtualLock failures
        "use_mlock"           = $false   # requires elevated privileges on Windows; leave false unless you know you need it
        "numa"                = $null
        "defrag_thold"        = $null
        "rope_freq_base"      = $null
        "rope_freq_scale"     = $null
        "rope_scaling"        = $null
        "yarn_ext_factor"     = $null
        "yarn_attn_factor"    = $null
        "yarn_beta_fast"      = $null
        "yarn_beta_slow"      = $null
        "yarn_orig_ctx"       = $null
        "grp_attn_n"          = $null
        "grp_attn_w"          = $null
        "n_predict"           = $null
        "seed"                = $null
        "temp"                = $null
        "top_k"               = $null
        "top_p"               = $null
        "min_p"               = $null
        "tfs_z"               = $null
        "typical_p"           = $null
        "repeat_penalty"      = $null
        "repeat_last_n"       = $null
        "presence_penalty"    = $null
        "frequency_penalty"   = $null
        "host"                = "127.0.0.1"
        "port"                = 8080
        "parallel"            = $null
        "cont_batching"       = $true
        "chat_template"       = $null   # e.g. "chatml", "llama3", "gemma", "mistral" — null = auto-detect
        "system_prompt"       = $null
    }
}

# =============================================================================
# HELP FILE GENERATOR
# Uses a here-string so Unicode, ampersands, pipes, arrows, etc. are all safe.
# =============================================================================
function Write-ConfigHelp {
    param([string]$JsonPath)
    $helpPath = $JsonPath -replace '\.json$', '.help.txt'
    $name     = Split-Path $JsonPath -Leaf

    $content = @"
===============================================================================
 LLAMA.CPP CONFIG REFERENCE GUIDE
 Generated alongside: $name
 Edit the matching .json file to change settings.
 Set a value to null to let llama.cpp use its built-in default.
===============================================================================


--- IDENTITY -------------------------------------------------------------------

  model_name   (string)
               Display name / alias for the model.
               Used as the model ID in the server API.
               Example: "ministral-3b-q4"

  gguf_path    (string)
               Absolute path to the main .gguf model file.
               Example: "C:\Users\you\Models\model.gguf"


--- VISION / MULTIMODAL (VLM) --------------------------------------------------

  mmproj_path  (string or null)
               Path to the multimodal projector .gguf file.
               Required to enable image/vision support.
               Vision models ship with a second file typically named:
                 *mmproj*.gguf  or  *mmproj-f16.gguf
               Leave null for text-only models.
               When set, the script automatically:
                 - Uses llama-mtmd-cli for interactive CLI sessions
                 - Passes --mmproj to llama-server for API image support
               In CLI mode type:  /image C:\path\to\photo.jpg
               then ask your question on the next line.
               Example: "C:\Users\you\Models\model-mmproj-f16.gguf"


--- CONTEXT AND BATCHING -------------------------------------------------------

  n_ctx        (integer, power of 2 recommended)
               KV cache / context window size in tokens.
               Total tokens the model can hold at once (prompt + reply).
               Larger values use more VRAM / RAM.
               Common values : 2048, 4096, 8192, 16384, 32768
               Default       : 512 (llama.cpp default; set higher for chat)

  batch_size   (integer)
               Max prompt tokens processed in one logical batch.
               Higher = faster prompt ingestion, more RAM usage.
               Common values : 128, 256, 512, 1024, 2048
               Default       : 2048

  ubatch_size  (integer or null)
               Physical micro-batch size within each logical batch.
               Tune down if you hit VRAM limits during prompt processing.
               Must be <= batch_size.
               Common values : 128, 256, 512
               Default       : same as batch_size (null = auto)


--- GPU OFFLOAD ----------------------------------------------------------------

  n_gpu_layers (integer)
               Number of transformer layers to offload to GPU.
               999 = offload everything including embeddings (recommended).
               0   = CPU-only, no GPU used at all.
               For partial offload use a value between 1 and the model total
               layer count (typically 16-80 depending on model size).
               Common values : 0, 20, 33, 80, 999

  split_mode   (string or null)
               How to distribute layers across multiple GPUs.
               Accepted : "none", "layer", "row"
                 none  = only use main_gpu
                 layer = split whole layers across GPUs (recommended)
                 row   = split weight rows across GPUs (advanced)
               Default : "layer"

  main_gpu     (integer or null)
               Index of the primary GPU, zero-based.
               Default : 0

  tensor_split (string or null)
               VRAM ratio per GPU, comma-separated.
               Example: "3,1" means 75% GPU0 and 25% GPU1
               Example: "1,1" means 50/50 split
               Default : null (auto-balance)


--- THREADING ------------------------------------------------------------------

  n_threads    (integer or null)
               CPU threads for token generation (decode phase).
               Best set to your physical core count, not hyperthreads.
               Example: 8 for an 8-core CPU
               Default : auto-detect

  n_threads_batch (integer or null)
               CPU threads for prompt ingestion (prefill phase).
               Can be set higher than n_threads as prefill parallelises better.
               Example: 16
               Default : same as n_threads


--- FLASH ATTENTION ------------------------------------------------------------

  flash_attn   ("on", "off", or "auto")
               Rewrites the attention kernel to use far less VRAM for long
               contexts. Typically 2-4x VRAM savings on the KV cache.
               Most modern GGUFs support this (Llama-3, Mistral, Qwen, Gemma).
               Note: not supported on the plain CPU backend.
               Accepted : "on", "off", "auto"
                 on   = always enable (errors if backend does not support it)
                 off  = always disable
                 auto = enable if the backend supports it, otherwise skip
               Default  : "auto"


--- KV CACHE QUANTIZATION ------------------------------------------------------

  Reduces VRAM used by the KV cache at a small quality cost.
  Works best combined with flash_attn = true.

  cache_type_k (string or null)
               Data type for the Key cache.
               Accepted : "f32", "f16", "bf16", "q8_0", "q4_0",
                          "q4_1", "iq4_nl", "q5_0", "q5_1"
               Recommended:
                 Balanced   -> "q8_0"  (minimal quality loss, good savings)
                 Aggressive -> "q4_0"  (more savings, slight quality drop)
               Default : "f16"

  cache_type_v (string or null)
               Data type for the Value cache.
               Accepted : same options as cache_type_k above.
               Tip: Keys are more quality-sensitive than Values.
                    You can use a lower quant on V for extra savings.
               Example pair: cache_type_k = "q8_0", cache_type_v = "q4_0"
               Default : "f16"


--- MEMORY MAPPING AND LOCKING -------------------------------------------------

  use_mmap     (true or false)
               Maps the model into virtual memory instead of fully loading
               into RAM. Enables fast load and lets the OS page model parts
               to disk if needed. Recommended: true.
               Accepted : true, false
               Default  : true

  use_mlock    (true or false)
               Locks the model in physical RAM, preventing the OS from
               paging it out. Eliminates latency spikes on long sessions
               but requires enough free RAM and may need admin privileges.
               Accepted : "on", "off", "auto"
                 on   = always enable (errors if backend does not support it)
                 off  = always disable
                 auto = enable if the backend supports it, otherwise skip
               Default  : "auto"


--- NUMA  (multi-socket servers only, ignore on desktops and laptops) ----------

  numa         (string or null)
               Non-Uniform Memory Access optimisation strategy.
               Accepted : "distribute", "isolate", "numactl"
                 distribute = spread threads across NUMA nodes
                 isolate    = restrict to a single NUMA node
                 numactl    = use the system numactl tool
               Default : null (disabled)


--- KV CACHE DEFRAGMENTATION ---------------------------------------------------

  defrag_thold (float or null)
               Threshold at which the KV cache is defragmented.
               During long multi-turn conversations the cache fragments,
               wasting slots. This reclaims them automatically.
               Range   : 0.0 to 1.0  (fraction of fragmentation to trigger at)
               Example : 0.1 triggers defrag when 10% of cache is fragmented
               -1      = disabled
               Default : null (disabled)


--- ROPE SCALING  (context length extension) -----------------------------------

  RoPE (Rotary Position Embedding) lets you run a model beyond its trained
  context length. Quality degrades past roughly 2-4x extension.

  rope_freq_base  (float or null)
               Base frequency for RoPE. Larger = more tolerance for long ctx.
               Modern models like Llama-3 use 500000.
               Default : model built-in value

  rope_freq_scale (float or null)
               Linear scale factor applied to RoPE frequencies.
               Values below 1.0 compress frequencies, extending context.
               Example : 0.5 gives approx 2x context extension
               Default : 1.0 (no scaling)

  rope_scaling  (string or null)
               Scaling method to use.
               Accepted : "none", "linear", "yarn"
                 none   = no scaling
                 linear = simple linear frequency scaling
                 yarn   = YaRN method (best quality for large extensions)
               Default  : null


--- YARN  (advanced context extension, use with rope_scaling = "yarn") ---------

  yarn_ext_factor   (float or null)
               Extrapolation vs interpolation mix. -1 = auto (recommended).
               Range   : -1.0 or 0.0 to 1.0
               Default : -1.0 (auto)

  yarn_attn_factor  (float or null)
               Scales attention magnitude to compensate for extended context.
               Range   : 0.0 to 2.0
               Default : 1.0

  yarn_beta_fast    (float or null)
               High-frequency extrapolation threshold.
               Default : 32.0

  yarn_beta_slow    (float or null)
               Low-frequency interpolation threshold.
               Default : 1.0

  yarn_orig_ctx     (integer or null)
               Original context size the model was trained with.
               Example : 4096 for Llama-2, 8192 for Llama-3
               Default : model built-in value


--- GROUP ATTENTION / SELF-EXTEND  (cheap context extension, no retraining) ----

  grp_attn_n   (integer or null)
               Group attention factor. Multiplies effective context length.
               Example : 8 gives 8x extension so 8192 becomes 65536 tokens
               Default : null (disabled)

  grp_attn_w   (integer or null)
               Width of the attention group window in tokens.
               Should match the model original training context size.
               Example : 512
               Default : 512


--- GENERATION AND SAMPLING ----------------------------------------------------

  n_predict    (integer or null)
               Max tokens to generate per response.
               -1 = unlimited, generate until EOS or context full
               Common values : 256, 512, 1024, 2048, -1
               Default  : -1

  seed         (integer or null)
               Random seed for reproducible outputs.
               -1 = random seed each run
               Example : 42
               Default  : -1

  temp         (float or null)
               Sampling temperature. Controls randomness of output.
               0.0  = fully deterministic, always picks highest prob token
               1.0  = neutral sampling
               Above 1.0 = more creative and random
               Range   : 0.0 to 2.0  (above 1.5 is rarely useful)
               Default : 0.8

  top_k        (integer or null)
               Limits next token selection to the K most probable tokens.
               0 = disabled
               Common values : 20, 40, 80
               Default  : 40

  top_p        (float or null)
               Nucleus sampling. Keeps tokens until cumulative probability
               reaches P. Works well combined with top_k.
               1.0 = disabled
               Range   : 0.0 to 1.0
               Common values : 0.9, 0.95
               Default  : 0.95

  min_p        (float or null)
               Removes tokens with probability below (min_p x top token prob).
               Good alternative to top_p at higher temperatures.
               0.0 = disabled
               Common values : 0.05, 0.1
               Default  : 0.05

  tfs_z        (float or null)
               Tail-Free Sampling. Removes low-probability tail tokens.
               1.0 = disabled
               Range   : 0.0 to 1.0
               Default : 1.0 (disabled)

  typical_p    (float or null)
               Locally Typical Sampling. Keeps tokens with information
               content close to the expected entropy of the distribution.
               1.0 = disabled
               Range   : 0.0 to 1.0
               Default : 1.0 (disabled)

  repeat_penalty  (float or null)
               Penalises tokens that have already appeared in the output.
               1.0 = no penalty
               Range   : 1.0 to 1.5  (1.1 to 1.3 is a good starting range)
               Default : 1.1

  repeat_last_n   (integer or null)
               How many past tokens to look back when applying repeat penalty.
               -1 = full context window
               0  = disabled
               Common values : 64, 128, 256, -1
               Default  : 64

  presence_penalty  (float or null)
               Flat penalty applied once for any token that has appeared.
               Encourages topic diversity regardless of repetition count.
               0.0 = disabled
               Range   : 0.0 to 1.0
               Default : 0.0

  frequency_penalty (float or null)
               Penalty scaled by how many times a token has appeared.
               More aggressively discourages repeated words than presence_penalty.
               0.0 = disabled
               Range   : 0.0 to 1.0
               Default : 0.0


--- SERVER-ONLY SETTINGS -------------------------------------------------------

  host         (string)
               IP address the HTTP server binds to.
               "127.0.0.1" = localhost only, not reachable from network
               "0.0.0.0"   = accessible from other devices on your LAN
               Default  : "127.0.0.1"

  port         (integer)
               TCP port for the HTTP API.
               Range   : 1024 to 65535
               Default : 8080
               API URL : http://<host>:<port>/v1

  parallel     (integer or null)
               Number of simultaneous inference slots.
               Each slot uses its own share of the KV cache (more VRAM).
               Common values : 1, 2, 4, 8
               Default  : 1

  cont_batching (true or false)
               Continuous batching. Processes new requests without waiting
               for the current batch to finish.
               Dramatically improves throughput for multi-user workloads.
               Accepted : true, false
               Default  : true

  chat_template (string or null)
               Override the chat template used to format messages sent to
               the model. By default llama-server auto-detects the template
               from the GGUF metadata, but it sometimes picks the wrong one
               (e.g. "Hermes 2 Pro" for a Qwen model), causing the model to
               return empty or garbled responses.
               Set this when the server logs show the wrong chat format, or
               when the model returns blank output on the first message.
               Common values:
                 "chatml"    — Qwen, Mistral-Nemo, many fine-tunes
                 "llama3"    — Meta Llama 3 / 3.1 / 3.2 / 3.3
                 "gemma"     — Google Gemma 2 / 3
                 "mistral"   — Mistral 7B v0.1 / v0.2
                 "phi3"      — Microsoft Phi-3 / Phi-3.5
                 "deepseek3" — DeepSeek-V3 / R1
                 "falcon3"   — Falcon 3
                 "command-r" — Cohere Command R / R+
               Full list: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
               Default  : null (auto-detect from GGUF metadata)

  system_prompt (string or null)
               Default system prompt injected at the start of every session.
               In CLI mode leave null to start with no injected text.
               Example  : "You are a helpful coding assistant."
               Default  : null


===============================================================================
 QUICK-START TUNING TIPS
===============================================================================

 CPU-only machine:
   flash_attn      = "off"        (not supported on CPU backend)
   cache_type_k    = "q8_0"
   cache_type_v    = "q8_0"
   n_threads       = <physical core count>
   n_threads_batch = <logical core count>
   n_gpu_layers    = 0

 NVIDIA GPU:
   flash_attn      = "auto"
   cache_type_k    = "q8_0"
   cache_type_v    = "q8_0"
   n_gpu_layers    = 999
   cont_batching   = true

 Vision / VLM model (e.g. Ministral 3 8B, LLaVA, Qwen2-VL, Qwen3-VL):
   mmproj_path     = "C:\path\to\mmproj-f16.gguf"
   chat_template   = "chatml"   (set this if the model returns empty responses)
   In CLI mode type /image C:\path\to\photo.jpg then ask your question.

===============================================================================
"@

    $content | Set-Content -Path $helpPath -Encoding UTF8
    Write-Host "  Reference guide: $(Split-Path $helpPath -Leaf)" -ForegroundColor DarkGray
}

# =============================================================================
# STEP 1: HARDWARE DETECTION
# =============================================================================
function Get-HardwareRecommendation {
    Write-Host "`nScanning hardware..." -ForegroundColor Cyan

    $arch = $env:PROCESSOR_ARCHITECTURE
    Write-Host "  Architecture : $arch" -ForegroundColor DarkGray

    try {
        $gpus     = Get-CimInstance -ClassName Win32_VideoController
        $gpuNames = $gpus.Name -join ", "
        Write-Host "  GPU(s)       : $gpuNames" -ForegroundColor DarkGray
    } catch {
        $gpuNames = "Unknown"
    }

    if ($arch -eq "ARM64") {
        $recommendedChoice = '4'
        $recommendedText   = "ARM64 CPU (Native Snapdragon / ARM)"
    } else {
        $recommendedChoice = '3'
        $recommendedText   = "CPU / AVX2 (No GPU / Basic Fallback)"

        if ($gpuNames -match "NVIDIA") {
            $recommendedChoice = '1'
            $recommendedText   = "CUDA (NVIDIA GPUs - Fastest)"
        } elseif ($gpuNames -match "AMD|Radeon|Intel") {
            $recommendedChoice = '2'
            $recommendedText   = "Vulkan (AMD/Intel GPUs - Good compatibility)"
        }
    }

    Write-Host "  Recommended  : Option $recommendedChoice - $recommendedText" -ForegroundColor Green
    return $recommendedChoice
}

# =============================================================================
# STEP 2: SELECT AND UPDATE BACKEND
# =============================================================================
function Setup-Backend {
    $recommended = Get-HardwareRecommendation

    Write-Host "`nWhich backend do you want to run?"
    Write-Host "  --- Standard x64 (Intel/AMD) ---"
    Write-Host "  1. CUDA   (NVIDIA GPUs)"
    Write-Host "  2. Vulkan (AMD/Intel GPUs)"
    Write-Host "  3. CPU    (includes AVX2 support)"
    Write-Host "  --- Windows on ARM ---"
    Write-Host "  4. ARM64  (Native Snapdragon / ARM execution)"

    $choice = Read-Host "`nEnter 1-4 (Enter = recommended: $recommended)"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = $recommended }

    switch ($choice) {
        '1' { $SubFolder = "cuda";   $searchPattern = "*bin-win-cuda-cu*-x64.zip" }
        '2' { $SubFolder = "vulkan"; $searchPattern = "*bin-win-vulkan-x64.zip"   }
        '3' { $SubFolder = "cpu";    $searchPattern = "*bin-win-cpu-x64.zip"      }
        '4' { $SubFolder = "arm64";  $searchPattern = "*bin-win*arm64*.zip"       }
        default { Write-Host "Invalid choice. Exiting." -ForegroundColor Red; exit }
    }

    $BackendPath = "$BaseInstallDir\$SubFolder"
    $ServerExe   = Get-ChildItem -Path $BackendPath -Filter "llama-server.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1

    if ($ServerExe) {
        $update = Read-Host "`nFound existing '$SubFolder' backend. Pull latest update? (y/N)"
        if ($update -eq 'y' -or $update -eq 'Y') {
            Write-Host "Removing old version..." -ForegroundColor Yellow
            Remove-Item -Path $BackendPath -Recurse -Force
            $ServerExe = $null
        } else {
            return $ServerExe.FullName
        }
    }

    Write-Host "`nFetching latest release from GitHub for '$SubFolder'..." -ForegroundColor Yellow
    try {
        $apiUrl  = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        $release = Invoke-RestMethod -Uri $apiUrl
        $asset   = $release.assets | Where-Object { $_.name -like $searchPattern } | Select-Object -First 1
        if (-not $asset) { throw "No matching asset found for pattern: $searchPattern" }

        $zipPath = "$PSScriptRoot\$($asset.name)"
        Write-Host "Downloading $($asset.name)..." -ForegroundColor Yellow
        $existing = Get-Item $zipPath -ErrorAction SilentlyContinue
        if ($existing -and $existing.Length -gt 0) {
            Write-Host "  Found partial download ($([math]::Round($existing.Length/1MB, 1)) MB). Resuming..." -ForegroundColor DarkYellow
        }
        & curl.exe -L "-C" "-" "--retry" "3" "--retry-delay" "5" "-o" $zipPath $asset.browser_download_url

        Write-Host "Extracting to $BackendPath..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $BackendPath -Force | Out-Null
        Expand-Archive -Path $zipPath -DestinationPath $BackendPath -Force
        Remove-Item $zipPath

        $ServerExe = Get-ChildItem -Path $BackendPath -Filter "llama-server.exe" -Recurse | Select-Object -First 1
        return $ServerExe.FullName
    } catch {
        Write-Host "Installation failed: $_" -ForegroundColor Red
        exit
    }
}

# =============================================================================
# STEP 3: HUGGING FACE DOWNLOADER
# =============================================================================
function Download-HuggingFaceModel {
    Write-Host "`n=== Hugging Face GGUF Downloader ===" -ForegroundColor Cyan
    Write-Host "Enter the Repo ID (e.g. 'unsloth/Ministral-3-8B-Instruct-2512-GGUF')"
    $repoId = (Read-Host "Repo ID").Trim()
    if ([string]::IsNullOrWhiteSpace($repoId)) { return $null }

    $headers = @{ "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" }

    Write-Host "`nQuerying Hugging Face API..." -ForegroundColor Yellow
    try {
        $repoData    = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$repoId" -Headers $headers
        $allGgufs    = $repoData.siblings | Where-Object { $_.rfilename -like "*.gguf" } | Select-Object -ExpandProperty rfilename
        $modelFiles  = $allGgufs | Where-Object { $_ -notlike "*mmproj*" }
        $mmprojFiles = $allGgufs | Where-Object { $_ -like "*mmproj*" }

        if ($modelFiles.Count -eq 0) {
            Write-Host "No .gguf model files found. Check the repo name." -ForegroundColor Red
            return $null
        }

        # Derive a safe subfolder name from the repo ID (use the part after the last '/')
        # and strip any characters that are invalid in Windows folder names.
        $repoFolderName = ($repoId -split '/')[-1] -replace '[\\/:*?"<>|]', '_'
        $ModelSubDir    = "$ModelsDir\$repoFolderName"
        if (-not (Test-Path $ModelSubDir)) {
            New-Item -ItemType Directory -Path $ModelSubDir | Out-Null
            Write-Host "  Created model folder: $repoFolderName" -ForegroundColor DarkGray
        } else {
            Write-Host "  Using existing model folder: $repoFolderName" -ForegroundColor DarkGray
        }

        # Select main model
        Write-Host "`nAvailable model files:"
        for ($i = 0; $i -lt $modelFiles.Count; $i++) {
            Write-Host "  $($i + 1). $($modelFiles[$i])"
        }
        $fileChoice   = Read-Host "Select a file (1-$($modelFiles.Count))"
        $selectedFile = $modelFiles[[int]$fileChoice - 1]
        $saveFileName = Split-Path $selectedFile -Leaf
        $savePath     = "$ModelSubDir\$saveFileName"

        # Vision: offer mmproj download
        $mmprojSavePath    = $null
        $selectedMmproj    = $null
        $mmprojSaveFileName = $null
        if ($mmprojFiles.Count -gt 0) {
            Write-Host "`nVision projector file(s) detected in this repo:" -ForegroundColor Cyan
            for ($i = 0; $i -lt $mmprojFiles.Count; $i++) {
                Write-Host "  $($i + 1). $($mmprojFiles[$i])"
            }
            $mmprojChoice = Read-Host "Download a projector for vision support? Enter number or Enter to skip"
            if (-not [string]::IsNullOrWhiteSpace($mmprojChoice)) {
                $selectedMmproj     = $mmprojFiles[[int]$mmprojChoice - 1]
                $mmprojSaveFileName = Split-Path $selectedMmproj -Leaf
                $mmprojSavePath     = "$ModelSubDir\$mmprojSaveFileName"
                Write-Host "  Will also download: $mmprojSaveFileName" -ForegroundColor Green
            }
        }

        # Download helper closure — supports resume via curl -C -
        function Invoke-HFDownload {
            param([string]$RemotePath, [string]$LocalPath, [string]$Token = "")
            $url = "https://huggingface.co/$repoId/resolve/main/$($RemotePath)?download=true"

            $existing = Get-Item $LocalPath -ErrorAction SilentlyContinue
            if ($existing -and $existing.Length -gt 0) {
                Write-Host "  Found partial file ($([math]::Round($existing.Length/1MB, 1)) MB). Resuming..." -ForegroundColor DarkYellow
            }

            $curlArgs = @(
                "-L",
                "-C", "-",
                "--retry", "3",
                "--retry-delay", "5",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            )
            if (-not [string]::IsNullOrWhiteSpace($Token)) {
                $curlArgs += "-H"
                $curlArgs += "Authorization: Bearer $Token"
            }
            $curlArgs += "-o"
            $curlArgs += $LocalPath
            $curlArgs += $url

            & curl.exe @curlArgs
            $info = Get-Item $LocalPath -ErrorAction SilentlyContinue
            return ($null -ne $info -and $info.Length -ge 1000000)
        }

        # Download main model (retry with token on failure)
        $hfToken    = ""
        $downloaded = $false
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            if ($attempt -eq 2) {
                Write-Host "`nDownload failed. Repo may require a Hugging Face token." -ForegroundColor Yellow
                $hfToken = (Read-Host "HF_TOKEN (Enter to cancel)").Trim()
                if ([string]::IsNullOrWhiteSpace($hfToken)) { Write-Host "Cancelled." -ForegroundColor Red; return $null }
                Write-Host "Retrying with token..." -ForegroundColor Yellow
            }
            Write-Host "`nDownloading $saveFileName..." -ForegroundColor Green
            $downloaded = Invoke-HFDownload -RemotePath $selectedFile -LocalPath $savePath -Token $hfToken
            if ($downloaded) { break }
            $info = Get-Item $savePath -ErrorAction SilentlyContinue
            if ($info) { Remove-Item $savePath }
        }

        if (-not $downloaded) {
            Write-Host "Download failed. Check your HF_TOKEN or try a different repo." -ForegroundColor Red
            return $null
        }

        # Download mmproj if selected
        if ($mmprojSavePath) {
            Write-Host "`nDownloading $mmprojSaveFileName..." -ForegroundColor Green
            $mmprojOk = Invoke-HFDownload -RemotePath $selectedMmproj -LocalPath $mmprojSavePath -Token $hfToken
            if (-not $mmprojOk) {
                Write-Host "mmproj download failed. Vision will be disabled in the config." -ForegroundColor Yellow
                $mmprojSavePath = $null
            } else {
                Write-Host "Projector downloaded successfully." -ForegroundColor Green
            }
        }

        # Write config and help file
        $modelName = ($saveFileName -replace "\.gguf$", "").ToLower()
        $jsonPath  = "$ConfigsDir\$modelName.json"
        $newConfig = New-DefaultConfig -ModelName $modelName -GgufPath $savePath -MmprojPath $mmprojSavePath
        $newConfig | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath
        Write-ConfigHelp -JsonPath $jsonPath

        Write-Host "`nConfig written to : $jsonPath" -ForegroundColor Green
        if ($mmprojSavePath) {
            Write-Host "Vision enabled.   : mmproj_path set in config." -ForegroundColor Cyan
        }
        Write-Host "Open the .help.txt for a full parameter reference." -ForegroundColor DarkGray
        return $jsonPath

    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $null
    }
}

# =============================================================================
# STEP 4: MODEL SELECTION MENU
# =============================================================================
function Select-ModelConfig {
    while ($true) {
        Write-Host "`n=== Model Selection ===" -ForegroundColor Cyan
        $configs = @(Get-ChildItem -Path $ConfigsDir -Filter "*.json")

        Write-Host "  0. [ Download New Model from Hugging Face ]" -ForegroundColor Green
        for ($i = 0; $i -lt $configs.Count; $i++) {
            Write-Host "  $($i + 1). $($configs[$i].Name)"
        }

        $modelChoice = Read-Host "`nSelect an option (0-$($configs.Count))"

        if ($modelChoice -eq '0') {
            $newConfigPath = Download-HuggingFaceModel
            if ($newConfigPath) { return $newConfigPath }
        } else {
            $index = [int]$modelChoice - 1
            if ($index -ge 0 -and $index -lt $configs.Count) {
                return $configs[$index].FullName
            } else {
                Write-Host "Invalid choice." -ForegroundColor Red
            }
        }
    }
}

# =============================================================================
# STEP 5: SELECT RUN MODE
# =============================================================================
function Select-RunMode {
    param([string]$RunnerPath)

    $backendDir = Split-Path $RunnerPath -Parent

    $serverExe = Join-Path $backendDir "llama-server.exe"
    $mtmdExe   = Join-Path $backendDir "llama-mtmd-cli.exe"  # vision CLI (newest builds)
    $runExe    = Join-Path $backendDir "llama-run.exe"         # text CLI (recent builds)
    $cliExe    = Join-Path $backendDir "llama-cli.exe"         # text CLI (older builds)

    $hasServer  = Test-Path $serverExe
    $hasMtmd    = Test-Path $mtmdExe
    $hasRun     = Test-Path $runExe
    $hasCli     = Test-Path $cliExe
    $hasCLIMode = $hasMtmd -or $hasRun -or $hasCli

    Write-Host "`n=== Select Run Mode ===" -ForegroundColor Cyan
    if ($hasServer) {
        Write-Host "  1. Server Mode (HTTP API - for apps and chat UIs)" -ForegroundColor Green
    }
    if ($hasCLIMode) {
        $cliLabel = if ($hasMtmd) { "llama-mtmd-cli" } elseif ($hasRun) { "llama-run" } else { "llama-cli" }
        Write-Host "  2. CLI Mode    (Interactive terminal) [$cliLabel]" -ForegroundColor Yellow
    }

    $modeChoice = Read-Host "`nSelect mode"

    switch ($modeChoice) {
        '1' {
            if ($hasServer) { return @($serverExe, "server") }
            Write-Host "llama-server.exe not found." -ForegroundColor Red
            return Select-RunMode -RunnerPath $RunnerPath
        }
        '2' {
            if ($hasMtmd) { return @($mtmdExe, "mtmd") }
            if ($hasRun)  { return @($runExe,  "run")  }
            if ($hasCli)  { return @($cliExe,  "cli")  }
            Write-Host "No CLI executable found." -ForegroundColor Red
            return Select-RunMode -RunnerPath $RunnerPath
        }
        default {
            Write-Host "Invalid choice." -ForegroundColor Red
            return Select-RunMode -RunnerPath $RunnerPath
        }
    }
}

# =============================================================================
# STEP 6: BUILD ARGUMENTS AND RUN
# =============================================================================
function Start-LlamaApp {
    param(
        [string]$ExecutablePath,
        [string]$ConfigFile,
        [string]$Mode
    )

    Write-Host "`nLoading configuration..." -ForegroundColor Cyan
    $config = Get-Content -Path $ConfigFile | ConvertFrom-Json

    # Migrate old configs: backfill any keys present in the template but missing here
    $defaults   = New-DefaultConfig -ModelName $config.model_name -GgufPath $config.gguf_path
    $dirty      = $false
    $configHash = [ordered]@{}
    foreach ($key in $defaults.Keys) {
        if ($null -eq $config.$key -and $null -ne $defaults[$key]) {
            $configHash[$key] = $defaults[$key]
            $dirty = $true
        } else {
            $configHash[$key] = $config.$key
        }
    }
    if ($dirty) {
        Write-Host "  Old config detected - backfilling new fields..." -ForegroundColor DarkYellow
        $configHash | ConvertTo-Json -Depth 5 | Set-Content -Path $ConfigFile
        Write-ConfigHelp -JsonPath $ConfigFile
        Write-Host "  Config updated: $ConfigFile" -ForegroundColor DarkYellow
    }
    $config = [PSCustomObject]$configHash

    if (-not (Test-Path $config.gguf_path)) {
        Write-Host "Error: GGUF not found at $($config.gguf_path)" -ForegroundColor Red
        Write-Host "Update gguf_path in $ConfigFile and try again." -ForegroundColor Yellow
        exit
    }

    $hasVision = (-not [string]::IsNullOrWhiteSpace($config.mmproj_path)) -and (Test-Path $config.mmproj_path)

    # llama-mtmd-cli requires --mmproj unconditionally — fall back to a text CLI when no mmproj is set
    if ($Mode -eq "mtmd" -and -not $hasVision) {
        $backendDir = Split-Path $ExecutablePath -Parent
        $runExe     = Join-Path $backendDir "llama-run.exe"
        $cliExe     = Join-Path $backendDir "llama-cli.exe"
        if (Test-Path $runExe) {
            Write-Host "  No mmproj configured - falling back to llama-run (text only)." -ForegroundColor DarkYellow
            $ExecutablePath = $runExe
            $Mode = "run"
        } elseif (Test-Path $cliExe) {
            Write-Host "  No mmproj configured - falling back to llama-cli (text only)." -ForegroundColor DarkYellow
            $ExecutablePath = $cliExe
            $Mode = "cli"
        } else {
            Write-Host "  No mmproj configured and no text CLI fallback found. Set mmproj_path in your config to use llama-mtmd-cli." -ForegroundColor Red
            exit
        }
    }

    $argList = [System.Collections.Generic.List[string]]::new()

    # ARM64 Windows builds have a deadlock in the warmup path -- inject --no-warmup automatically.
    # The first real inference request will be marginally slower but the process will not hang.
    $isArm64Backend = $ExecutablePath -match "arm64"
    if ($isArm64Backend) {
        $argList.Add("--no-warmup")
        Write-Host "  ARM64 backend detected - warmup disabled automatically." -ForegroundColor DarkYellow
    }

    # ---- llama-mtmd-cli (vision + text CLI) ----------------------------------
    if ($Mode -eq "mtmd") {
        Add-Arg    $argList "-m"                $config.gguf_path
        if ($hasVision) { Add-Arg $argList "--mmproj" $config.mmproj_path }
        Add-Arg    $argList "-c"                $config.n_ctx
        Add-Arg    $argList "-ngl"              $config.n_gpu_layers
        Add-Arg    $argList "-b"                $config.batch_size
        Add-Arg    $argList "-ub"               $config.ubatch_size
        Add-Arg    $argList "-t"                $config.n_threads
        Add-Arg    $argList "-tb"               $config.n_threads_batch
        Add-Arg    $argList "-fa"               (Get-FlashAttnValue $config.flash_attn)
        Add-Arg    $argList "--cache-type-k"    $config.cache_type_k
        Add-Arg    $argList "--cache-type-v"    $config.cache_type_v
        Add-Arg    $argList "--temp"            $config.temp
        Add-Arg    $argList "--top-k"           $config.top_k
        Add-Arg    $argList "--top-p"           $config.top_p
        Add-Arg    $argList "-s"                $config.seed
        if ($config.use_mmap  -eq $false) { $argList.Add("--no-mmap") }
        if ($config.use_mlock -eq $true)  { $argList.Add("--mlock")   }
        if (-not [string]::IsNullOrWhiteSpace($config.system_prompt)) {
            Add-Arg $argList "-p" $config.system_prompt
        }

        Write-Host "`n========================================================" -ForegroundColor Magenta
        Write-Host " Mode      : INTERACTIVE CLI (llama-mtmd-cli)"             -ForegroundColor Green
        Write-Host " Model     : $($config.model_name)"                        -ForegroundColor Green
        if ($hasVision) {
            Write-Host " Vision    : ENABLED" -ForegroundColor Cyan
            Write-Host " Send image: type /image C:\path\to\photo.jpg"         -ForegroundColor Cyan
            Write-Host "             then type your question on the next line." -ForegroundColor Cyan
        } else {
            Write-Host " Vision    : disabled (set mmproj_path in config)"     -ForegroundColor DarkGray
        }
        Write-Host " Ctrl+C to exit."                                          -ForegroundColor DarkGray
        Write-Host "========================================================`n" -ForegroundColor Magenta

    # ---- llama-run (text CLI, recent builds) ---------------------------------
    } elseif ($Mode -eq "run") {
        Add-Arg    $argList "--model"            $config.gguf_path
        Add-Arg    $argList "--ctx-size"         $config.n_ctx
        Add-Arg    $argList "--gpu-layers"       $config.n_gpu_layers
        Add-Arg    $argList "--threads"          $config.n_threads
        Add-Arg    $argList "--temp"             $config.temp
        Add-Arg    $argList "--seed"             $config.seed

        Write-Host "`n========================================================" -ForegroundColor Magenta
        Write-Host " Mode      : INTERACTIVE CLI (llama-run)"                  -ForegroundColor Green
        Write-Host " Model     : $($config.model_name)"                        -ForegroundColor Green
        Write-Host " Ctrl+C to exit."                                          -ForegroundColor DarkGray
        Write-Host "========================================================`n" -ForegroundColor Magenta

    # ---- llama-server --------------------------------------------------------
    } elseif ($Mode -eq "server") {
        Add-Arg    $argList "-m"                 $config.gguf_path
        if ($hasVision) { Add-Arg $argList "--mmproj" $config.mmproj_path }
        Add-Arg    $argList "-c"                 $config.n_ctx
        Add-Arg    $argList "-ngl"               $config.n_gpu_layers
        Add-Arg    $argList "-b"                 $config.batch_size
        Add-Arg    $argList "-ub"                $config.ubatch_size
        Add-Arg    $argList "-t"                 $config.n_threads
        Add-Arg    $argList "-tb"                $config.n_threads_batch
        Add-Arg    $argList "-fa"               (Get-FlashAttnValue $config.flash_attn)
        Add-Arg    $argList "--cache-type-k"     $config.cache_type_k
        Add-Arg    $argList "--cache-type-v"     $config.cache_type_v
        Add-Arg    $argList "--defrag-thold"     $config.defrag_thold
        Add-Arg    $argList "--split-mode"       $config.split_mode
        Add-Arg    $argList "--main-gpu"         $config.main_gpu
        Add-Arg    $argList "--tensor-split"     $config.tensor_split
        Add-Arg    $argList "--numa"             $config.numa
        Add-Arg    $argList "--rope-freq-base"   $config.rope_freq_base
        Add-Arg    $argList "--rope-freq-scale"  $config.rope_freq_scale
        Add-Arg    $argList "--rope-scaling"     $config.rope_scaling
        Add-Arg    $argList "--yarn-ext-factor"  $config.yarn_ext_factor
        Add-Arg    $argList "--yarn-attn-factor" $config.yarn_attn_factor
        Add-Arg    $argList "--yarn-beta-fast"   $config.yarn_beta_fast
        Add-Arg    $argList "--yarn-beta-slow"   $config.yarn_beta_slow
        Add-Arg    $argList "--yarn-orig-ctx"    $config.yarn_orig_ctx
        Add-Arg    $argList "--grp-attn-n"       $config.grp_attn_n
        Add-Arg    $argList "--grp-attn-w"       $config.grp_attn_w
        Add-Arg    $argList "-n"                 $config.n_predict
        Add-Arg    $argList "-s"                 $config.seed
        Add-Arg    $argList "--temp"             $config.temp
        Add-Arg    $argList "--top-k"            $config.top_k
        Add-Arg    $argList "--top-p"            $config.top_p
        Add-Arg    $argList "--min-p"            $config.min_p
        Add-Arg    $argList "--tfs"              $config.tfs_z
        Add-Arg    $argList "--typical"          $config.typical_p
        Add-Arg    $argList "--repeat-penalty"   $config.repeat_penalty
        Add-Arg    $argList "--repeat-last-n"    $config.repeat_last_n
        Add-Arg    $argList "--presence-penalty" $config.presence_penalty
        Add-Arg    $argList "--frequency-penalty" $config.frequency_penalty
        if ($config.use_mmap  -eq $false) { $argList.Add("--no-mmap") }
        if ($config.use_mlock -eq $true)  { $argList.Add("--mlock")   }
        Add-Arg    $argList "--host"             $config.host
        Add-Arg    $argList "--port"             $config.port
        Add-Arg    $argList "--alias"            $config.model_name
        Add-Arg    $argList "-np"                $config.parallel
        Add-Arg    $argList "--chat-template"    $config.chat_template
        Add-Arg    $argList "--system-prompt"    $config.system_prompt
        Add-Switch $argList "--cont-batching"    $config.cont_batching

        Write-Host "`n========================================================" -ForegroundColor Magenta
        Write-Host " Mode      : SERVER"                                        -ForegroundColor Green
        Write-Host " Model     : $($config.model_name)"                        -ForegroundColor Green
        if ($hasVision) {
            Write-Host " Vision    : ENABLED (mmproj loaded)"                  -ForegroundColor Cyan
        }
        Write-Host " Endpoint  : http://$($config.host):$($config.port)/v1"    -ForegroundColor Yellow
        Write-Host " Ctrl+C to stop."                                          -ForegroundColor DarkGray
        Write-Host "========================================================`n" -ForegroundColor Magenta

    # ---- llama-cli (older builds fallback) -----------------------------------
    } else {
        Add-Arg    $argList "-m"                 $config.gguf_path
        Add-Arg    $argList "-c"                 $config.n_ctx
        Add-Arg    $argList "-ngl"               $config.n_gpu_layers
        Add-Arg    $argList "-b"                 $config.batch_size
        Add-Arg    $argList "-t"                 $config.n_threads
        Add-Arg    $argList "-fa"               (Get-FlashAttnValue $config.flash_attn)
        Add-Arg    $argList "--temp"             $config.temp
        if ($config.use_mmap  -eq $false) { $argList.Add("--no-mmap") }
        if ($config.use_mlock -eq $true)  { $argList.Add("--mlock")   }
        if (-not [string]::IsNullOrWhiteSpace($config.system_prompt)) {
            Add-Arg $argList "-p" $config.system_prompt
        }

        Write-Host "`n========================================================" -ForegroundColor Magenta
        Write-Host " Mode      : INTERACTIVE CLI (llama-cli fallback)"         -ForegroundColor Green
        Write-Host " Model     : $($config.model_name)"                        -ForegroundColor Green
        Write-Host " Ctrl+C to exit."                                          -ForegroundColor DarkGray
        Write-Host "========================================================`n" -ForegroundColor Magenta
    }

    Write-Host "Running: $ExecutablePath $($argList -join ' ')" -ForegroundColor DarkGray
    
    # Set console encoding to UTF-8 to properly display Unicode characters
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    
    & $ExecutablePath @argList
}

# =============================================================================
# MAIN
# =============================================================================
try {
    $RunnerPath     = Setup-Backend
    $SelectedConfig = Select-ModelConfig
    $AppPath, $Mode = Select-RunMode -RunnerPath $RunnerPath
    Start-LlamaApp -ExecutablePath $AppPath -ConfigFile $SelectedConfig -Mode $Mode
} catch {
    Write-Host "`nUnexpected error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
}