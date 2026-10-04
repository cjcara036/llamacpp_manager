#!/usr/bin/env bash

# Exit on severe errors
set -e
set +e # Allow command failures gracefully where handled

SCRIPT_VERSION="v4.1.0-FIX"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_INSTALL_DIR="$SCRIPT_DIR/llamacpp_bin"
MODELS_DIR="$SCRIPT_DIR/Models"
CONFIGS_DIR="$SCRIPT_DIR/Configs"

mkdir -p "$MODELS_DIR" "$CONFIGS_DIR"

CURRENT_EXE_PATH=""
unset SUPPORTED_FLAGS_MAP FLAG_HELP_MAP FLAG_TYPE_MAP
declare -gA SUPPORTED_FLAGS_MAP
declare -gA FLAG_HELP_MAP
declare -gA FLAG_TYPE_MAP

# =============================================================================
# DEPENDENCY AUTO-INSTALLER
# =============================================================================
install_dependencies() {
    local missing_deps=()
    for cmd in curl jq tar unzip grep sed awk; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if ! command -v lspci &> /dev/null; then
        missing_deps+=("pciutils")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "\033[33mMissing required dependencies: ${missing_deps[*]}. Attempting auto-install...\033[0m" >&2
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y "${missing_deps[@]}" >&2
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y "${missing_deps[@]}" >&2
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm "${missing_deps[@]}" >&2
        elif command -v zypper &> /dev/null; then
            sudo zypper install -y "${missing_deps[@]}" >&2
        else
            echo -e "\033[31mError: Unsupported package manager. Please install manually: ${missing_deps[*]}\033[0m" >&2
            exit 1
        fi
        echo -e "\033[32mDependencies installed successfully.\033[0m" >&2
    fi
}

install_dependencies

# =============================================================================
# DYNAMIC BINARY HELP PARSER & FULL AUTO-DISCOVERY
# =============================================================================
extract_supported_flags() {
    local exe_path="$1"
    if [[ -z "$exe_path" || ! -x "$exe_path" ]]; then return 0; fi

    unset SUPPORTED_FLAGS_MAP FLAG_HELP_MAP FLAG_TYPE_MAP
    declare -gA SUPPORTED_FLAGS_MAP
    declare -gA FLAG_HELP_MAP
    declare -gA FLAG_TYPE_MAP
    
    echo -e "  \033[90mExtracting complete flag schema from $(basename "$exe_path") --help...\033[0m" >&2
    
    local help_out
    help_out=$("$exe_path" --help 2>&1 || true)

    while IFS= read -r line; do
        # Extract all long flags (--flag-name)
        local long_flags
        long_flags=$(echo "$line" | grep -oE '\-\-[a-zA-Z0-9_-]+' || true)
        
        if [[ -n "$long_flags" ]]; then
            local clean_line
            clean_line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
            
            for f in $long_flags; do
                SUPPORTED_FLAGS_MAP["$f"]=1
                if [[ -z "${FLAG_HELP_MAP[$f]}" ]]; then
                    FLAG_HELP_MAP["$f"]="$clean_line"
                fi
                
                if echo "$line" | grep -qE '\-\-[a-zA-Z0-9_-]+[[:space:]]+N|\-\-[a-zA-Z0-9_-]+[[:space:]]+[A-Z_]+'; then
                    FLAG_TYPE_MAP["$f"]="value"
                elif echo "$line" | grep -qE 'on|off|auto|true|false'; then
                    FLAG_TYPE_MAP["$f"]="enum"
                else
                    FLAG_TYPE_MAP["$f"]="bool"
                fi
            done
        fi
        
        # Extract short flags (-c, -m, -ngl, etc.)
        local short_flags
        short_flags=$(echo "$line" | grep -oE '^[[:space:]]*\-[a-zA-Z0-9]\b' || true)
        for sf in $short_flags; do
            sf=$(echo "$sf" | xargs)
            if [[ -n "$sf" ]]; then
                SUPPORTED_FLAGS_MAP["$sf"]=1
            fi
        done
    done <<< "$help_out"

    echo -e "  \033[90mAuto-discovered ${#SUPPORTED_FLAGS_MAP[@]} supported binary options.\033[0m" >&2
}

is_flag_in_binary() {
    local flag="$1"
    [[ -n "${SUPPORTED_FLAGS_MAP[$flag]}" ]]
}

resolve_best_flag() {
    local key="$1"
    local mode="$2"
    
    case "$key" in
        mmproj_path)
            if is_flag_in_binary "--mmproj"; then echo "--mmproj"; return; fi
            if is_flag_in_binary "--mmproj-path"; then echo "--mmproj-path"; return; fi
            ;;
        gguf_path)
            if [[ "$mode" == "run" ]] && is_flag_in_binary "--model"; then echo "--model"; return; fi
            if is_flag_in_binary "-m"; then echo "-m"; return; fi
            if is_flag_in_binary "--model"; then echo "--model"; return; fi
            ;;
        n_ctx)
            if is_flag_in_binary "-c"; then echo "-c"; return; fi
            if is_flag_in_binary "--ctx-size"; then echo "--ctx-size"; return; fi
            ;;
        n_gpu_layers)
            if is_flag_in_binary "-ngl"; then echo "-ngl"; return; fi
            if is_flag_in_binary "--n-gpu-layers"; then echo "--n-gpu-layers"; return; fi
            ;;
        batch_size)
            if is_flag_in_binary "-b"; then echo "-b"; return; fi
            if is_flag_in_binary "--batch-size"; then echo "--batch-size"; return; fi
            ;;
        ubatch_size)
            if is_flag_in_binary "-ub"; then echo "-ub"; return; fi
            ;;
        n_threads)
            if is_flag_in_binary "-t"; then echo "-t"; return; fi
            ;;
        flash_attn)
            if is_flag_in_binary "-fa"; then echo "-fa"; return; fi
            if is_flag_in_binary "--flash-attn"; then echo "--flash-attn"; return; fi
            ;;
        parallel)
            if is_flag_in_binary "-np"; then echo "-np"; return; fi
            ;;
        perf)
            if is_flag_in_binary "--perf"; then echo "--perf"; return; fi
            ;;
    esac

    # Universal Fallback: Convert snake_case key to --kebab-case flag
    local kebab_flag="--${key//_/-}"
    if is_flag_in_binary "$kebab_flag"; then
        echo "$kebab_flag"
        return
    fi

    local direct_flag="--$key"
    if is_flag_in_binary "$direct_flag"; then
        echo "$direct_flag"
        return
    fi

    echo ""
}

sync_config_with_binary() {
    local config_file="$1"
    local exe_path="$2"

    if [[ ! -f "$config_file" ]]; then return 0; fi
    if [[ -z "$exe_path" || ! -x "$exe_path" ]]; then
        exe_path=$(find "$BASE_INSTALL_DIR" -type f -name "llama-server" 2>/dev/null | head -n 1)
    fi

    if [[ -n "$exe_path" && ${#SUPPORTED_FLAGS_MAP[@]} -eq 0 ]]; then
        extract_supported_flags "$exe_path"
    fi

    if [[ ${#SUPPORTED_FLAGS_MAP[@]} -eq 0 ]]; then return 0; fi

    local tmp_json=$(mktemp)
    cp "$config_file" "$tmp_json"

    local added_count=0
    for flag in "${!SUPPORTED_FLAGS_MAP[@]}"; do
        if [[ "$flag" =~ ^\-\-[a-zA-Z0-9_-]+$ ]]; then
            local clean_key="${flag#--}"
            clean_key="${clean_key//-/_}"
            
            if [[ "$clean_key" =~ ^no_ ]]; then continue; fi

            local exists
            exists=$(jq --arg k "$clean_key" 'has($k)' "$tmp_json")
            if [[ "$exists" == "false" ]]; then
                jq --arg k "$clean_key" '.[$k] = null' "$tmp_json" > "${tmp_json}.tmp" && mv "${tmp_json}.tmp" "$tmp_json"
                ((added_count++)) || true
            fi
        fi
    done

    if (( added_count > 0 )); then
        echo -e "  \033[32mAuto-discovered and added $added_count new settings to $(basename "$config_file")\033[0m" >&2
        mv "$tmp_json" "$config_file"
    else
        rm -f "$tmp_json"
    fi
}

get_field_value_hint() {
    local key="$1"
    local help_line="$2"
    local hint=""

    if [[ "$help_line" =~ ([a-zA-Z0-9_]+(,[[:space:]]*[a-zA-Z0-9_]+)+) ]]; then
        hint="Allowed Options: [ ${BASH_REMATCH[1]} | null ]"
    fi

    if [[ -z "$hint" ]]; then
        case "$key" in
            preserve_thinking|use_mmap|use_mlock|cont_batching|perf)
                hint="Boolean: [ true | false | null ]"
                ;;
            flash_attn)
                hint="FlashAttention mode: [ on | off | auto | null ]"
                ;;
            spec_type)
                hint="Speculative mode: [ draft-mtp | draft-simple | draft-eagle3 | draft-dflash | null ]"
                ;;
            host)
                hint="IP Address: [ 127.0.0.1 (Local) | 0.0.0.0 (LAN Access) ]"
                ;;
            *)
                hint="Auto-Detected: String, Number, Boolean, or null"
                ;;
        esac
    fi

    echo "$hint"
}

declare -a ARG_LIST

build_dynamic_args() {
    local config_file="$1"
    local mode="$2"
    ARG_LIST=()

    while IFS="=" read -r key val; do
        val="${val%$'\r'}"
        if [[ -z "$val" || "$val" == "null" ]]; then
            continue
        fi

        if [[ "$key" == "model_name" ]]; then
            continue
        fi

        if [[ "$key" == "mmproj_path" ]]; then
            if [[ ! -f "$val" ]]; then continue; fi
        fi

        local flag
        flag=$(resolve_best_flag "$key" "$mode")

        if [[ -n "$flag" ]]; then
            if [[ "$val" == "true" ]]; then
                ARG_LIST+=("$flag")
            elif [[ "$val" == "false" ]]; then
                local no_flag="--no-${key//_/-}"
                if is_flag_in_binary "$no_flag"; then
                    ARG_LIST+=("$no_flag")
                fi
            else
                if [[ "$key" == "flash_attn" ]]; then
                    if [[ "$val" == "on" || "$val" == "off" || "$val" == "auto" ]]; then
                        ARG_LIST+=("$flag" "$val")
                    fi
                else
                    ARG_LIST+=("$flag" "$val")
                fi
            fi
        else
            echo -e "  \033[90mNotice: Key '$key' not supported by this binary build. Omitted.\033[0m" >&2
        fi
    done < <(jq -r 'to_entries | .[] | .key + "=" + (.value | tostring)' "$config_file")
}

new_default_config() {
    local model_name="$1"
    local gguf_path="$2"
    local mmproj_path="${3:-null}"
    
    [[ "$mmproj_path" != "null" ]] && mmproj_path="\"$mmproj_path\""

    cat <<EOF
{
  "model_name": "$model_name",
  "gguf_path": "$gguf_path",
  "mmproj_path": $mmproj_path,
  "n_ctx": 8192,
  "batch_size": 512,
  "n_gpu_layers": 33,
  "flash_attn": "auto",
  "perf": true,
  "host": "127.0.0.1",
  "port": 8080,
  "cont_batching": true
}
EOF
}

# =============================================================================
# HARDWARE DETECTION & BACKEND SETUP
# =============================================================================
get_hardware_recommendation() {
    echo -e "\n\033[36m========================================================\033[0m" >&2
    echo -e " \033[32mllama.cpp Manager $SCRIPT_VERSION\033[0m" >&2
    echo -e "\033[36m========================================================\033[0m" >&2
    echo -e "Scanning hardware..." >&2

    local arch=$(uname -m)
    echo -e "  \033[90mArchitecture : $arch\033[0m" >&2

    local gpus=""
    if command -v lspci &> /dev/null; then
        gpus=$(lspci | grep -iE 'vga|3d|display' || true)
    fi

    local gpu_names="Unknown"
    [[ -n "$gpus" ]] && gpu_names=$(echo "$gpus" | head -n 1 | awk -F': ' '{print $2}')
    echo -e "  \033[90mGPU(s)       : $gpu_names\033[0m" >&2

    local recommended_choice="3"
    local recommended_text="CPU / AVX2 (No GPU / Basic Fallback)"

    if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
        recommended_choice="4"
        recommended_text="ARM64 CPU (Native ARM)"
    else
        if echo "$gpus" | grep -iq "nvidia"; then
            recommended_choice="1"
            recommended_text="CUDA (NVIDIA GPUs - Fastest)"
        elif echo "$gpus" | grep -iqE "amd|radeon"; then
            recommended_choice="5"
            recommended_text="ROCm (AMD GPUs - Native Accelerated)"
        elif echo "$gpus" | grep -iq "intel"; then
            recommended_choice="2"
            recommended_text="Vulkan (Intel/AMD GPUs - Good compatibility)"
        fi
    fi

    echo -e "  \033[32mRecommended  : Option $recommended_choice - $recommended_text\033[0m" >&2
    echo "$recommended_choice"
}

parse_asset_list() {
    local list="$1"
    local choice="$2"

    case "$choice" in
        1) echo "$list" | grep -iE '/llama-b[0-9]+-bin-ubuntu-cuda.*x64\.(tar\.gz|zip)$' | grep -v 'cudart' | head -n 1 ;;
        2) echo "$list" | grep -iE '/llama-b[0-9]+-bin-ubuntu-vulkan.*x64\.(tar\.gz|zip)$' | head -n 1 ;;
        3) echo "$list" | grep -iE '/llama-b[0-9]+-bin-ubuntu-x64\.(tar\.gz|zip)$' | head -n 1 ;;
        4) echo "$list" | grep -iE '/llama-b[0-9]+-bin-ubuntu-.*(arm64|aarch64).*\.(tar\.gz|zip)$' | grep -v 'cuda' | grep -v 'vulkan' | head -n 1 ;;
        5) echo "$list" | grep -iE '/llama-b[0-9]+-bin-ubuntu-rocm.*x64\.(tar\.gz|zip)$' | head -n 1 ;;
    esac
}

fetch_release_download_url() {
    local choice="$1"
    local download_url=""

    local api_url="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
    local release_json
    release_json=$(curl -s -L -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64)" "$api_url" 2>/dev/null || true)

    if [[ -n "$release_json" ]]; then
        local all_urls
        all_urls=$(echo "$release_json" | jq -r '.[].assets[]?.browser_download_url // empty' 2>/dev/null || true)
        if [[ -n "$all_urls" ]]; then
            download_url=$(parse_asset_list "$all_urls" "$choice")
        fi
    fi

    if [[ -z "$download_url" ]]; then
        echo -e "  \033[33mScraping release page directly for asset match...\033[0m" >&2
        local html_data
        html_data=$(curl -s -L -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64)" "https://github.com/ggml-org/llama.cpp/releases" 2>/dev/null || true)

        local extracted_paths
        extracted_paths=$(echo "$html_data" | grep -oE '/ggml-org/llama\.cpp/releases/download/[^"]+' || true)
        
        if [[ -n "$extracted_paths" ]]; then
            local relative_path
            relative_path=$(parse_asset_list "$extracted_paths" "$choice")
            if [[ -n "$relative_path" ]]; then
                download_url="https://github.com$relative_path"
            fi
        fi
    fi

    echo "$download_url"
}

setup_backend() {
    local recommended=$(get_hardware_recommendation)

    echo -e "\nWhich backend do you want to run?" >&2
    echo "  --- Standard x64 (Intel/AMD) ---" >&2
    echo "  1. CUDA   (NVIDIA GPUs)" >&2
    echo "  2. Vulkan (Intel/AMD GPUs)" >&2
    echo "  3. CPU    (includes AVX2 support)" >&2
    echo "  --- Linux on ARM ---" >&2
    echo "  4. ARM64  (Native ARM execution)" >&2
    echo "  --- Dedicated AMD ---" >&2
    echo "  5. ROCm   (Native AMD execution)" >&2

    read -p "$(echo -e "\nEnter 1-5 (Enter = recommended: $recommended): ")" choice
    if [[ -z "$choice" ]]; then choice="$recommended"; fi

    local sub_folder
    case "$choice" in
        1) sub_folder="cuda" ;;
        2) sub_folder="vulkan" ;;
        3) sub_folder="cpu" ;;
        4) sub_folder="arm64" ;;
        5) sub_folder="rocm" ;;
        *) echo -e "\033[31mInvalid choice. Exiting...\033[0m" >&2; exit 1 ;;
    esac

    local backend_path="$BASE_INSTALL_DIR/$sub_folder"
    local server_exe=$(find "$backend_path" -type f -name "llama-server" 2>/dev/null | head -n 1)

    if [[ -n "$server_exe" ]]; then
        read -p "$(echo -e "\nFound existing '$sub_folder' backend. Pull latest update? (y/N): ")" update
        if [[ "$update" =~ ^[Yy]$ ]]; then
            echo -e "\033[33mRemoving old version...\033[0m" >&2
            rm -rf "$backend_path"
            server_exe=""
        else
            extract_supported_flags "$server_exe"
            echo "$server_exe"
            return
        fi
    fi

    echo -e "\n\033[33mFetching latest release binary for '$sub_folder'...\033[0m" >&2
    
    local download_url=$(fetch_release_download_url "$choice")

    if [[ -z "$download_url" || "$download_url" == "null" ]]; then
        echo -e "\033[31mError: Could not locate a release asset matching choice $choice\033[0m" >&2
        exit 1
    fi

    local asset_name=$(basename "$download_url")
    local archive_path="$SCRIPT_DIR/$asset_name"

    echo -e "\033[33mDownloading $asset_name...\033[0m" >&2
    if [[ -f "$archive_path" && -s "$archive_path" ]]; then
        local file_size=$(stat -c%s "$archive_path")
        local mb_size=$(( file_size / 1048576 ))
        echo -e "  \033[33mFound partial download (${mb_size} MB). Resuming...\033[0m" >&2
    fi

    curl -f -L -C - --retry 3 --retry-delay 5 -o "$archive_path" "$download_url" >&2 || {
        echo -e "\033[31mDownload failed. Removing partial archive...\033[0m" >&2
        rm -f "$archive_path"
        exit 1
    }

    echo -e "\033[33mExtracting to $backend_path...\033[0m" >&2
    mkdir -p "$backend_path"
    
    if [[ "$asset_name" == *.tar.gz ]]; then
        tar -xzf "$archive_path" -C "$backend_path" >&2
    elif [[ "$asset_name" == *.zip ]]; then
        unzip -q -o "$archive_path" -d "$backend_path" >&2
    else
        echo -e "\033[31mUnsupported archive format: $asset_name\033[0m" >&2
        rm -f "$archive_path"
        exit 1
    fi
    
    find "$backend_path" -type f -name "llama-*" -exec chmod +x {} \; >&2 2>/dev/null
    rm -f "$archive_path"

    server_exe=$(find "$backend_path" -type f -name "llama-server" 2>/dev/null | head -n 1)
    extract_supported_flags "$server_exe"
    echo "$server_exe"
}

# =============================================================================
# HUGGING FACE DOWNLOADER
# =============================================================================
invoke_hf_download() {
    local remote_path="$1"
    local local_path="$2"
    local token="$3"
    
    local url="https://huggingface.co/$repo_id/resolve/main/$remote_path?download=true"

    if [[ -f "$local_path" && -s "$local_path" ]]; then
        local file_size=$(stat -c%s "$local_path")
        local mb_size=$(( file_size / 1048576 ))
        echo -e "  \033[33mFound partial file (${mb_size} MB). Resuming...\033[0m" >&2
    fi

    local curl_args=("-L" "-C" "-" "--retry" "3" "--retry-delay" "5" \
                     "--user-agent" "Mozilla/5.0 (X11; Linux x86_64)")
    if [[ -n "$token" ]]; then
        curl_args+=("-H" "Authorization: Bearer $token")
    fi
    curl_args+=("-o" "$local_path" "$url")
    curl "${curl_args[@]}" >&2

    local info_size=$(stat -c%s "$local_path" 2>/dev/null || echo 0)
    if (( info_size >= 1000000 )); then return 0; else return 1; fi
}

download_huggingface_model() {
    echo -e "\n\033[36m=== Hugging Face GGUF Downloader ===\033[0m" >&2
    read -p "Enter Repo ID or URL (e.g. 'unsloth/Qwen3.8-27B-GGUF'): " repo_id
    repo_id=$(echo "$repo_id" | xargs)
    
    repo_id=$(echo "$repo_id" | sed -E 's|https?://huggingface\.co/||g; s|/*$||')

    if [[ -z "$repo_id" ]]; then echo "" >&2; return; fi

    echo -e "\n\033[33mQuerying Hugging Face API for '$repo_id'...\033[0m" >&2
    
    local repo_data=$(curl -s -L -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64)" \
                      "https://huggingface.co/api/models/$repo_id")
    
    local all_ggufs=$(echo "$repo_data" | jq -r '.siblings[]?.rfilename // empty' \
                      | grep -iE '\.gguf' || true)

    readarray -t model_files  < <(echo "$all_ggufs" | grep -v "mmproj" || true)
    readarray -t mmproj_files < <(echo "$all_ggufs" | grep "mmproj"    || true)

    if [[ ${#model_files[@]} -eq 0 || -z "${model_files[0]}" ]]; then
        echo -e "\033[31mNo .gguf model files found. Check the repo name.\033[0m" >&2
        echo "" >&2; return
    fi

    local repo_folder_name=$(basename "$repo_id" | sed 's/[\\/:*?"<>|]/_/g')
    local model_subdir="$MODELS_DIR/$repo_folder_name"
    
    if [[ ! -d "$model_subdir" ]]; then
        mkdir -p "$model_subdir"
        echo -e "  \033[90mCreated model folder: $repo_folder_name\033[0m" >&2
    else
        echo -e "  \033[90mUsing existing model folder: $repo_folder_name\033[0m" >&2
    fi

    echo -e "\nAvailable model files:" >&2
    for i in "${!model_files[@]}"; do
        echo "  $((i + 1)). ${model_files[$i]}" >&2
    done
    
    read -p "Select a file (1-${#model_files[@]}): " file_choice
    local selected_file="${model_files[$((file_choice - 1))]}"
    local save_file_name=$(basename "$selected_file")
    local save_path="$model_subdir/$save_file_name"

    local mmproj_save_path="null" selected_mmproj="" mmproj_save_file_name=""

    if [[ ${#mmproj_files[@]} -gt 0 && -n "${mmproj_files[0]}" ]]; then
        echo -e "\n\033[36mVision projector file(s) detected in this repo:\033[0m" >&2
        for i in "${!mmproj_files[@]}"; do
            echo "  $((i + 1)). ${mmproj_files[$i]}" >&2
        done
        read -p "Download a projector for vision support? Enter number or Enter to skip: " mmproj_choice
        if [[ -n "$mmproj_choice" ]]; then
            selected_mmproj="${mmproj_files[$((mmproj_choice - 1))]}"
            mmproj_save_file_name=$(basename "$selected_mmproj")
            mmproj_save_path="$model_subdir/$mmproj_save_file_name"
            echo -e "  \033[32mWill also download: $mmproj_save_file_name\033[0m" >&2
        fi
    fi

    local hf_token="" downloaded=false

    for attempt in 1 2; do
        if [[ $attempt -eq 2 ]]; then
            echo -e "\n\033[33mDownload failed. Repo may require a Hugging Face token.\033[0m" >&2
            read -p "HF_TOKEN (Enter to cancel): " hf_token
            hf_token=$(echo "$hf_token" | xargs)
            if [[ -z "$hf_token" ]]; then echo -e "\033[31mCancelled.\033[0m" >&2; return; fi
            echo -e "\033[33mRetrying with token...\033[0m" >&2
        fi
        echo -e "\n\033[32mDownloading $save_file_name...\033[0m" >&2
        if invoke_hf_download "$selected_file" "$save_path" "$hf_token"; then
            downloaded=true; break
        else
            [[ -f "$save_path" ]] && rm -f "$save_path"
        fi
    done

    if [[ "$downloaded" == false ]]; then
        echo -e "\033[31mDownload failed. Check your HF_TOKEN or try a different repo.\033[0m" >&2
        return
    fi

    if [[ "$mmproj_save_path" != "null" ]]; then
        echo -e "\n\033[32mDownloading $mmproj_save_file_name...\033[0m" >&2
        if invoke_hf_download "$selected_mmproj" "$mmproj_save_path" "$hf_token"; then
            echo -e "\033[32mProjector downloaded successfully.\033[0m" >&2
        else
            echo -e "\033[33mmmproj download failed. Vision will be disabled in the config.\033[0m" >&2
            mmproj_save_path="null"
        fi
    fi

    local model_name_str="${save_file_name%.gguf}"
    model_name_str="${model_name_str,,}"
    local json_path="$CONFIGS_DIR/$model_name_str.json"
    
    new_default_config "$model_name_str" "$save_path" "$mmproj_save_path" | jq . > "$json_path"
    sync_config_with_binary "$json_path" "$CURRENT_EXE_PATH"

    echo -e "\n\033[32mConfig written to : $json_path\033[0m" >&2
    if [[ "$mmproj_save_path" != "null" ]]; then
        echo -e "\033[36mVision enabled.   : mmproj_path set in config.\033[0m" >&2
    fi
    
    read -p "Would you like to edit settings for this new model now? (y/N): " edit_now
    if [[ "$edit_now" =~ ^[Yy]$ ]]; then
        edit_model_config "$json_path"
    fi

    echo "$json_path"
}

# =============================================================================
# DYNAMIC INTERACTIVE SETTINGS EDITOR PER MODEL
# =============================================================================
edit_model_config() {
    local json_path="$1"
    
    sync_config_with_binary "$json_path" "$CURRENT_EXE_PATH"

    while true; do
        echo -e "\n\033[36m=== Configuration Editor ($SCRIPT_VERSION): $(basename "$json_path") ===\033[0m" >&2
        
        readarray -t keys < <(jq -r 'keys[]' "$json_path")

        if [[ ${#keys[@]} -eq 0 ]]; then
            echo -e "\033[31mError: Empty or invalid JSON configuration file.\033[0m" >&2
            return 1
        fi

        for i in "${!keys[@]}"; do
            local key="${keys[$i]}"
            local val=$(jq -r --arg k "$key" '.[$k] | if . == null then "null" else tostring end' "$json_path")
            printf "  %2d. %-24s : %s\n" "$((i + 1))" "$key" "$val" >&2
        done
        echo "   0. [ Save and Exit Editor ]" >&2

        read -p "$(echo -e "\nSelect setting number to edit (0-${#keys[@]}): ")" choice

        if [[ "$choice" == "0" || -z "$choice" ]]; then
            echo -e "\033[32mConfiguration saved.\033[0m" >&2
            break
        fi

        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#keys[@]} )); then
            local selected_key="${keys[$((choice - 1))]}"
            local current_val=$(jq -r --arg k "$selected_key" '.[$k] | if . == null then "null" else tostring end' "$json_path")

            local matched_flag
            matched_flag=$(resolve_best_flag "$selected_key" "server")

            local help_info=""
            if [[ -n "$matched_flag" ]]; then
                help_info="${FLAG_HELP_MAP[$matched_flag]}"
            fi

            local value_hint
            value_hint=$(get_field_value_hint "$selected_key" "$help_info")

            echo -e "\n\033[36m-------------------------------------------------------------------------------\033[0m" >&2
            echo -e " Editing Setting : \033[33m$selected_key\033[0m (Current: \033[36m$current_val\033[0m)" >&2
            if [[ -n "$matched_flag" ]]; then
                echo -e " Matched CLI Flag: \033[32m$matched_flag\033[0m" >&2
            fi
            if [[ -n "$help_info" ]]; then
                echo -e " Binary Help     : \033[90m$help_info\033[0m" >&2
            fi
            echo -e " Valid Values    : \033[35m$value_hint\033[0m" >&2
            echo -e "\033[36m-------------------------------------------------------------------------------\033[0m" >&2

            read -p "Enter new value: " new_val

            if [[ "$new_val" == "null" || -z "$new_val" ]]; then
                jq --arg k "$selected_key" '.[$k] = null' "$json_path" > "$json_path.tmp" && mv "$json_path.tmp" "$json_path"
            elif [[ "$new_val" =~ ^(true|false)$ || "$new_val" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
                jq --arg k "$selected_key" --argjson v "$new_val" '.[$k] = $v' "$json_path" > "$json_path.tmp" && mv "$json_path.tmp" "$json_path"
            else
                jq --arg k "$selected_key" --arg v "$new_val" '.[$k] = $v' "$json_path" > "$json_path.tmp" && mv "$json_path.tmp" "$json_path"
            fi
            echo -e "\033[32mUpdated '$selected_key' successfully.\033[0m" >&2
        else
            echo -e "\033[31mInvalid selection.\033[0m" >&2
        fi
    done
}

# =============================================================================
# DELETE MODEL & CONFIG
# =============================================================================
delete_model_config() {
    echo -e "\n\033[31m=== Delete Model & Configuration ===\033[0m" >&2
    readarray -t configs < <(find "$CONFIGS_DIR" -maxdepth 1 -name "*.json" | sort)

    if [[ ${#configs[@]} -eq 0 ]]; then
        echo -e "\033[33mNo configurations found to delete.\033[0m" >&2
        return
    fi

    echo "Select a configuration to delete:" >&2
    for i in "${!configs[@]}"; do
        echo "  $((i + 1)). $(basename "${configs[$i]}")" >&2
    done
    echo "  0. [ Cancel ]" >&2

    read -p "Choice (0-${#configs[@]}): " del_choice
    if [[ "$del_choice" == "0" || -z "$del_choice" ]]; then return; fi

    local idx=$((del_choice - 1))
    if [[ $idx -lt 0 || $idx -ge ${#configs[@]} ]]; then
        echo -e "\033[31mInvalid choice.\033[0m" >&2
        return
    fi

    local json_path="${configs[$idx]}"
    local help_path="${json_path%.json}.help.txt"
    local gguf_path=$(jq -r '.gguf_path // ""' "$json_path")
    local mmproj_path=$(jq -r '.mmproj_path // "null"' "$json_path")

    echo -e "\n\033[33mThe following files will be PERMANENTLY DELETED:\033[0m" >&2
    echo -e "  - Config File  : $json_path" >&2
    [[ -f "$help_path" ]] && echo -e "  - Help Guide   : $help_path" >&2
    if [[ -n "$gguf_path" && -f "$gguf_path" ]]; then
        echo -e "  - Model GGUF   : $gguf_path" >&2
    fi
    if [[ "$mmproj_path" != "null" && -n "$mmproj_path" && -f "$mmproj_path" ]]; then
        echo -e "  - Vision mmproj: $mmproj_path" >&2
    fi

    read -p "$(echo -e "\n\033[31mAre you sure you want to delete these files? (y/N): \033[0m")" confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -f "$json_path" "$help_path"
        [[ -n "$gguf_path" && -f "$gguf_path" ]] && rm -f "$gguf_path"
        [[ "$mmproj_path" != "null" && -n "$mmproj_path" && -f "$mmproj_path" ]] && rm -f "$mmproj_path"

        if [[ -n "$gguf_path" ]]; then
            local model_dir=$(dirname "$gguf_path")
            if [[ -d "$model_dir" && "$model_dir" != "$MODELS_DIR" ]]; then
                rmdir "$model_dir" 2>/dev/null || true
            fi
        fi

        echo -e "\033[32mModel and configuration deleted successfully.\033[0m" >&2
    else
        echo -e "\033[33mDeletion cancelled.\033[0m" >&2
    fi
}

# =============================================================================
# MODEL SELECTION & MENU
# =============================================================================
select_model_config() {
    while true; do
        echo -e "\n\033[36m=== Model Selection & Settings ($SCRIPT_VERSION) ===\033[0m" >&2
        readarray -t configs < <(find "$CONFIGS_DIR" -maxdepth 1 -name "*.json" | sort)
        echo -e "  \033[32m0. [ Download New Model from Hugging Face ]\033[0m" >&2
        for i in "${!configs[@]}"; do
            echo "  $((i + 1)). Run $(basename "${configs[$i]}")" >&2
        done
        if [[ ${#configs[@]} -gt 0 ]]; then
            echo -e "  \033[33me. [ Edit Settings for an Existing Config ]\033[0m" >&2
            echo -e "  \033[31md. [ Delete Model and its Config ]\033[0m" >&2
        fi

        read -p "$(echo -e "\nSelect an option: ")" choice

        if [[ "$choice" == "0" ]]; then
            local new_config_path=$(download_huggingface_model)
            if [[ -n "$new_config_path" ]]; then echo "$new_config_path"; return; fi
        elif [[ "$choice" =~ ^[Ee]$ && ${#configs[@]} -gt 0 ]]; then
            echo -e "\nSelect config to edit:" >&2
            for i in "${!configs[@]}"; do
                echo "  $((i + 1)). $(basename "${configs[$i]}")" >&2
            done
            read -p "Choice (1-${#configs[@]}): " e_choice
            local e_idx=$((e_choice - 1))
            if [[ $e_idx -ge 0 && $e_idx -lt ${#configs[@]} ]]; then
                edit_model_config "${configs[$e_idx]}"
            fi
        elif [[ "$choice" =~ ^[Dd]$ && ${#configs[@]} -gt 0 ]]; then
            delete_model_config
        else
            local index=$((choice - 1))
            if [[ $index -ge 0 && $index -lt ${#configs[@]} ]]; then
                echo "${configs[$index]}"; return
            else
                echo -e "\033[31mInvalid choice.\033[0m" >&2
            fi
        fi
    done
}

# =============================================================================
# SELECT RUN MODE & SWEEP ENGINE
# =============================================================================
select_run_mode() {
    local runner_path="$1"
    local backend_dir=$(dirname "$runner_path")

    local server_exe="$backend_dir/llama-server"
    local mtmd_exe="$backend_dir/llama-mtmd-cli"
    local run_exe="$backend_dir/llama-run"
    local cli_exe="$backend_dir/llama-cli"

    local has_server=false; [[ -f "$server_exe" ]] && has_server=true
    local has_mtmd=false;   [[ -f "$mtmd_exe"   ]] && has_mtmd=true
    local has_run=false;    [[ -f "$run_exe"     ]] && has_run=true
    local has_cli=false;    [[ -f "$cli_exe"     ]] && has_cli=true

    local has_cli_mode=false
    if [[ "$has_mtmd" == true || "$has_run" == true || "$has_cli" == true ]]; then
        has_cli_mode=true
    fi

    echo -e "\n\033[36m=== Select Run Mode ($SCRIPT_VERSION) ===\033[0m" >&2
    echo -e "  \033[35m0. SWEEP Mode  (Benchmark & recommend optimal config)\033[0m" >&2
    if [[ "$has_server" == true ]]; then
        echo -e "  \033[32m1. Server Mode (HTTP API - for apps and chat UIs)\033[0m" >&2
    fi
    if [[ "$has_cli_mode" == true ]]; then
        local cli_label=""
        if [[ "$has_mtmd" == true ]];  then cli_label="llama-mtmd-cli"
        elif [[ "$has_run" == true ]]; then cli_label="llama-run"
        else cli_label="llama-cli"
        fi
        echo -e "  \033[33m2. CLI Mode    (Interactive terminal) [$cli_label]\033[0m" >&2
    fi

    read -p "$(echo -e "\nSelect mode: ")" mode_choice

    case "$mode_choice" in
        0)
            if [[ "$has_server" == true ]]; then echo "$server_exe sweep"; return; fi
            echo -e "\033[31mllama-server not found. SWEEP requires the server binary.\033[0m" >&2
            select_run_mode "$runner_path"
            ;;
        1)
            if [[ "$has_server" == true ]]; then echo "$server_exe server"; return; fi
            echo -e "\033[31mllama-server not found.\033[0m" >&2
            select_run_mode "$runner_path"
            ;;
        2)
            if [[ "$has_mtmd" == true ]]; then echo "$mtmd_exe mtmd"; return; fi
            if [[ "$has_run"  == true ]]; then echo "$run_exe run";   return; fi
            if [[ "$has_cli"  == true ]]; then echo "$cli_exe cli";   return; fi
            echo -e "\033[31mNo CLI executable found.\033[0m" >&2
            select_run_mode "$runner_path"
            ;;
        *)
            echo -e "\033[31mInvalid choice.\033[0m" >&2
            select_run_mode "$runner_path"
            ;;
    esac
}

# ---------- SWEEP MODE HELPER FUNCTIONS ----------
_SW_SERVER_PID=""
_SW_SPINNER_PID=""
_SW_TPS_FILE=""
_SW_GGUF=""
_SW_NCTX=""
_SW_NGL=""
declare -a _SW_RESULTS=()

_sw_spinner_start() {
    local msg="$1"
    (
        local frames=('/' '-' '\' '|')
        local i=0
        while true; do
            printf "\r  \033[36m%s\033[0m %s  " "${frames[$i]}" "$msg" >&2
            i=$(( (i+1) % 4 ))
            sleep 0.15
        done
    ) &
    _SW_SPINNER_PID=$!
    disown "$_SW_SPINNER_PID" 2>/dev/null || true
}

_sw_spinner_stop() {
    if [[ -n "$_SW_SPINNER_PID" ]]; then
        kill "$_SW_SPINNER_PID" 2>/dev/null || true
        wait "$_SW_SPINNER_PID" 2>/dev/null || true
        _SW_SPINNER_PID=""
        printf "\r\033[2K" >&2
    fi
}

_sw_kill_server() {
    if [[ -n "$_SW_SERVER_PID" ]]; then
        if kill -0 "$_SW_SERVER_PID" 2>/dev/null; then
            kill "$_SW_SERVER_PID" 2>/dev/null || true
            wait "$_SW_SERVER_PID" 2>/dev/null || true
        fi
        _SW_SERVER_PID=""
    fi
}

_sw_wait_port_free() {
    local port="$1" limit="${2:-20}" n=0
    while (( n < limit )); do
        if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
            return 0
        fi
        sleep 0.5
        (( n++ )) || true
    done
    return 0
}

_sw_attempt() {
    local server_exe="$1" port="$2" bench_tokens="$3"
    local bench_prompt="$4" timeout_secs="$5"
    shift 5

    > "$_SW_TPS_FILE"
    _sw_kill_server
    _sw_wait_port_free "$port"

    "$server_exe" "$@" >/dev/null 2>&1 &
    _SW_SERVER_PID=$!

    local ready=false elapsed=0
    while (( elapsed < timeout_secs )); do
        sleep 1
        (( elapsed++ )) || true

        if ! kill -0 "$_SW_SERVER_PID" 2>/dev/null; then
            break
        fi

        local health
        health=$(curl -sf --max-time 2 "http://127.0.0.1:${port}/health" 2>/dev/null || true)
        if echo "$health" | grep -q '"ok"'; then
            ready=true
            break
        fi
    done

    if [[ "$ready" == true ]]; then
        local body
        printf -v body \
            '{"prompt":"%s","n_predict":%d,"temperature":0.0,"cache_prompt":false}' \
            "$bench_prompt" "$bench_tokens"
        local resp
        resp=$(curl -sf --max-time 300 \
               -H "Content-Type: application/json" \
               -d "$body" \
               "http://127.0.0.1:${port}/completion" 2>/dev/null || true)
        if [[ -n "$resp" ]]; then
            local tps
            tps=$(printf '%s' "$resp" \
                  | jq -r '.timings.predicted_per_second // empty' 2>/dev/null || true)
            [[ -n "$tps" ]] && printf "%.2f" "$tps" > "$_SW_TPS_FILE"
        fi
    fi

    _sw_kill_server
}

_sw_run() {
    local server_exe="$1" port="$2" bench_tokens="$3"
    local bench_prompt="$4" timeout="$5"
    local batch_val="$6" parallel_val="$7" cont_batch="$8"
    local use_mmap="$9" use_mlock="${10}" numa_val="${11}"

    local numa_disp="null"
    [[ "$numa_val" != "null" && -n "$numa_val" ]] && numa_disp="$numa_val"

    printf "\n  \033[37mbatch=%-5s parallel=%-3s cont=%-6s mmap=%-6s mlock=%-6s numa=%s\033[0m\n" \
           "$batch_val" "$parallel_val" "$cont_batch" \
           "$use_mmap" "$use_mlock" "$numa_disp" >&2

    local -a sargs=(
        "-m"      "$_SW_GGUF"
        "-c"      "$_SW_NCTX"
        "-ngl"    "$_SW_NGL"
        "-b"      "$batch_val"
        "--host"  "127.0.0.1"
        "--port"  "$port"
        "--alias" "sweep"
    )
    (( parallel_val > 1 )) && sargs+=("-np" "$parallel_val")
    [[ "$cont_batch" == "true"  ]] && sargs+=("--cont-batching")
    [[ "$use_mmap"   == "false" ]] && sargs+=("--no-mmap")
    [[ "$use_mlock"  == "true"  ]] && sargs+=("--mlock")
    [[ "$numa_val" != "null" && -n "$numa_val" ]] && sargs+=("--numa" "$numa_val")

    _sw_spinner_start "Starting server..."
    _sw_attempt "$server_exe" "$port" "$bench_tokens" "$bench_prompt" "$timeout" "${sargs[@]}"
    _sw_spinner_stop
    local tps; tps=$(cat "$_SW_TPS_FILE" 2>/dev/null || true)

    if [[ -z "$tps" ]]; then
        printf "  \033[33mAttempt 1 failed -- waiting 5s then retrying...\033[0m\n" >&2
        sleep 5
        _sw_spinner_start "Retrying..."
        _sw_attempt "$server_exe" "$port" "$bench_tokens" "$bench_prompt" "$timeout" "${sargs[@]}"
        _sw_spinner_stop
        tps=$(cat "$_SW_TPS_FILE" 2>/dev/null || true)
    fi

    if [[ -n "$tps" ]]; then
        printf "  \033[32mResult: %s tok/s\033[0m\n" "$tps" >&2
    else
        printf "  \033[31mResult: FAILED (both attempts)\033[0m\n" >&2
        tps="FAILED"
    fi

    _SW_RESULTS+=("${batch_val}|${parallel_val}|${cont_batch}|${use_mmap}|${use_mlock}|${numa_disp}|${tps}")
}

_sw_is_better() {
    local new_t="$1" best_t="$2"
    [[ "$new_t" == "FAILED" ]] && return 1
    [[ "$best_t" == "-1"    ]] && return 0
    awk -v a="$new_t" -v b="$best_t" 'BEGIN{exit (a>b)?0:1}'
}

start_sweep_mode() {
    local server_exe="$1"
    local config_file="$2"

    _SW_GGUF=$(jq -r '.gguf_path    // ""'   "$config_file")
    _SW_NCTX=$(jq -r '.n_ctx        // 8192' "$config_file")
    _SW_NGL=$( jq -r '.n_gpu_layers // 33'   "$config_file")

    if [[ ! -f "$_SW_GGUF" ]]; then
        echo -e "\033[31mSWEEP: GGUF not found at $_SW_GGUF\033[0m" >&2
        return 1
    fi

    _SW_TPS_FILE=$(mktemp)
    trap '_sw_kill_server; _sw_spinner_stop; rm -f "$_SW_TPS_FILE"' EXIT INT TERM

    local sweep_port=18099
    local bench_tokens=4096
    local bench_prompt="Explain the difference between RAM and a hard drive in simple terms."
    local startup_wait=45

    local is_root=false
    [[ "$(id -u)" -eq 0 ]] && is_root=true

    local has_multi_numa=false numa_count
    numa_count=$(ls /sys/devices/system/node/ 2>/dev/null | grep -c '^node[0-9]' || echo 1)
    (( numa_count > 1 )) && has_multi_numa=true

    local cur_batch;    cur_batch=$(   jq -r '.batch_size    // 512'    "$config_file")
    local cur_parallel; cur_parallel=$(jq -r '.parallel      // 1'      "$config_file")
    local cur_cont;     cur_cont=$(    jq -r '.cont_batching // "true"' "$config_file")
    local cur_mmap;     cur_mmap=$(    jq -r '.use_mmap      // "true"' "$config_file")
    local cur_mlock;    cur_mlock=$(   jq -r '.use_mlock     // "false"' "$config_file")
    local cur_numa;     cur_numa=$(    jq -r '.numa          // "null"' "$config_file")
    local model_name;   model_name=$(  jq -r '.model_name   // "unknown"' "$config_file")

    [[ "$cur_batch"    == "null" ]] && cur_batch=512
    [[ "$cur_parallel" == "null" ]] && cur_parallel=1
    [[ "$cur_cont"     == "null" ]] && cur_cont="true"
    [[ "$cur_mmap"     == "null" ]] && cur_mmap="true"
    [[ "$cur_mlock"    == "null" ]] && cur_mlock="false"
    [[ "$cur_numa"     == "null" ]] && cur_numa="null"

    echo -e "\n\033[35m========================================================" >&2
    echo -e " SWEEP MODE -- llama.cpp Manager $SCRIPT_VERSION" >&2
    echo -e "========================================================\033[0m" >&2
    echo -e " Config  : \033[36m$(basename "$config_file")\033[0m" >&2
    echo -e " Model   : \033[36m${model_name}\033[0m" >&2
    echo "" >&2

    read -p "Start sweep? (y/N): " confirm
    [[ ! "$confirm" =~ ^[Yy]$ ]] && { echo -e "\033[33mSweep cancelled.\033[0m" >&2; return 0; }

    _SW_RESULTS=()

    echo -e "\n\033[36m--- Phase 0: batch_size ---\033[0m" >&2
    local best_batch="$cur_batch" best_tps="-1"
    for bval in 128 256 512 1024 2048; do
        _sw_run "$server_exe" "$sweep_port" "$bench_tokens" "$bench_prompt" "$startup_wait" \
                "$bval" 1 "$cur_cont" "$cur_mmap" "$cur_mlock" "$cur_numa"
        local t="${_SW_RESULTS[-1]##*|}"
        if _sw_is_better "$t" "$best_tps"; then best_batch=$bval; best_tps=$t; fi
    done

    echo -e "\n\033[36m--- Phase 1: parallel ---\033[0m" >&2
    local best_parallel=1; best_tps="-1"
    for pval in 1 2 4 8 16; do
        _sw_run "$server_exe" "$sweep_port" "$bench_tokens" "$bench_prompt" "$startup_wait" \
                "$best_batch" "$pval" "$cur_cont" "$cur_mmap" "$cur_mlock" "$cur_numa"
        local t="${_SW_RESULTS[-1]##*|}"
        if _sw_is_better "$t" "$best_tps"; then best_parallel=$pval; best_tps=$t; fi
    done

    echo -e "\n\033[36m--- Phase 2: cont_batching ---\033[0m" >&2
    local best_cont="$cur_cont"; best_tps="-1"
    for cbval in true false; do
        _sw_run "$server_exe" "$sweep_port" "$bench_tokens" "$bench_prompt" "$startup_wait" \
                "$best_batch" "$best_parallel" "$cbval" "$cur_mmap" "$cur_mlock" "$cur_numa"
        local t="${_SW_RESULTS[-1]##*|}"
        if _sw_is_better "$t" "$best_tps"; then best_cont=$cbval; best_tps=$t; fi
    done

    echo -e "\n\033[36m--- Phase 3: use_mmap ---\033[0m" >&2
    local best_mmap="$cur_mmap"; best_tps="-1"
    for mmval in true false; do
        _sw_run "$server_exe" "$sweep_port" "$bench_tokens" "$bench_prompt" "$startup_wait" \
                "$best_batch" "$best_parallel" "$best_cont" "$mmval" "$cur_mlock" "$cur_numa"
        local t="${_SW_RESULTS[-1]##*|}"
        if _sw_is_better "$t" "$best_tps"; then best_mmap=$mmval; best_tps=$t; fi
    done

    local best_mlock="$cur_mlock"
    if $is_root; then
        echo -e "\n\033[36m--- Phase 4: use_mlock ---\033[0m" >&2
        best_tps="-1"
        for mlval in false true; do
            _sw_run "$server_exe" "$sweep_port" "$bench_tokens" "$bench_prompt" "$startup_wait" \
                    "$best_batch" "$best_parallel" "$best_cont" "$best_mmap" "$mlval" "$cur_numa"
            local t="${_SW_RESULTS[-1]##*|}"
            if _sw_is_better "$t" "$best_tps"; then best_mlock=$mlval; best_tps=$t; fi
        done
    fi

    local best_numa="$cur_numa"
    if $has_multi_numa; then
        echo -e "\n\033[36m--- Phase 5: numa ---\033[0m" >&2
        best_tps="-1"
        for numaopt in null distribute isolate; do
            _sw_run "$server_exe" "$sweep_port" "$bench_tokens" "$bench_prompt" "$startup_wait" \
                    "$best_batch" "$best_parallel" "$best_cont" "$best_mmap" "$best_mlock" "$numaopt"
            local t="${_SW_RESULTS[-1]##*|}"
            if _sw_is_better "$t" "$best_tps"; then best_numa=$numaopt; best_tps=$t; fi
        done
    fi

    echo -e "\n\033[35m========================================================" >&2
    echo -e " SWEEP RESULTS" >&2
    echo -e "========================================================\033[0m" >&2
    printf "  %-8s %-10s %-13s %-10s %-10s %-12s %s\n" \
           "batch" "parallel" "cont_batch" "use_mmap" "use_mlock" "numa" "tok/s" >&2
    printf "  %.0s-" {1..72} >&2; echo >&2

    for row in "${_SW_RESULTS[@]}"; do
        IFS='|' read -r rb rp rc rm rl rn rt <<< "$row"
        local col="\033[0m"
        [[ "$rt" == "FAILED" ]] && col="\033[31m"
        printf "  %-8s %-10s %-13s %-10s %-10s %-12s ${col}%s\033[0m\n" \
               "$rb" "$rp" "$rc" "$rm" "$rl" "$rn" "$rt" >&2
    done

    trap - EXIT INT TERM
    rm -f "$_SW_TPS_FILE"
}

# =============================================================================
# ARGUMENT BUILDER AND RUNNER
# =============================================================================
start_llama_app() {
    local executable_path="$1"
    local config_file="$2"
    local mode="$3"

    CURRENT_EXE_PATH="$executable_path"

    echo -e "\n\033[36mLoading configuration...\033[0m"

    extract_supported_flags "$executable_path"
    sync_config_with_binary "$config_file" "$executable_path"

    local raw_model_name=$(jq -r '.model_name // "unknown"' "$config_file")

    build_dynamic_args "$config_file" "$mode"

    if [[ "$executable_path" == *arm64* || "$executable_path" == *aarch64* ]]; then
        if is_flag_in_binary "--no-warmup"; then
            ARG_LIST+=("--no-warmup")
            echo -e "  \033[33mARM64 backend detected - warmup disabled automatically.\033[0m"
        fi
    fi

    if [[ "$mode" == "server" ]] && is_flag_in_binary "--alias"; then
        ARG_LIST+=("--alias" "$raw_model_name")
    fi

    echo -e "\n\033[35m========================================================\033[0m"
    echo -e " \033[32mllama.cpp Manager $SCRIPT_VERSION\033[0m"
    echo -e " \033[32mRunning : $executable_path\033[0m"
    echo -e " \033[32mModel   : $raw_model_name\033[0m"
    echo -e " \033[90mCtrl+C to stop.\033[0m"
    echo -e "\033[35m========================================================\n\033[0m"

    echo -e "\033[90mExecuting: $executable_path ${ARG_LIST[*]}\033[0m\n"
    "$executable_path" "${ARG_LIST[@]}"
}

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================
main() {
    runner_path=$(setup_backend)
    selected_config=$(select_model_config)

    run_mode_info=$(select_run_mode "$runner_path")
    app_path=$(echo "$run_mode_info" | awk '{print $1}')
    mode=$(echo "$run_mode_info" | awk '{print $2}')

    if [[ "$mode" == "sweep" ]]; then
        start_sweep_mode "$app_path" "$selected_config"
    else
        start_llama_app "$app_path" "$selected_config" "$mode"
    fi
}

main "$@" || {
    echo -e "\n\033[31mUnexpected error occurred. Exiting...\033[0m"
    exit 1
}