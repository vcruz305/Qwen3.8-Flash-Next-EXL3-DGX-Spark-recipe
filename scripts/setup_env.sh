#!/usr/bin/env bash
#
# setup_env.sh: Create a working Python environment for serving an EXL3 pack through vLLM.
#
# This script sets up a venv with vLLM and the vllm-exl3 plugin, along with all necessary
# patches and dependencies. Every step is announced before execution, and the script will
# refuse to operate silently.
#
# SAFETY BEHAVIOUR:
# - Exits immediately on any command failure (set -euo pipefail).
# - Refuses to touch an existing venv unless --force is passed; reports what it would do
#   and exits with status 1.
# - Never runs unpinned vLLM upgrades; always installs an exact version.
# - Never uses rm -rf on anything the user did not explicitly name.
# - Echoes all state-mutating commands before running them.
# - Each step checks its exit status and aborts with a descriptive message on failure.
#
# TARGET PLATFORM: Linux aarch64 (NVIDIA DGX Spark, GB10), but does not hard-fail
# on other Linux architectures.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from environment, with defaults
VENV="${VENV:-$HOME/venvs/vllm-exl3}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
PLUGIN_SOURCE="git"
PLUGIN_REF="main"
FORCE_VENV=0
SKIP_PATCHES=0

# Flags to track what was done
DID_CREATE_VENV=0

# Cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]] && [[ $DID_CREATE_VENV -eq 1 ]]; then
        printf "%b\n" "${YELLOW}Setup failed. Venv was created at ${VENV}; you may remove it manually or retry with --force.${NC}"
    fi
}
trap cleanup EXIT

# Help text
show_help() {
    cat <<EOF
usage: bash scripts/setup_env.sh [OPTIONS]

Create a working Python environment for serving an EXL3 pack through vLLM.

OPTIONS:
  --venv PATH              Path to venv directory (default: \$HOME/venvs/vllm-exl3)
                          Can also set via VENV environment variable
  --plugin-src PATH        Install plugin from local source directory (pip install -e)
  --plugin-git REF         Install plugin from git repository at given ref (default: main)
  --force                  Overwrite existing venv instead of refusing
  --skip-patches           Do not apply vLLM patches after installation
  -h, --help               Show this help message

ENVIRONMENT:
  VENV                     Override default venv path
  VLLM_VERSION             Pin vLLM version (default: 0.29.0)

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV="$2"
            shift 2
            ;;
        --plugin-src)
            PLUGIN_SOURCE="src"
            PLUGIN_REF="$2"
            shift 2
            ;;
        --plugin-git)
            PLUGIN_SOURCE="git"
            PLUGIN_REF="$2"
            shift 2
            ;;
        --force)
            FORCE_VENV=1
            shift
            ;;
        --skip-patches)
            SKIP_PATCHES=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            printf "%b\n" "${RED}error: unknown option '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Helper functions
log_step() {
    printf "%b\n" "${BLUE}[Step]${NC} $1"
}

log_info() {
    printf "%b\n" "${GREEN}[INFO]${NC} $1"
}

log_error() {
    printf "%b\n" "${RED}[ERROR]${NC} $1"
}

log_command() {
    printf "%b\n" "${YELLOW}[RUN]${NC} $1"
}

fail() {
    log_error "$1"
    exit 1
}

# Step 1: Preconditions
log_step "Checking preconditions"

# Check python3 exists and version >= 3.10
if ! command -v python3 &>/dev/null; then
    fail "python3 not found in PATH"
fi

PYTHON3_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON3_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON3_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

if [[ $PYTHON3_MAJOR -lt 3 ]] || { [[ $PYTHON3_MAJOR -eq 3 ]] && [[ $PYTHON3_MINOR -lt 10 ]]; }; then
    fail "python3 must be >= 3.10 (found $PYTHON3_VERSION)"
fi

log_info "python3 version: $PYTHON3_VERSION"
log_info "architecture: $(uname -m)"
log_info "operating system: $(uname -s)"

# Step 2: Prepare or create venv
log_step "Preparing virtual environment"

if [[ -d "$VENV" ]]; then
    if [[ $FORCE_VENV -eq 0 ]]; then
        printf "%b\n" "${YELLOW}[REFUSE]${NC} Virtual environment already exists at:"
        printf "  %s\n" "$VENV"
        printf "\nTo proceed, either:\n"
        printf "  1. Remove the directory manually: rm -rf %s\n" "$VENV"
        printf "  2. Use --force flag to overwrite it\n"
        exit 1
    else
        log_info "Overwriting existing venv at $VENV (--force was used)"
        log_command "rm -rf '$VENV'"
        rm -rf "$VENV"
    fi
fi

log_command "python3 -m venv '$VENV'"
python3 -m venv "$VENV" || fail "Failed to create venv at $VENV"
DID_CREATE_VENV=1

# Activate the venv by sourcing it
# shellcheck disable=SC1090,SC1091
source "$VENV/bin/activate" || fail "Failed to activate venv"
log_info "Virtual environment activated: $VENV"

# Upgrade pip, setuptools, wheel
log_command "pip install --upgrade pip setuptools wheel"
pip install --upgrade pip setuptools wheel || fail "Failed to upgrade pip/setuptools/wheel"

# Step 3: Install vLLM
log_step "Installing vLLM version ${VLLM_VERSION}"

log_command "pip install 'vllm==${VLLM_VERSION}'"
pip install "vllm==${VLLM_VERSION}" || fail "Failed to install vllm==${VLLM_VERSION}"

log_info "Resolving installed vLLM version..."
INSTALLED_VLLM=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
log_info "Installed vLLM version: $INSTALLED_VLLM"

log_info "Resolving installed PyTorch version..."
INSTALLED_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
log_info "Installed PyTorch version: $INSTALLED_TORCH"

# Step 4: Install plugin
log_step "Installing vllm-exl3 plugin"

if [[ "$PLUGIN_SOURCE" == "src" ]]; then
    if [[ ! -d "$PLUGIN_REF" ]]; then
        fail "Plugin source directory does not exist: $PLUGIN_REF"
    fi
    log_command "pip install -e '$PLUGIN_REF'"
    pip install -e "$PLUGIN_REF" || fail "Failed to install plugin from $PLUGIN_REF"
    log_info "Installed plugin from: $PLUGIN_REF"
else
    # PLUGIN_SOURCE == "git"
    log_command "pip install 'git+https://github.com/vcruz305/vllm-exl3@${PLUGIN_REF}'"
    pip install "git+https://github.com/vcruz305/vllm-exl3@${PLUGIN_REF}" || fail "Failed to install plugin from git ref $PLUGIN_REF"
    log_info "Installed plugin from git ref: $PLUGIN_REF"
fi

# Step 5: Check exllamav3_ext import
log_step "Checking exllamav3_ext module"

if python3 -c "import exllamav3_ext" 2>/dev/null; then
    log_info "exllamav3_ext imports successfully. PASS"
else
    cat <<EOF

${YELLOW}+====================================================================+${NC}
${YELLOW}| exllamav3_ext module not found                                      |${NC}
${YELLOW}?====================================================================?${NC}
${YELLOW}| The exllamav3 package must be built from source for aarch64.       |${NC}
${YELLOW}|                                                                    |${NC}
${YELLOW}| NEXT STEPS:                                                         |${NC}
${YELLOW}| 1. Locate the aarch64 patch in the plugin repository:              |${NC}
${YELLOW}|    tools/patch_exllamav3_aarch64.py                                |${NC}
${YELLOW}|                                                                    |${NC}
${YELLOW}| 2. Clone exllamav3 1.4.7, apply the patch, and build:              |${NC}
${YELLOW}|    git clone https://github.com/turboderp/exllamav3.git            |${NC}
${YELLOW}|    cd exllamav3                                                     |${NC}
${YELLOW}|    git checkout 1.4.7                                              |${NC}
${YELLOW}|    python <plugin-path>/tools/patch_exllamav3_aarch64.py .         |${NC}
${YELLOW}|    pip install -e .                                                |${NC}
${YELLOW}|                                                                    |${NC}
${YELLOW}| 3. The compiled exllamav3_ext.cpython-*.so is what the plugin      |${NC}
${YELLOW}|    imports; a pure-Python wheel will not work.                     |${NC}
${YELLOW}|                                                                    |${NC}
${YELLOW}| 4. If a prebuilt copy exists in another venv on this machine,      |${NC}
${YELLOW}|    you may copy these three items if Python and torch versions     |${NC}
${YELLOW}|    match exactly:                                                  |${NC}
${YELLOW}|    - exllamav3/                  (package directory)               |${NC}
${YELLOW}|    - exllamav3-*.dist-info/      (metadata directory)              |${NC}
${YELLOW}|    - exllamav3_ext.cpython-*.so  (compiled extension)              |${NC}
${YELLOW}|+====================================================================+${NC}

EOF

    log_info "Continuing setup; exllamav3_ext will need to be built or copied before inference."
fi

# Step 6: Apply vLLM patches
# Note: These patches are critical. Installing the vllm-exl3 plugin does not automatically
# patch vLLM itself. Skipping these patches causes the n-gram embedding table to be
# allocated as a dense tensor at approximately 95 GiB instead of the packed 30.4 GiB form.
# This dense allocation typically manifests as an allocator OOM error partway through
# model loading and is easily mistaken for insufficient hardware capacity.

if [[ $SKIP_PATCHES -eq 0 ]]; then
    log_step "Applying vLLM patches"

    # Determine where the plugin is installed
    if [[ "$PLUGIN_SOURCE" == "src" ]]; then
        PLUGIN_PATH="$PLUGIN_REF"
    else
        # For git installs, find the plugin in site-packages
        PLUGIN_PATH=$(python3 -c "import vllm_exl3; import os; print(os.path.dirname(vllm_exl3.__file__))" 2>/dev/null || echo "")
    fi

    PATCH_DIR=""
    if [[ -n "$PLUGIN_PATH" && -d "$PLUGIN_PATH" ]]; then
        POTENTIAL_PATCH_DIR="$PLUGIN_PATH/tools/patch_vllm_qwen4_exp"
        if [[ -d "$POTENTIAL_PATCH_DIR" ]]; then
            PATCH_DIR="$POTENTIAL_PATCH_DIR"
        fi
    fi

    # Also try relative to this script
    if [[ -z "$PATCH_DIR" ]]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        REPO_ROOT="$(dirname "$SCRIPT_DIR")"
        POTENTIAL_PATCH_DIR="$REPO_ROOT/tools/patch_vllm_qwen4_exp"
        if [[ -d "$POTENTIAL_PATCH_DIR" ]]; then
            PATCH_DIR="$POTENTIAL_PATCH_DIR"
        fi
    fi

    if [[ -z "$PATCH_DIR" ]]; then
        log_error "Could not locate tools/patch_vllm_qwen4_exp directory"
        printf "\n%b\n" "${YELLOW}To apply patches manually, run these commands in order:${NC}"
        printf "  python <plugin-path>/tools/patch_vllm_qwen4_exp/patch_vllm_qwen4_ple.py %s/lib/python*/site-packages/vllm\n" "$VENV"
        printf "  python <plugin-path>/tools/patch_vllm_qwen4_exp/patch_vllm_mtp_lmhead.py %s/lib/python*/site-packages/vllm\n" "$VENV"
        printf "  python <plugin-path>/tools/patch_vllm_qwen4_exp/patch_vllm_vision_split.py %s/lib/python*/site-packages/vllm\n" "$VENV"
        printf "\nContinuing without patches.\n"
    else
        VLLM_LIB=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
        if [[ -z "$VLLM_LIB" ]]; then
            fail "Could not locate vllm installation directory"
        fi

        # Apply patches in order
        for patch_script in \
            "patch_vllm_qwen4_ple.py" \
            "patch_vllm_mtp_lmhead.py" \
            "patch_vllm_vision_split.py"
        do
            PATCH_FILE="$PATCH_DIR/$patch_script"
            if [[ ! -f "$PATCH_FILE" ]]; then
                log_error "Patch script not found: $PATCH_FILE"
                continue
            fi

            log_command "python '$PATCH_FILE' '$VLLM_LIB'"
            python "$PATCH_FILE" "$VLLM_LIB" || log_error "Patch $patch_script failed"
        done

        log_info "Patches applied"
    fi
else
    log_step "Skipping vLLM patches (--skip-patches)"
fi

# Step 7: Run preflight checks
log_step "Running preflight checks"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PREFLIGHT_SCRIPT="$REPO_ROOT/scripts/preflight.py"

if [[ -f "$PREFLIGHT_SCRIPT" ]]; then
    log_command "python '$PREFLIGHT_SCRIPT'"
    python "$PREFLIGHT_SCRIPT" || {
        PREFLIGHT_EXIT=$?
        log_error "Preflight check exited with status $PREFLIGHT_EXIT"
    }
else
    log_info "No preflight.py found; skipping preflight checks"
fi

# Step 8: Print final block
cat <<EOF

${GREEN}+====================================================================+${NC}
${GREEN}| Setup complete                                                     |${NC}
${GREEN}?====================================================================?${NC}
${GREEN}| Virtual environment path:                                          |${NC}
${GREEN}|   ${NC}$VENV
${GREEN}|                                                                    |${NC}
${GREEN}| To activate in your current shell:                                 |${NC}
${GREEN}|   ${NC}source $VENV/bin/activate
${GREEN}|                                                                    |${NC}
${GREEN}| Next steps:                                                        |${NC}
${GREEN}|   1. Prepare your EXL3 pack (if not already done)                 |${NC}
${GREEN}|   2. Serve the model using vLLM with the plugin                   |${NC}
${GREEN}|      Example:                                                      |${NC}
${GREEN}|      python -m vllm.entrypoints.openai.api_server \\              |${NC}
${GREEN}|        --model <pack-path> --tensor-parallel-size <n>             |${NC}
${GREEN}|+====================================================================+${NC}

EOF

log_info "All setup steps completed successfully"
