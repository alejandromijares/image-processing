#!/bin/sh
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"
ENV_ERROR="This module requires Python >=3.10, pip, and virtualenv to be installed."

# Detect a usable sudo (empty when already root or sudo is absent).
SUDO="sudo"
if [ "$(id -u)" = "0" ] || ! command -v sudo >/dev/null 2>&1; then
    SUDO=""
fi

# apt_install <packages...> : install system packages on Debian/Ubuntu.
# Returns nonzero on non-apt systems so callers can fall back to a clear error.
apt_install() {
    if ! command -v apt-get >/dev/null 2>&1; then
        return 1
    fi
    if ! apt info "$1" >/dev/null 2>&1; then
        echo "Package info not found, trying apt update"
        $SUDO apt -qq update >/dev/null
    fi
    $SUDO apt install -qqy "$@" >/dev/null 2>&1
}

# --- virtualenv -----------------------------------------------------------
if ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
    echo "Failed to create virtualenv."
    if command -v apt-get >/dev/null 2>&1; then
        echo "Detected Debian/Ubuntu, attempting to install python3-venv automatically."
        if ! apt_install python3-venv || ! python3 -m venv $VENV_NAME >/dev/null 2>&1; then
            echo $ENV_ERROR >&2
            exit 1
        fi
    else
        echo $ENV_ERROR >&2
        exit 1
    fi
fi

# --- python + native dependencies -----------------------------------------
# The RAW pipeline pulls in native wheels: rawpy (bundles libraw), tifffile
# (pure python), and opencv-python-headless (bundles its own libs). These
# normally install and load with no system packages on linux/amd64,
# linux/arm64, and darwin/arm64. Two things can still go wrong on a minimal or
# headless target, though:
#   * pip can't find a matching wheel and builds rawpy from source -> needs the
#     LibRaw headers (libraw-dev)
#   * OpenCV-headless fails to load libgthread-2.0.so              -> needs glib
#     (libglib2.0-0)
# So we install requirements, then verify the imports actually load; only if
# that fails do we apt-install the system libs and retry - mirroring the
# reactive approach used for python3-venv above.
echo "Virtualenv found/created. Installing/upgrading Python packages..."
if ! [ -f .installed ]; then
    # A current pip is what lets rawpy/opencv resolve to manylinux + aarch64
    # wheels (common on ARM SBCs) instead of slow, build-heavy source installs.
    $PYTHON -m pip install -Uqq pip >/dev/null 2>&1

    # remove -U if viam-sdk should not be upgraded whenever possible
    # -qq suppresses extraneous output from pip
    if ! $PYTHON -m pip install -r requirements.txt -Uqq; then
        exit 1
    fi

    if ! $PYTHON -c "import rawpy, cv2, tifffile" >/dev/null 2>&1; then
        echo "Native imaging libraries failed to load; installing system dependencies..."
        if apt_install libraw-dev libglib2.0-0; then
            # If the missing piece was LibRaw, rebuild rawpy now that its headers
            # and runtime are present.
            $PYTHON -m pip install --no-cache-dir --force-reinstall rawpy -qq || true
            if ! $PYTHON -c "import rawpy, cv2, tifffile" >/dev/null 2>&1; then
                echo "ERROR: rawpy/cv2/tifffile still won't import after installing libraw-dev and libglib2.0-0." >&2
                exit 1
            fi
        else
            echo "ERROR: rawpy/cv2/tifffile won't import and this isn't a Debian/apt system." >&2
            echo "Install LibRaw and glib for your OS (e.g. 'brew install libraw' on macOS) and re-run." >&2
            exit 1
        fi
    fi

    touch .installed
fi
