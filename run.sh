#!/bin/sh
cd `dirname $0`

# Build the virtualenv and install dependencies on the target machine. The
# gphoto2 wheel ships native libgphoto2 plus its camera/port driver plugins
# (camlibs/iolibs), which are dlopen'd at runtime - they only resolve when the
# wheel sits at its real install path, so we install on-device rather than
# freezing with PyInstaller. setup.sh is idempotent (guards on .installed).
if [ ! -f .installed ]; then
    ./setup.sh || exit 1
fi

# Run from src/ so `from models...` resolves (src is sys.path[0]). Forward "$@"
# so viam-server's socket-path argument reaches the module.
exec venv/bin/python src/main.py "$@"
