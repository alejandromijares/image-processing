#!/bin/sh
cd `dirname $0`

# No PyInstaller: libgphoto2 dlopen's its camera/port drivers at runtime, which
# a single-file freeze can't carry. Instead we ship the source and let run.sh
# build the venv (via setup.sh) on the target, where the gphoto2 wheel's bundled
# drivers land at a path libgphoto2 can actually find.
mkdir -p dist

# Compatibility launcher: a machine that was reloaded under the old PyInstaller
# build has `dist/main` cached as its entrypoint. Ship a shim at that path that
# forwards (with args) to the real venv launcher, so a reload works whether the
# config points at dist/main or run.sh.
cat > dist/main <<'EOF'
#!/bin/sh
exec "`dirname $0`/../run.sh" "$@"
EOF
chmod +x dist/main

tar -czvf dist/archive.tar.gz \
    --exclude='__pycache__' \
    run.sh setup.sh requirements.txt meta.json src dist/main
