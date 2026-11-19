#!/usr/bin/env bash
# Build the RustGIS web app (with native NetCDF-4/HDF5 reading) and publish it as static files
# under a sub-path of an Apache site: the local demo server, or hdfeos.org.
#
#   packaging/hdfeos/deploy.sh local                 # -> /usr/local/var/www/rustgis  (http://localhost:8080/rustgis/)
#   HDFEOS_HOST=user@hdfeos.org HDFEOS_PATH=/var/www/html/rustgis \
#     packaging/hdfeos/deploy.sh remote              # -> https://hdfeos.org/rustgis/
#
# Options (environment):
#   SKIP_BUILD=1       reuse dist/hdfeos from the last build
#   LOCAL_WWW=dir      local target (default /usr/local/var/www/rustgis)
#   LOCAL_URL=url      URL checked after a local deploy (default http://localhost:8080/rustgis/)
#   HDFEOS_HOST        ssh login for the remote server (required for `remote`)
#   HDFEOS_PATH        directory on the server served at the sub-path (required for `remote`)
#   HDFEOS_URL         URL checked after a remote deploy (default https://hdfeos.org/rustgis/)
#
# The app is static (HTML, JS, WebAssembly); it needs no server code or special headers, only
# `application/wasm` for .wasm files, which the bundled .htaccess sets where AllowOverride permits.
set -euo pipefail

mode=${1:-}
case "$mode" in local|remote|build) ;; *) sed -n '2,20p' "$0"; exit 2 ;; esac

repo=$(cd "$(dirname "$0")/../.." && pwd)
out="$repo/dist/hdfeos"
image=rustgis-nchdf:deploy

if [ "${SKIP_BUILD:-0}" != 1 ]; then
  echo ">> building the web app in Docker (first build takes a while)"
  (cd "$repo" && DOCKER_BUILDKIT=1 docker build -t "$image" .)
  echo ">> extracting static files to $out"
  rm -rf "$out"; mkdir -p "$out"
  cid=$(docker create "$image")
  trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
  docker cp "$cid:/usr/share/nginx/html/." "$out/"
  docker rm -f "$cid" >/dev/null; trap - EXIT
  cat > "$out/.htaccess" <<'EOF'
AddType application/wasm .wasm
AddType application/manifest+json .webmanifest
<IfModule mod_headers.c>
  <FilesMatch "-[0-9a-f]{8,}(_bg)?\.(js|wasm)$">
    Header set Cache-Control "public, max-age=31536000, immutable"
  </FilesMatch>
  <FilesMatch "\.(html|webmanifest)$|^sw\.js$">
    Header set Cache-Control "no-cache"
  </FilesMatch>
</IfModule>
EOF
fi
[ -f "$out/index.html" ] || { echo "no build in $out (run without SKIP_BUILD)" >&2; exit 1; }

check() {
  local url=$1 wasm
  echo ">> checking $url"
  curl -fsS -o /dev/null "$url" || { echo "index not served at $url" >&2; return 1; }
  wasm=$(cd "$out" && ls *_bg.wasm | head -1)
  curl -fsSI "$url$wasm" | grep -i '^content-type: *application/wasm' >/dev/null \
    || { echo "warning: $url$wasm is not served as application/wasm (browsers then refuse streaming compile)" >&2; return 1; }
  echo ">> OK: open $url and drag in a .nc/.h5/.he5 file"
}

case "$mode" in
  build) echo ">> built: $out" ;;
  local)
    dest=${LOCAL_WWW:-/usr/local/var/www/rustgis}
    mkdir -p "$dest"
    rsync -a --delete "$out/" "$dest/"
    check "${LOCAL_URL:-http://localhost:8080/rustgis/}"
    ;;
  remote)
    : "${HDFEOS_HOST:?set HDFEOS_HOST, e.g. user@hdfeos.org}"
    : "${HDFEOS_PATH:?set HDFEOS_PATH, the directory served at the sub-path}"
    ssh "$HDFEOS_HOST" "mkdir -p '$HDFEOS_PATH'"
    rsync -az --delete "$out/" "$HDFEOS_HOST:$HDFEOS_PATH/"
    check "${HDFEOS_URL:-https://hdfeos.org/rustgis/}"
    ;;
esac
