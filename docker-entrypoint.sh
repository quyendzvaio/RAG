#!/bin/sh
set -eu

if [ "$(id -u)" = "0" ]; then
  mkdir -p /app/faiss_index /app/hybrid_index /app/eval_data /cache
  chown -R appuser:appuser /app/faiss_index /app/hybrid_index /app/eval_data /cache
  exec gosu appuser "$0" "$@"
fi

if [ "${AUTO_BUILD_INDEX:-false}" = "true" ]; then
  if [ ! -d "/app/data_raw" ] || [ "$(find /app/data_raw -maxdepth 1 -name '*.html' 2>/dev/null | head -n 1)" = "" ]; then
    echo "Khong thay /app/data_raw/*.html. Hay mount data_raw vao container."
    exit 1
  fi
  if [ ! -f "/app/hybrid_index/index_manifest.json" ]; then
    echo "Chua co hybrid index. Dang build index theo AUTO_BUILD_INDEX=true..."
    python create_faiss.py
  else
    echo "Da co hybrid index, bo qua build."
  fi
fi

exec "$@"
