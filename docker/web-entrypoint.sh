#!/bin/sh
# Replace the hardcoded dev backend URL with the configured BACKEND_URL
# in all built JS assets at container startup.
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
find /usr/share/nginx/html/assets -name '*.js' \
  -exec sed -i "s|http://localhost:17493|${BACKEND_URL}|g" {} +
