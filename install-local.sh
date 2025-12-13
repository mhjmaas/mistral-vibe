#!/usr/bin/env bash
set -e  # Exit on error

echo "🧹 Cleaning dist directory..."
rm -rf dist/

echo "🔨 Building package..."
uv build

echo "📦 Installing package..."
# Find the wheel file (there should be only one after a fresh build)
WHEEL_FILE=$(find dist/ -name "*.whl" -type f | head -n 1)

if [ -z "$WHEEL_FILE" ]; then
    echo "❌ Error: No wheel file found in dist/"
    exit 1
fi

echo "   Installing: $WHEEL_FILE"
uv tool install --force "$WHEEL_FILE"

echo "✅ Done! Package installed successfully"
uv tool list | grep mistral-vibe
