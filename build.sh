#!/bin/zsh
# Builds GlassMarbles.app from the sources in this folder. Needs Xcode or the Command Line Tools.
set -euo pipefail
cd "$(dirname "$0")"

APP=build/GlassMarbles.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# The shader is compiled by the Metal framework when the app starts, so no Metal Toolchain is needed.
swiftc -O -wmo -swift-version 5 Sources/*.swift -o "$APP/Contents/MacOS/GlassMarbles"
cp Shaders/PathTracer.metal "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Glass Marbles</string>
  <key>CFBundleIdentifier</key><string>local.glassmarbles</string>
  <key>CFBundleExecutable</key><string>GlassMarbles</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

codesign --force --sign - "$APP" >/dev/null 2>&1
echo "Built $APP"
