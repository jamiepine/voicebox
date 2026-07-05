# Ensure virtual environment is activated if present
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (.venv)..."
    . .venv\Scripts\Activate.ps1
}

# Check and install PyInstaller
python -c "import PyInstaller" 2>$null
if ($LastExitCode -ne 0) {
    Write-Host "Installing PyInstaller..."
    pip install pyinstaller
}

# Get the target triple from rustc
$Platform = (rustc --print host-tuple 2>$null)
if (-not $Platform) {
    $Platform = "x86_64-pc-windows-msvc"
}

Write-Host "Building Voicebox sidecars for platform: $Platform"

# Ensure binaries directory exists
$BinariesDir = "tauri/src-tauri/binaries"
if (-not (Test-Path $BinariesDir)) {
    New-Item -ItemType Directory -Path $BinariesDir -Force | Out-Null
}

# Build server binary
cd backend
Write-Host "Compiling voicebox-server sidecar..."
python build_binary.py
if (Test-Path "dist/voicebox-server.exe") {
    Copy-Item "dist/voicebox-server.exe" "../$BinariesDir/voicebox-server-$Platform.exe" -Force
    Write-Host "Successfully compiled voicebox-server-$Platform.exe"
} else {
    Write-Error "Error: voicebox-server binary not found in dist/"
    cd ..
    exit 1
}

# Build MCP shim binary
Write-Host "Compiling voicebox-mcp sidecar..."
python build_binary.py --shim
if (Test-Path "dist/voicebox-mcp.exe") {
    Copy-Item "dist/voicebox-mcp.exe" "../$BinariesDir/voicebox-mcp-$Platform.exe" -Force
    Write-Host "Successfully compiled voicebox-mcp-$Platform.exe"
} else {
    Write-Error "Error: voicebox-mcp binary not found in dist/"
    cd ..
    exit 1
}

cd ..

# Build the Tauri application
Write-Host "Compiling Tauri desktop application..."
if (Get-Command "bun" -ErrorAction SilentlyContinue) {
    Write-Host "Using bun to install dependencies and build..."
    bun install
    cd tauri
    bun run tauri build
} else {
    Write-Host "Using npm to install dependencies and build..."
    npm install
    cd tauri
    npx tauri build
}

Write-Host "Compilation complete! Your standalone installer/executable is built."
