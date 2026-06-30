#!/usr/bin/env node

/**
 * Creates placeholder sidecar binaries for development mode.
 *
 * In dev mode, Tauri requires the sidecar binary files to exist at compile time,
 * even though developers typically run the Python server manually.
 *
 * This script creates minimal placeholder binaries that allow Tauri to compile.
 * The actual server should be started separately with `bun run dev:server`.
 */

import { execSync } from 'child_process';
import { existsSync, mkdirSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const BINARIES_DIR = join(__dirname, '..', 'tauri', 'src-tauri', 'binaries');

// Minimum size to consider a binary "real" (placeholder is ~256 bytes, real is MBs)
const MIN_REAL_BINARY_SIZE = 10000;

// Get the current platform's target triple
function getTargetTriple() {
  try {
    const triple = execSync('rustc --print host-tuple', { encoding: 'utf-8' }).trim();
    return triple;
  } catch {
    // Fallback detection
    const platform = process.platform;
    const arch = process.arch;

    if (platform === 'win32') {
      return arch === 'x64' ? 'x86_64-pc-windows-msvc' : 'i686-pc-windows-msvc';
    } else if (platform === 'darwin') {
      return arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
    } else if (platform === 'linux') {
      return arch === 'x64' ? 'x86_64-unknown-linux-gnu' : 'aarch64-unknown-linux-gnu';
    }

    throw new Error(`Unsupported platform: ${platform}/${arch}`);
  }
}

// Create a minimal executable for the platform. ``baseName`` is the
// sidecar identifier as declared in tauri.conf.json's ``externalBin``
// (e.g. "voicebox-server", "voicebox-mcp"). Tauri appends the target
// triple to that name at compile time.
function createPlaceholderBinary(targetTriple, baseName) {
  const isWindows = targetTriple.includes('windows');
  const binaryName = `${baseName}-${targetTriple}${isWindows ? '.exe' : ''}`;
  const binaryPath = join(BINARIES_DIR, binaryName);

  // Check if real binary already exists (larger than our placeholder)
  if (existsSync(binaryPath)) {
    try {
      const stats = statSync(binaryPath);
      if (stats.size > MIN_REAL_BINARY_SIZE) {
        console.log(
          `Real binary already exists: ${binaryName} (${(stats.size / 1024 / 1024).toFixed(1)} MB)`,
        );
        return;
      }
    } catch {
      // File exists but can't stat - try to replace it
    }
  }

  // Ensure binaries directory exists
  if (!existsSync(BINARIES_DIR)) {
    mkdirSync(BINARIES_DIR, { recursive: true });
  }

  if (isWindows) {
    // Create a minimal valid Windows PE executable that exits with code 1
    // This is the smallest valid PE that Windows will accept
    const minimalPE = Buffer.from([
      // DOS Header
      0x4d,
      0x5a,
      0x90,
      0x00,
      0x03,
      0x00,
      0x00,
      0x00,
      0x04,
      0x00,
      0x00,
      0x00,
      0xff,
      0xff,
      0x00,
      0x00,
      0xb8,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x40,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x80,
      0x00,
      0x00,
      0x00,
      // DOS Stub
      0x0e,
      0x1f,
      0xba,
      0x0e,
      0x00,
      0xb4,
      0x09,
      0xcd,
      0x21,
      0xb8,
      0x01,
      0x4c,
      0xcd,
      0x21,
      0x54,
      0x68,
      0x69,
      0x73,
      0x20,
      0x70,
      0x72,
      0x6f,
      0x67,
      0x72,
      0x61,
      0x6d,
      0x20,
      0x63,
      0x61,
      0x6e,
      0x6e,
      0x6f,
      0x74,
      0x20,
      0x62,
      0x65,
      0x20,
      0x72,
      0x75,
      0x6e,
      0x20,
      0x69,
      0x6e,
      0x20,
      0x44,
      0x4f,
      0x53,
      0x20,
      0x6d,
      0x6f,
      0x64,
      0x65,
      0x2e,
      0x0d,
      0x0d,
      0x0a,
      0x24,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      // PE Signature
      0x50,
      0x45,
      0x00,
      0x00,
      // COFF Header (x64)
      0x64,
      0x86, // Machine: AMD64
      0x01,
      0x00, // NumberOfSections: 1
      0x00,
      0x00,
      0x00,
      0x00, // TimeDateStamp
      0x00,
      0x00,
      0x00,
      0x00, // PointerToSymbolTable
      0x00,
      0x00,
      0x00,
      0x00, // NumberOfSymbols
      0xf0,
      0x00, // SizeOfOptionalHeader
      0x22,
      0x00, // Characteristics: EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
      // Optional Header (PE32+)
      0x0b,
      0x02, // Magic: PE32+
      0x00,
      0x00, // Linker version
      0x00,
      0x00,
      0x00,
      0x00, // SizeOfCode
      0x00,
      0x00,
      0x00,
      0x00, // SizeOfInitializedData
      0x00,
      0x00,
      0x00,
      0x00, // SizeOfUninitializedData
      0x00,
      0x10,
      0x00,
      0x00, // AddressOfEntryPoint
      0x00,
      0x00,
      0x00,
      0x00, // BaseOfCode
      0x00,
      0x00,
      0x00,
      0x40,
      0x01,
      0x00,
      0x00,
      0x00, // ImageBase
      0x00,
      0x10,
      0x00,
      0x00, // SectionAlignment
      0x00,
      0x02,
      0x00,
      0x00, // FileAlignment
      0x06,
      0x00,
      0x00,
      0x00, // OS version
      0x00,
      0x00,
      0x00,
      0x00, // Image version
      0x06,
      0x00,
      0x00,
      0x00, // Subsystem version
      0x00,
      0x00,
      0x00,
      0x00, // Win32VersionValue
      0x00,
      0x20,
      0x00,
      0x00, // SizeOfImage
      0x00,
      0x02,
      0x00,
      0x00, // SizeOfHeaders
      0x00,
      0x00,
      0x00,
      0x00, // CheckSum
      0x03,
      0x00, // Subsystem: CONSOLE
      0x60,
      0x01, // DllCharacteristics
      // Stack/Heap sizes (8 bytes each for PE32+)
      0x00,
      0x00,
      0x10,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x10,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00, // LoaderFlags
      0x10,
      0x00,
      0x00,
      0x00, // NumberOfRvaAndSizes
    ]);

    // Pad to 512 bytes minimum for valid PE
    const paddedPE = Buffer.alloc(512);
    minimalPE.copy(paddedPE);
    writeFileSync(binaryPath, paddedPE);
  } else {
    // Create a minimal shell script for Unix-like systems
    const script = `#!/bin/sh
echo "[${baseName}] Dev mode placeholder - start the real server with: bun run dev:server"
exit 1
`;
    writeFileSync(binaryPath, script, { mode: 0o755 });
  }

  console.log(`Created dev placeholder: ${binaryName}`);
}

// Every sidecar listed in tauri.conf.json's ``externalBin`` needs a
// file on disk at compile time, even in dev. Add to this list whenever
// a new sidecar is introduced.
const SIDECAR_BASE_NAMES = ['voicebox-server', 'voicebox-mcp'];

function main() {
  const targetTriple = getTargetTriple();
  for (const baseName of SIDECAR_BASE_NAMES) {
    createPlaceholderBinary(targetTriple, baseName);
  }
}

main();
