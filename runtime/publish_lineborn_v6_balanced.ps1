param(
  [string]$ModelPath = "",
  [string]$ReleaseTag = "beta-v20.4.0-beta.15",
  [int]$PartSizeMb = 1700,
  [switch]$KeepWork
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = "SumamaAhmed69/Axemetric-Caller-Beta-Runtime"
$ModelTag = "lineborn-v6-balanced"
$ModelFileName = "lineborn-v6-balanced-q4_k_m.gguf"
$ExpectedGgufSha256 = "467514dd216d25b7e291345f11dba9f57d2515473434f13715d159e2db6fa398"
$Quantization = "Q4_K_M"
$OllamaVersion = "v0.34.0"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $ModelPath) {
  $ModelPath = Join-Path $Root "model-source\$ModelFileName"
}
$ModelPath = [IO.Path]::GetFullPath($ModelPath)

if (-not (Test-Path $ModelPath -PathType Leaf)) {
  throw ("Model file not found: " + $ModelPath + [Environment]::NewLine +
         "Place " + $ModelFileName + " in " + (Join-Path $Root "model-source") + " and rerun.")
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  throw "GitHub CLI (gh) is required. Install it, then run: gh auth login"
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Python is required to package the split runtime archive."
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  throw "curl.exe is required to download portable Ollama."
}
& gh auth status | Out-Host
if ($LASTEXITCODE -ne 0) {
  throw "GitHub CLI is not authenticated. Run: gh auth login"
}

Write-Host "Verifying trained Lineborn v6 Balanced GGUF..." -ForegroundColor Cyan
$actualGgufSha = (Get-FileHash $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualGgufSha -ne $ExpectedGgufSha256) {
  throw ("GGUF SHA-256 mismatch." + [Environment]::NewLine +
         "Expected: " + $ExpectedGgufSha256 + [Environment]::NewLine +
         "Actual:   " + $actualGgufSha)
}
Write-Host "GGUF verified: $actualGgufSha" -ForegroundColor Green

& gh release view $ReleaseTag --repo $Repo *> $null
if ($LASTEXITCODE -ne 0) {
  throw "Release $ReleaseTag does not exist in $Repo"
}

$Work = Join-Path $Root ".lineborn-v6-publish"
$Cache = Join-Path $Work "cache"
$OllamaDir = Join-Path $Work "ollama"
$Store = Join-Path $Work "ollama-models"
$InputDir = Join-Path $Work "model-input"
$ReleaseAssets = Join-Path $Work "release-assets"
$ManifestWork = Join-Path $Work "manifest-work"
$VerifyDir = Join-Path $Work "verify"

if (Test-Path $Work) {
  Remove-Item $Work -Recurse -Force
}
New-Item -ItemType Directory -Force $Cache,$OllamaDir,$Store,$InputDir,$ReleaseAssets,$ManifestWork,$VerifyDir | Out-Null

try {
  Write-Host "Downloading current beta runtime manifest..." -ForegroundColor Cyan
  & gh release download $ReleaseTag --repo $Repo --pattern manifest.json --dir $ManifestWork --clobber
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $ManifestWork "manifest.json"))) {
    throw "Could not download the current $ReleaseTag runtime manifest"
  }

  $ollamaCommand = Get-Command ollama.exe -ErrorAction SilentlyContinue
  $ollama = if ($ollamaCommand) { $ollamaCommand.Source } else { "" }
  if (-not $ollama) {
    $archive = Join-Path $Cache "ollama-windows-amd64.zip"
    $url = "https://github.com/ollama/ollama/releases/download/$OllamaVersion/ollama-windows-amd64.zip"
    Write-Host "Downloading portable Ollama $OllamaVersion..." -ForegroundColor Cyan
    & curl.exe -L --fail --retry 5 --retry-delay 3 -o $archive $url
    if ($LASTEXITCODE -ne 0) { throw "Could not download portable Ollama" }
    Expand-Archive $archive $OllamaDir -Force
    $found = Get-ChildItem $OllamaDir -Recurse -Filter ollama.exe | Select-Object -First 1
    if (-not $found) { throw "ollama.exe missing from portable archive" }
    $ollama = $found.FullName
  }
  Write-Host "Using Ollama: $ollama" -ForegroundColor DarkGray

  $linkedModel = Join-Path $InputDir $ModelFileName
  try {
    New-Item -ItemType HardLink -Path $linkedModel -Target $ModelPath | Out-Null
    Write-Host "Created zero-copy hard link to GGUF." -ForegroundColor DarkGray
  } catch {
    Write-Host "Hard link unavailable; copying GGUF into temporary build folder..." -ForegroundColor Yellow
    Copy-Item $ModelPath $linkedModel -Force
  }

  $modelfile = Join-Path $InputDir "Modelfile.balanced"
  @"
FROM ./$ModelFileName
PARAMETER num_ctx 4096
"@ | Set-Content $modelfile -Encoding utf8

  $oldModels = $env:OLLAMA_MODELS
  $oldHost = $env:OLLAMA_HOST
  $env:OLLAMA_MODELS = $Store
  $env:OLLAMA_HOST = "127.0.0.1:11439"
  $ollamaOut = Join-Path $Work "ollama.stdout.log"
  $ollamaErr = Join-Path $Work "ollama.stderr.log"
  $server = $null
  try {
    Write-Host "Importing trained GGUF into a portable Ollama model store..." -ForegroundColor Cyan
    $server = Start-Process -FilePath $ollama -ArgumentList "serve" -PassThru -WindowStyle Hidden -RedirectStandardOutput $ollamaOut -RedirectStandardError $ollamaErr
    $ready = $false
    for ($i=0; $i -lt 120; $i++) {
      try {
        $response = Invoke-WebRequest "http://127.0.0.1:11439/api/tags" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $ready = $true; break }
      } catch {}
      if ($server.HasExited) {
        $outTail = if (Test-Path $ollamaOut) { Get-Content $ollamaOut -Tail 100 | Out-String } else { "" }
        $errTail = if (Test-Path $ollamaErr) { Get-Content $ollamaErr -Tail 100 | Out-String } else { "" }
        throw ("Ollama exited before becoming ready." + [Environment]::NewLine + $outTail + [Environment]::NewLine + $errTail)
      }
      Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "Portable Ollama did not become ready" }

    Push-Location $InputDir
    try {
      & $ollama create $ModelTag -f $modelfile
      if ($LASTEXITCODE -ne 0) { throw "ollama create failed for $ModelTag" }
      & $ollama show $ModelTag *> $null
      if ($LASTEXITCODE -ne 0) { throw "Ollama cannot see $ModelTag after import" }
    } finally {
      Pop-Location
    }
  } finally {
    if ($server -and -not $server.HasExited) {
      Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
      try { $server.WaitForExit(10000) | Out-Null } catch {}
    }
    $env:OLLAMA_MODELS = $oldModels
    $env:OLLAMA_HOST = $oldHost
  }

  $manifestDir = Join-Path $Store "manifests"
  $blobDir = Join-Path $Store "blobs"
  if (-not (Test-Path $manifestDir) -or -not (Test-Path $blobDir)) {
    throw "Ollama model store is incomplete"
  }
  if (-not (Get-ChildItem $manifestDir -Recurse -File | Select-Object -First 1)) {
    throw "Ollama model manifest was not created"
  }
  if (-not (Get-ChildItem $blobDir -Recurse -File | Select-Object -First 1)) {
    throw "Ollama model blobs were not created"
  }

  $packScript = Join-Path $Work "pack_model.py"
  @'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

store = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
base_name = sys.argv[3]
part_size = int(sys.argv[4])
repo = sys.argv[5]
tag = sys.argv[6]
model = sys.argv[7]
gguf_sha = sys.argv[8]
quant = sys.argv[9]
metadata_path = Path(sys.argv[10]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

class SplitZipWriter:
    def __init__(self, directory, base, limit):
        self.directory = directory
        self.base = base
        self.limit = limit
        self.position = 0
        self.part_index = 0
        self.part_position = 0
        self.handle = None
        self.part_path = None
        self.part_hash = None
        self.archive_hash = hashlib.sha256()
        self.parts = []

    def writable(self): return True
    def seekable(self): return False
    def tell(self): return self.position
    def seek(self, *_args, **_kwargs): raise OSError("split stream is unseekable")

    def _open(self):
        self.part_index += 1
        name = f"{self.base}.part{self.part_index:03d}"
        self.part_path = self.directory / name
        self.handle = self.part_path.open("wb")
        self.part_position = 0
        self.part_hash = hashlib.sha256()

    def _finish(self):
        if self.handle is None:
            return
        self.handle.flush()
        self.handle.close()
        self.parts.append({
            "file": self.part_path.name,
            "url": f"https://github.com/{repo}/releases/download/{tag}/{self.part_path.name}",
            "sha256": self.part_hash.hexdigest(),
            "size": self.part_path.stat().st_size,
        })
        self.handle = None
        self.part_path = None
        self.part_hash = None
        self.part_position = 0

    def write(self, data):
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            if self.handle is None:
                self._open()
            room = self.limit - self.part_position
            take = min(room, len(view) - offset)
            chunk = view[offset:offset+take]
            self.handle.write(chunk)
            self.archive_hash.update(chunk)
            self.part_hash.update(chunk)
            self.position += take
            self.part_position += take
            offset += take
            if self.part_position >= self.limit:
                self._finish()
        return len(view)

    def flush(self):
        if self.handle is not None:
            self.handle.flush()

    def finish(self):
        self._finish()

files = sorted(p for p in store.rglob("*") if p.is_file())
if not files:
    raise SystemExit(f"empty Ollama model store: {store}")

writer = SplitZipWriter(out_dir, base_name, part_size)
with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
    for path in files:
        archive.write(path, arcname=path.relative_to(store).as_posix(), compress_type=zipfile.ZIP_STORED)
writer.finish()

if not writer.parts:
    raise SystemExit("no split ZIP parts were produced")

package = {
    "name": "qwen-model",
    "model": model,
    "sha256": writer.archive_hash.hexdigest(),
    "size": writer.position,
    "profiles": ["compatibility", "balanced", "performance"],
    "platforms": ["windows", "macos"],
    "arches": ["x64", "arm64"],
    "trained": True,
    "model_family": "lineborn-v6",
    "quantization": quant,
    "source_gguf_sha256": gguf_sha,
    "parts": writer.parts,
}
metadata_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
print(json.dumps(package, indent=2))
'@ | Set-Content $packScript -Encoding utf8

  $packageJson = Join-Path $Work "qwen-package-lineborn-v6-balanced.json"
  Write-Host "Packaging trained Ollama store into GitHub-safe split ZIP parts..." -ForegroundColor Cyan
  & python $packScript $Store $ReleaseAssets "qwen-model-lineborn-v6-balanced.zip" ([int64]$PartSizeMb * 1MB) $Repo $ReleaseTag $ModelTag $ExpectedGgufSha256 $Quantization $packageJson
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $packageJson)) {
    throw "Could not package the trained Lineborn model"
  }

  $patchScript = Join-Path $Work "patch_manifest.py"
  @'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
package_path = Path(sys.argv[2])
model = sys.argv[3]
gguf_sha = sys.argv[4]

doc = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
package = json.loads(package_path.read_text(encoding="utf-8"))
packages = [p for p in doc.get("packages", []) if p.get("name") != "qwen-model"]
packages.append(package)

doc["qwen_model"] = "profile-selected"
doc["qwen_models"] = {
    "compatibility": model,
    "balanced": model,
    "performance": model,
}
doc["default_profile"] = "balanced"
doc["lineborn_model_family"] = "lineborn-v6"
doc["trained_model"] = {
    "model": model,
    "quantization": "Q4_K_M",
    "source_gguf_sha256": gguf_sha,
    "release_gate_pass": True,
}
doc["generated_at"] = datetime.now(timezone.utc).isoformat()
doc["packages"] = packages

required = {"ollama", "qwen-model", "chatterbox", "speech-cache", "sip"}
for platform_name, arches in {"windows":["x64"], "macos":["x64","arm64"]}.items():
    for arch in arches:
        for profile in ("compatibility", "balanced", "performance"):
            selected = [
                p for p in packages
                if profile in (p.get("profiles") or [])
                and platform_name in (p.get("platforms") or ["windows"])
                and arch in (p.get("arches") or ["x64"])
            ]
            names = {p["name"] for p in selected}
            if names != required:
                raise SystemExit(
                    f"{platform_name}/{arch}/{profile} runtime mismatch: "
                    f"expected {sorted(required)}, got {sorted(names)}"
                )
            qwen = [p for p in selected if p["name"] == "qwen-model"]
            if len(qwen) != 1 or qwen[0].get("model") != model:
                raise SystemExit(f"{platform_name}/{arch}/{profile}: trained Lineborn qwen-model selection failed")

manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
print("Patched runtime manifest with trained Lineborn v6 Balanced model.")
'@ | Set-Content $patchScript -Encoding utf8

  $manifestPath = Join-Path $ManifestWork "manifest.json"
  & python $patchScript $manifestPath $packageJson $ModelTag $ExpectedGgufSha256
  if ($LASTEXITCODE -ne 0) { throw "Could not patch runtime manifest" }
  Copy-Item $manifestPath (Join-Path $ReleaseAssets "manifest.json") -Force

  $parts = @(Get-ChildItem $ReleaseAssets -File -Filter "qwen-model-lineborn-v6-balanced.zip.part*" | Sort-Object Name)
  if ($parts.Count -lt 1) {
    throw "No release parts were produced"
  }

  Write-Host "Uploading trained Lineborn v6 runtime assets..." -ForegroundColor Cyan
  foreach ($part in $parts) {
    Write-Host "Uploading $($part.Name) ($([math]::Round($part.Length / 1MB,1)) MB)..."
    & gh release upload $ReleaseTag $part.FullName --repo $Repo --clobber
    if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($part.Name)" }
  }
  & gh release upload $ReleaseTag (Join-Path $ReleaseAssets "manifest.json") --repo $Repo --clobber
  if ($LASTEXITCODE -ne 0) { throw "Manifest upload failed" }

  Write-Host "Verifying published release..." -ForegroundColor Cyan
  & gh release download $ReleaseTag --repo $Repo --pattern manifest.json --dir $VerifyDir --clobber
  if ($LASTEXITCODE -ne 0) { throw "Could not redownload published manifest" }

  $verifyScript = Join-Path $Work "verify_published.py"
  @'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
model = sys.argv[2]
gguf_sha = sys.argv[3]
qwen = [p for p in manifest.get("packages", []) if p.get("name") == "qwen-model"]
if len(qwen) != 1:
    raise SystemExit(f"expected one qwen-model package, found {len(qwen)}")
pkg = qwen[0]
assert pkg["model"] == model, pkg.get("model")
assert pkg.get("trained") is True
assert pkg.get("source_gguf_sha256") == gguf_sha
assert manifest.get("default_profile") == "balanced"
assert set(manifest.get("qwen_models", {}).values()) == {model}
assert pkg.get("parts")
print("Published manifest selects trained Lineborn v6 Balanced for every beta hardware profile.")
print(json.dumps(pkg, indent=2))
'@ | Set-Content $verifyScript -Encoding utf8

  & python $verifyScript (Join-Path $VerifyDir "manifest.json") $ModelTag $ExpectedGgufSha256
  if ($LASTEXITCODE -ne 0) { throw "Published manifest verification failed" }

  $published = (& gh api "repos/$Repo/releases/tags/$ReleaseTag") | ConvertFrom-Json
  $package = Get-Content $packageJson -Raw | ConvertFrom-Json
  foreach ($part in @($package.parts)) {
    $asset = @($published.assets | Where-Object { $_.name -eq $part.file }) | Select-Object -First 1
    if (-not $asset) { throw "Published release asset missing: $($part.file)" }
    if ([int64]$asset.size -ne [int64]$part.size) {
      throw "Published asset size mismatch for $($part.file)"
    }
    if ($asset.digest -and $asset.digest -ne "sha256:$($part.sha256)") {
      throw "Published asset digest mismatch for $($part.file)"
    }
  }

  Write-Host ""
  Write-Host "TRAINED LINEBORN V6 BALANCED MODEL PUBLISHED SUCCESSFULLY" -ForegroundColor Green
  Write-Host "Release: https://github.com/$Repo/releases/tag/$ReleaseTag"
  Write-Host "Model:   $ModelTag"
  Write-Host "GGUF:    $ExpectedGgufSha256"
  Write-Host "The beta runtime manifest now installs the trained Lineborn model by default." -ForegroundColor Green
}
finally {
  if (-not $KeepWork -and (Test-Path $Work)) {
    Remove-Item $Work -Recurse -Force -ErrorAction SilentlyContinue
  } elseif ($KeepWork) {
    Write-Host "Build work preserved at: $Work" -ForegroundColor DarkGray
  }
}
