param(
  [Parameter(Mandatory=$true)][ValidateSet('compatibility','balanced','performance')][string]$Profile,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$ReleaseTag,
  [int]$PartSizeMb = 1700
)

$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Push-Location $Root
try {
  $ollama = (Resolve-Path 'stage/ollama/ollama.exe').Path
  $store = Join-Path (Resolve-Path 'stage').Path "qwen-$Profile"
  if (Test-Path $store) { Remove-Item $store -Recurse -Force }
  New-Item -ItemType Directory -Force $store | Out-Null

  $env:OLLAMA_MODELS = $store
  $server = Start-Process -FilePath $ollama -ArgumentList 'serve' -PassThru -WindowStyle Hidden
  try {
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
      try {
        $r = Invoke-WebRequest 'http://127.0.0.1:11434/api/tags' -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
      } catch {}
      Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'Ollama did not become ready' }

    Write-Host "Pulling $Model for $Profile profile..."
    & $ollama pull $Model
    if ($LASTEXITCODE -ne 0) { throw "Ollama pull failed for $Model" }

    $tags = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 10
    $found = @($tags.models | Where-Object { $_.name -eq $Model -or $_.model -eq $Model })
    if (-not $found) { throw "Ollama store does not report $Model after pull" }
  } finally {
    if ($server -and -not $server.HasExited) {
      Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
      try { $server.WaitForExit(10000) | Out-Null } catch {}
    }
  }

  if (-not (Get-ChildItem $store -Recurse -File | Select-Object -First 1)) {
    throw "Qwen model store for $Profile is empty"
  }

  $baseName = "qwen-model-$Profile.zip"
  $basePath = Join-Path 'release-assets' $baseName
  Get-ChildItem 'release-assets' -File -Filter "$baseName.*" -ErrorAction SilentlyContinue |
    Remove-Item -Force

  Write-Host "Packaging $Model into split release volumes..."
  & 7z a -tzip -mx=0 "-v${PartSizeMb}m" $basePath "$store\*"
  if ($LASTEXITCODE -ne 0) { throw "7-Zip packaging failed for $Model" }

  $volumes = @(Get-ChildItem 'release-assets' -File -Filter "$baseName.*" | Sort-Object Name)
  if ($volumes.Count -lt 1) { throw "No split archive volumes were produced for $Model" }

  $env:DF_TIER_PROFILE = $Profile
  $env:DF_TIER_MODEL = $Model
  $env:DF_TIER_RELEASE = $ReleaseTag
  $env:DF_TIER_BASE = $baseName
  @'
import hashlib
import json
import os
from pathlib import Path

profile = os.environ['DF_TIER_PROFILE']
model = os.environ['DF_TIER_MODEL']
tag = os.environ['DF_TIER_RELEASE']
base = os.environ['DF_TIER_BASE']
repo = 'SumamaAhmed69/Axemetric-Caller-Beta-Runtime'
volumes = sorted(Path('release-assets').glob(base + '.*'))
if not volumes:
    raise SystemExit('no archive volumes found')

archive_hash = hashlib.sha256()
archive_size = 0
parts = []
for path in volumes:
    part_hash = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            archive_hash.update(chunk)
            part_hash.update(chunk)
            archive_size += len(chunk)
    parts.append({
        'file': path.name,
        'url': f'https://github.com/{repo}/releases/download/{tag}/{path.name}',
        'sha256': part_hash.hexdigest(),
        'size': path.stat().st_size,
    })

package = {
    'name': 'qwen-model',
    'model': model,
    'sha256': archive_hash.hexdigest(),
    'size': archive_size,
    'profiles': [profile],
    'parts': parts,
}
out = Path('work') / f'qwen-package-{profile}.json'
out.write_text(json.dumps(package, indent=2), encoding='utf-8')
print(json.dumps(package, indent=2))
'@ | python -
  if ($LASTEXITCODE -ne 0) { throw "Could not generate manifest metadata for $Model" }

  foreach ($volume in $volumes) {
    Write-Host "Uploading $($volume.Name) ($([math]::Round($volume.Length / 1MB, 1)) MB)..."
    gh release upload $ReleaseTag $volume.FullName --clobber
    if ($LASTEXITCODE -ne 0) { throw "Release upload failed: $($volume.Name)" }
  }

  # Keep only the tiny JSON metadata between tiers. This prevents the 8B build
  # from exhausting GitHub's Windows runner disk with old model archives.
  Remove-Item $store -Recurse -Force
  foreach ($volume in $volumes) { Remove-Item $volume.FullName -Force }
  Write-Host "$Profile tier published: $Model"
  Get-PSDrive C | Format-Table -AutoSize
} finally {
  Pop-Location
}
