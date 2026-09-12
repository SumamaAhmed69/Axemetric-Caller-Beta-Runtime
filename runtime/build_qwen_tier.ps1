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
  Get-ChildItem 'release-assets' -File -Filter "$baseName*" -ErrorAction SilentlyContinue |
    Remove-Item -Force

  # Produce one standards-compliant ZIP byte stream and split that raw stream
  # into GitHub-safe parts as it is written. Dialforge concatenates these parts
  # byte-for-byte before Python zipfile extracts them, exactly like the original
  # v20.4 runtime feed. ZIP_STORED avoids wasting CPU on already-compressed model
  # blobs and the streaming writer avoids needing a second full 5+ GB archive on
  # the Actions runner while the 8B model store is still present.
  $env:DF_TIER_PROFILE = $Profile
  $env:DF_TIER_MODEL = $Model
  $env:DF_TIER_RELEASE = $ReleaseTag
  $env:DF_TIER_BASE = $baseName
  $env:DF_TIER_STORE = $store
  $env:DF_TIER_PART_BYTES = ([int64]$PartSizeMb * 1MB).ToString()
  @'
import hashlib
import json
import os
import zipfile
from pathlib import Path

profile = os.environ['DF_TIER_PROFILE']
model = os.environ['DF_TIER_MODEL']
tag = os.environ['DF_TIER_RELEASE']
base = os.environ['DF_TIER_BASE']
store = Path(os.environ['DF_TIER_STORE']).resolve()
part_size = int(os.environ['DF_TIER_PART_BYTES'])
out_dir = Path('release-assets').resolve()
repo = 'SumamaAhmed69/Axemetric-Caller-Beta-Runtime'


class SplitZipWriter:
    def __init__(self, directory: Path, base_name: str, limit: int):
        self.directory = directory
        self.base_name = base_name
        self.limit = limit
        self.position = 0
        self.part_index = 0
        self.part_position = 0
        self.handle = None
        self.part_path = None
        self.part_hash = None
        self.archive_hash = hashlib.sha256()
        self.parts = []

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, *_args, **_kwargs):
        raise OSError('split ZIP stream is intentionally unseekable')

    def _open_part(self):
        self.part_index += 1
        name = f'{self.base_name}.part{self.part_index:03d}'
        self.part_path = self.directory / name
        self.handle = self.part_path.open('wb')
        self.part_position = 0
        self.part_hash = hashlib.sha256()

    def _finish_part(self):
        if self.handle is None:
            return
        self.handle.flush()
        self.handle.close()
        self.parts.append({
            'file': self.part_path.name,
            'url': f'https://github.com/{repo}/releases/download/{tag}/{self.part_path.name}',
            'sha256': self.part_hash.hexdigest(),
            'size': self.part_path.stat().st_size,
        })
        self.handle = None
        self.part_path = None
        self.part_hash = None
        self.part_position = 0

    def write(self, data):
        view = memoryview(data)
        total = len(view)
        offset = 0
        while offset < total:
            if self.handle is None:
                self._open_part()
            room = self.limit - self.part_position
            take = min(room, total - offset)
            chunk = view[offset:offset + take]
            self.handle.write(chunk)
            self.archive_hash.update(chunk)
            self.part_hash.update(chunk)
            self.part_position += take
            self.position += take
            offset += take
            if self.part_position >= self.limit:
                self._finish_part()
        return total

    def flush(self):
        if self.handle is not None:
            self.handle.flush()

    def finish(self):
        self._finish_part()


writer = SplitZipWriter(out_dir, base, part_size)
files = sorted(path for path in store.rglob('*') if path.is_file())
if not files:
    raise SystemExit(f'no files found in model store: {store}')

with zipfile.ZipFile(writer, mode='w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
    for path in files:
        archive.write(path, arcname=path.relative_to(store).as_posix(), compress_type=zipfile.ZIP_STORED)
writer.finish()

if not writer.parts:
    raise SystemExit('no split ZIP parts were produced')

package = {
    'name': 'qwen-model',
    'model': model,
    'sha256': writer.archive_hash.hexdigest(),
    'size': writer.position,
    'profiles': [profile],
    'parts': writer.parts,
}
out = Path('work') / f'qwen-package-{profile}.json'
out.write_text(json.dumps(package, indent=2), encoding='utf-8')
print(json.dumps(package, indent=2))
'@ | python -
  if ($LASTEXITCODE -ne 0) { throw "Could not package $Model into raw split ZIP parts" }

  $volumes = @(Get-ChildItem 'release-assets' -File -Filter "$baseName.part*" | Sort-Object Name)
  if ($volumes.Count -lt 1) { throw "No split archive parts were produced for $Model" }

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
