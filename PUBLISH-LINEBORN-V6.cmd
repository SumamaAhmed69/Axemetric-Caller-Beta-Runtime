@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Lineborn v6 Balanced - trained model publisher
echo ============================================================
echo.
echo Expected file:
echo   model-source\lineborn-v6-balanced-q4_k_m.gguf
echo.
if not exist "model-source\lineborn-v6-balanced-q4_k_m.gguf" (
  echo ERROR: trained GGUF not found.
  echo.
  echo Copy lineborn-v6-balanced-q4_k_m.gguf into:
  echo   %CD%\model-source\
  echo.
  pause
  exit /b 1
)

where gh >nul 2>nul
if errorlevel 1 (
  echo ERROR: GitHub CLI is not installed.
  echo Install it with:
  echo   winget install --id GitHub.cli
  echo Then run:
  echo   gh auth login
  echo.
  pause
  exit /b 1
)

gh auth status >nul 2>nul
if errorlevel 1 (
  echo GitHub CLI needs authentication.
  echo A browser login will open now.
  gh auth login --web --git-protocol https
  if errorlevel 1 (
    echo ERROR: GitHub authentication failed.
    pause
    exit /b 1
  )
)

echo.
echo Verifying, packaging, uploading and publishing the trained model...
echo This uploads several GB to the beta runtime release.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\runtime\publish_lineborn_v6_balanced.ps1" -ModelPath "%CD%\model-source\lineborn-v6-balanced-q4_k_m.gguf"
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" (
  echo PUBLISH FAILED with exit code %ERR%.
  echo Keep this window open and copy the error if you need help.
  pause
  exit /b %ERR%
)

echo.
echo ============================================================
echo TRAINED LINEBORN V6 BALANCED MODEL IS PUBLISHED
echo ============================================================
echo.
pause
exit /b 0
