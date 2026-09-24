@echo off
rem Starts the stack on the GPU when Docker can hand a container this machine's
rem NVIDIA GPU, and on the CPU everywhere else, then saves that choice in .env so
rem a plain `docker compose up` on this machine keeps making it. Arguments go to
rem docker compose; with none, it runs "up". WH_DEVICE=cpu stays on the CPU.
rem
rem   run.cmd              start everything
rem   run.cmd up --build   rebuild the images first
rem   run.cmd down         stop
rem
rem A .cmd rather than a .ps1 because Windows blocks unsigned PowerShell
rem scripts by default; this runs from PowerShell and cmd alike.
setlocal
cd /d "%~dp0"

set "COMPOSE_FILES=docker-compose.yml"

if /i "%WH_DEVICE%"=="cpu" (
  echo WH_DEVICE=cpu: using the CPU build.
  goto remember
)

rem Actually asking for the GPU is the only reliable test: Docker Desktop serves
rem GPUs through WSL 2 and lists no "nvidia" runtime in `docker info`.
docker run --rm --gpus all python:3.10-slim true >nul 2>&1
if not errorlevel 1 (
  echo NVIDIA GPU available to Docker: using the CUDA build.
  set "COMPOSE_FILES=%COMPOSE_FILES%,docker-compose.gpu.yml"
  goto remember
)

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  echo An NVIDIA GPU is present but Docker cannot use it. Enable GPU support in Docker Desktop ^(WSL 2 backend^). Using the CPU build.
) else (
  echo No NVIDIA GPU available to Docker: using the CPU build.
)

:remember
rem Only the two lines this script owns are replaced; the rest of .env is kept.
rem A comma separator reads the same on Windows and Linux.
set "TMPENV=%TEMP%\what_happened_ai_env_%RANDOM%.tmp"
if exist .env (
  findstr /v /b /c:"COMPOSE_FILE=" /c:"COMPOSE_PATH_SEPARATOR=" .env > "%TMPENV%"
) else (
  type nul > "%TMPENV%"
)
>> "%TMPENV%" echo COMPOSE_PATH_SEPARATOR=,
>> "%TMPENV%" echo COMPOSE_FILE=%COMPOSE_FILES%
move /y "%TMPENV%" .env >nul

set "COMPOSE_PATH_SEPARATOR=,"
set "COMPOSE_FILE=%COMPOSE_FILES%"

rem %* goes straight to docker, never through a variable: `set "X=%*"` ends its
rem quoting at the first quote inside an argument, and whatever follows (a | in
rem a python -c string, say) is then parsed as cmd syntax. For the same reason
rem it is kept out of parenthesised blocks.
if not "%~1"=="" goto passthrough
docker compose up
exit /b %errorlevel%

:passthrough
docker compose %*
exit /b %errorlevel%
