@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "LLMX_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%LLMX_VSWHERE%" ( echo [build] vswhere.exe not found at "%LLMX_VSWHERE%": install Visual Studio 2022 or later with the Desktop development with C++ workload & popd & exit /b 1 )
set "LLMX_VS="
for /f "usebackq delims=" %%P in (`"%LLMX_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "LLMX_VS=%%P"
if not defined LLMX_VS for /f "usebackq delims=" %%P in (`"%LLMX_VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "LLMX_VS=%%P"
if not defined LLMX_VS ( echo [build] no Visual Studio installation has the x64 C++ build tools: add the Desktop development with C++ workload in the Visual Studio Installer & popd & exit /b 1 )
if not exist "%LLMX_VS%\VC\Auxiliary\Build\vcvars64.bat" ( echo [build] vcvars64.bat not found under "%LLMX_VS%": repair the C++ workload in the Visual Studio Installer & popd & exit /b 1 )
call "%LLMX_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo [build] vcvars64 failed & popd & exit /b 1 )
set "LLMX_BUILD_ID=unknown"
set "LLMX_GIT_ROOT="
for /f "delims=" %%R in ('git rev-parse --show-toplevel 2^>nul') do set "LLMX_GIT_ROOT=%%R"
set "LLMX_GIT_ROOT=%LLMX_GIT_ROOT:/=\%"
if /I not "%LLMX_GIT_ROOT%"=="%CD%" goto version_ready
for /f "delims=" %%R in ('git rev-parse "--short=12" HEAD 2^>nul') do set "LLMX_BUILD_ID=g%%R"
if "%LLMX_BUILD_ID%"=="unknown" goto version_ready
git diff --quiet HEAD --
if errorlevel 2 (
    set "LLMX_BUILD_ID=unknown"
) else if errorlevel 1 (
    set "LLMX_BUILD_ID=%LLMX_BUILD_ID%.dirty"
)
:version_ready
if not exist build\plain-generated (
    mkdir build\plain-generated
    if errorlevel 1 ( echo [build] cannot create version directory & popd & exit /b 1 )
)
ver >nul
>build\plain-generated\llmx-build-info.hpp (
    echo #pragma once
    echo #define LLMX_BUILD_REVISION "%LLMX_BUILD_ID%"
) || ( echo [build] cannot write version header & popd & exit /b 1 )
cl /nologo /std:c++17 /O2 /EHsc /W4 /arch:AVX2 /I build\plain-generated /I src /Fe:llmx.exe src\cli\main.cpp
set "LLMX_BUILD_EXIT=%errorlevel%"
popd
exit /b %LLMX_BUILD_EXIT%
