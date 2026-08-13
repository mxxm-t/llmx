@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo [build] vcvars64 failed & exit /b 1 )
cl /nologo /std:c++17 /O2 /EHsc /W4 /arch:AVX2 /I src /Fe:llmx.exe src\cli\main.cpp
exit /b %errorlevel%
