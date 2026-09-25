@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 /I "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-layout-20260925\build\generated" /I "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-layout-20260925\src" "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\compare.cpp" /Fo:"C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\layout.obj" /Fe:"C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\layout.exe"
