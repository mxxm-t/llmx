@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 /I "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-base-20260925\build\generated" /I "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-base-20260925\src" "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\compare.cpp" /Fo:"C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\base.obj" /Fe:"C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\base.exe"
