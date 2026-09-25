@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 /DLLMX_COMPARE_REFERENCE /I "C:\Users\Marko\Desktop\Projects\llmx-ref\include" /I "C:\Users\Marko\Desktop\Projects\llmx-ref\ggml\include" "C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\compare.cpp" /Fo:"C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\mx.obj" /Fe:"C:\Users\Marko\AppData\Local\Temp\llmx-activation-final-perf-20260925\mx.exe" /link "C:\Users\Marko\AppData\Local\Temp\llmx-activation-perf-20260924\mx-build\src\Release\llama.lib"
