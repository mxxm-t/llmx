@call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" >nul
@cl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 /I C:\Users\Marko\AppData\Local\Temp\llmx-activation-perf-candidate-20260924\build\generated /I C:\Users\Marko\AppData\Local\Temp\llmx-activation-perf-candidate-20260924\src C:\Users\Marko\AppData\Local\Temp\llmx-activation-perf-20260924\compare.cpp /Fo:C:\Users\Marko\AppData\Local\Temp\llmx-activation-perf-20260924\candidate.obj /Fe:C:\Users\Marko\AppData\Local\Temp\llmx-activation-perf-20260924\candidate.exe
@exit /b %errorlevel%
