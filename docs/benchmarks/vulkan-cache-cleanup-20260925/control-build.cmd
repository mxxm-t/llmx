@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /O2 /Ob2 /DNDEBUG /EHsc /W4 /arch:AVX2 /I "C:\Users\Marko\AppData\Local\Temp\llmx-vulkan-lifetime-main-20260925\src" /I "C:\Users\Marko\AppData\Local\Temp\llmx-vulkan-lifetime-main-20260925\build\generated" /I "C:/VulkanSDK/1.4.357.0/Include" "C:\Users\Marko\AppData\Local\Temp\llmx-vulkan-cache-cleanup-final-evidence-20260925\control-tests.cpp" /Fo:"C:\Users\Marko\AppData\Local\Temp\llmx-vulkan-cache-cleanup-final-evidence-20260925\control-tests.obj" /Fe:"C:\Users\Marko\AppData\Local\Temp\llmx-vulkan-cache-cleanup-final-evidence-20260925\control-tests.exe"
