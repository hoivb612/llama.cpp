================================================================================
ggml-dx12 / Xbox Series X (Scarlett, GDKX) cross-compile notes
================================================================================

This is a *prototype* path. It is not cert-clean, has no Microsoft Game
Config, and will not pass any submission gates. It exists to let you sideload
the DX12 backend into a dev-kit Game OS partition and exercise it via xbcmd.

Source layout decision
----------------------
The Xbox build shares ggml-dx12.cpp / shaders/ with the desktop DX12 backend.
Xbox-specific code paths are guarded by `GGML_DX12_XBOX_GDKX` defined by
ggml/src/ggml-dx12/CMakeLists.txt when GGML_XBOX=ON. The output DLL is
renamed to ggml-xbox.dll so you can tell the two apart in a deploy package.

Three platform branches now coexist in ggml-dx12.cpp:
  defined(GGML_DX12_XBOX_GDKX)   -- Scarlett / d3d12_xs.h, single-device init
  defined(_WIN32)                -- desktop Windows / DXGI adapter enum
  else                            -- WSL2 / DXCore adapter enum

What is *not* covered here
--------------------------
  * No common/ or tools/ -- those pull in cpp-httplib, jinja, hnswlib and
    other things that don't cross-compile to the Gaming.Xbox.Scarlett.x64
    partition. The one exception is examples/llm-infer/minslm-cli, which
    links only against libllama and is built when GGML_XBOX is ON. That
    gives you a single self-contained Scarlett EXE that opens a local GGUF
    and prints tok/s.
  * No KV-cache scratch path tuned for Scarlett's memory tiers.
  * No PIX-on-Xbox capture markers.

Prerequisites
-------------
  1. Visual Studio 2022 (17.x) with the GDKX (April 2026 build) workload.
     Verify the Gaming.Xbox.Scarlett.x64 platform appears in:
        Configuration Manager -> Active solution platform -> New...
  2. %GameDK% environment variable set (the GDK installer normally does this).
  3. dxc.exe shipped with the GDK -- the desktop dxc from Windows SDK will
     produce DXIL that the Scarlett D3D12 runtime rejects.

Configure
---------
From a "Microsoft GDK Command Prompt" (which sets %GameDK% and pulls in the
right MSBuild props):

    cd C:\repos\llama.cpp

    cmake -B build.xbox ^
          -G "Visual Studio 17 2022" ^
          -A Gaming.Xbox.Scarlett.x64 ^
          -DCMAKE_TOOLCHAIN_FILE=cmake/xbox-scarlett.toolchain.cmake ^
          -DGGML_XBOX=ON ^
          -DLLAMA_BUILD_TESTS=OFF ^
          -DLLAMA_BUILD_TOOLS=OFF ^
          -DLLAMA_BUILD_COMMON=OFF ^
          -DLLAMA_BUILD_EXAMPLES=ON

Note: LLAMA_BUILD_EXAMPLES=ON is intentional even with LLAMA_BUILD_COMMON=OFF.
The top-level CMakeLists.txt has a special branch for GGML_XBOX that allows
the examples descent without common/, and examples/CMakeLists.txt then
filters down to just llm-infer/, which itself filters down to just the
minslm-cli target on Xbox. End result: one Scarlett EXE + the backend DLLs.

If %GameDK% is not set, also pass:

    -DGGML_XBOX_GDK_EDITION=260400   (or whichever subdir of the GDK root)
    -DGGML_XBOX_SHADER_COMPILER=C:\full\path\to\dxc.exe

Build the backend DLL:

    cmake --build build.xbox --config Release --target ggml-dx12

Build the host EXE:

    cmake --build build.xbox --config Release --target minslm-cli

Artifacts land at:

    build.xbox\bin\Release\ggml-xbox.dll
    build.xbox\bin\Release\minslm-cli.exe   (Scarlett, links libllama statically-ish)

Compiled shader headers land under:

    build.xbox\ggml\src\ggml-dx12\shaders_scarlett\

Sanity check the link before deploying:

    dumpbin /HEADERS build.xbox\bin\Release\ggml-xbox.dll | findstr machine
    dumpbin /DEPENDENTS build.xbox\bin\Release\ggml-xbox.dll
    dumpbin /DEPENDENTS build.xbox\bin\Release\minslm-cli.exe

The "machine" line should report "8664 machine (x64)". The dependents lines
should reference d3d12_xs.dll, xg_xs.dll, xgameruntime.dll -- *not*
d3d12.dll / dxgi.dll. If you see the latter, the Gaming.Xbox.Scarlett.x64
platform did not engage and CMake fell back to a desktop link.

Deploy
------
Lay out a loose-format package:

    deploy\
      MicrosoftGame.config       (point ExecutableList -> Executable Name at minslm-cli.exe)
      minslm-cli.exe
      ggml-xbox.dll
      ggml.dll
      ggml-base.dll
      ggml-cpu.dll
      <model>.gguf
      <prompts>.txt

Then from a GDK command prompt with your dev kit accessible:

    xbapp deploy /x:<dev kit>  /loose deploy
    xbapp launch /x:<dev kit>  <package AUMID>
    xbrun /x:<dev kit> /O minslm-cli.exe G:\<model>.gguf 4 G:\prompts.txt

stdout from the title is piped back over the dev-link to your dev PC.

Open issues / things you will hit
---------------------------------
  * The "telnet to Triangle and run llama-cli alongside it" idea will not
    work as stated -- titles are sandboxed. Triangle plays no role. You need
    your own loose-format package; xbcmd / xbrun is how you drive it from
    the command line.
  * The Scarlett HLSL compiler binary name has drifted across GDK
    revisions. The CMake search list (XboxScarlettShaderCompiler.exe,
    XSCDxc.exe, dxc-xs.exe, dxc.exe under <GDKX>\GXDK\bin\Scarlett\) covers
    the names I have seen; if April 2026 picked a new one, override with
    -DGGML_XBOX_SHADER_COMPILER=<path>.
  * D3D12XBOX_CREATE_DEVICE_PARAMETERS field names occasionally change
    between GDK revisions. The init in dx12_device::init() (Xbox branch) is
    the most likely first build break -- expect to tweak field names against
    the shipped d3d12_xs.h.
  * The cooperative-vector / WaveMMA probes are skipped on Xbox since
    Scarlett is RDNA 2. The auto-tune falls back cleanly to non-MMA paths.
  * The TDR-yield / DWM-yield logic in graph_compute() is a no-op on Xbox
    (no compositor). It does no harm but provides no value either; if you
    hit a Scarlett TDR, the fix is in the Scarlett command-list submission
    cadence, not in this yield.
