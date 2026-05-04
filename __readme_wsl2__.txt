==== REM getting all the tools and gdb
 sudo apt-get install build-essential gdb cmake

==== REM For the directx headers (preferred)
 git clone https://github.com/microsoft/DirectX-Headers
 cd DirectX-Headers
 cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
 cmake --build build
 sudo cmake --install build

 AND

 sudo apt install directx-headers-dev <<==== For winadapter.h...

==== REM Dxc shader compiler
 # Install from Microsoft's DirectXShaderCompiler releases (Ubuntu 1.9(dev) failed to compile)
 cd ~/
 mkdir DXC
 wget https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/linux_dxc_2024_07_31.x86_64.tar.gz
 tar xzf linux_dxc_2024_07_31.x86_64.tar.gz
 # Use it
 cmake . -DDXC_EXECUTABLE=~/dxc/bin/dxc
 cmake --build . --config RelWithDebInfo --target ...
 # Check that build/ggml/src/ggml-dx12/shaders/ is populated with *_dxil.h files
 # grep -i "dxc\|DXC" ~/b612/build.dx12/CMakeCache.txt
 #  DXCORE_LIBRARY:FILEPATH=/usr/lib/wsl/lib/libdxcore.so
 #  DXC_EXECUTABLE:FILEPATH=/home/hoiv/dxc/bin/dxc

==== REM D3D libs
 On WSL2, Microsoft mounts the DX12 libraries at:
 /usr/lib/wsl/lib/libd3d12.so
 /usr/lib/wsl/lib/libdxcore.so

==== REM Building for DX12
 cmake .. -DGGML_DX12=ON
 cmake --build . --config RelWithDebInfo -j ($nproc) --target llama-cli llama-server

==== REM Building for Vulkan
 sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers
 vulkaninfo --summary

 or

 # Install the latest Vulkan SDK from LunarG:

 wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
 sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
 sudo apt update
 sudo apt install vulkan-sdk

 # Replace jammy with your Ubuntu version (lsb_release -cs) if different. Then rebuild.

==== REM Running with multiple adapters systems 
==== REM -mg 1 -> pick adapter 1 instead of 0 
==== REM -sm none -> -sm none (or --split-mode none) means load the entire model on a single GPU - no splitting across multiple GPUs. 
    The model goes entirely to the GPU specified by -mg (main GPU, default 0).

 The other options:

 - layer (default) — split layers across GPUs
 - row — split tensor rows across GPUs for more granular distribution
 - tensor — split individual tensors across GPUs

 bin/llama-cli --model /mnt/d/llama.cpp/models/Phi-3/Phi-3-mini-4k-instruct-Q2_K.gguf -ngl 99 -mg 1 -sm none

==== REM Running with GDB
 gdb --args bin/llama-cli --model /mnt/d/llama.cpp/models/Phi-3/Phi-3-mini-4k-instruct-Q2_K.gguf -ngl 99 -sm none -mg 1
 run
 <ctrl-C> brings it back to GDB - "signal SIGINT" passes the <ctrl-C> to the currently running app
