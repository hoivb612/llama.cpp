add_library(phi4mm OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(phi4mm PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(phi4mm PUBLIC .)
target_include_directories(phi4mm PUBLIC ../..)
target_include_directories(phi4mm PUBLIC ../../common)

target_compile_features(phi4mm PRIVATE cxx_std_17)

add_library(phi4mm_static STATIC $<TARGET_OBJECTS:phi4mm>)
if (BUILD_SHARED_LIBS)
    set_target_properties(phi4mm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(phi4mm PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(phi4mm_shared SHARED $<TARGET_OBJECTS:phi4mm>)
    target_link_libraries(phi4mm_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS phi4mm_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(phi4mm PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(phi4mm BUILD_INFO)
endif()

set(TARGET llama-phi4mm-cli)
add_executable(${TARGET} phi4mm-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-phi4mm-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common phi4mm ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
