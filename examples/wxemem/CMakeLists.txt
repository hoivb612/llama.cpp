set(TARGET wxemem)

if (NOT WIN32)
    message(STATUS "wxemem: Windows-only example; skipping on this platform")
    return()
endif()

add_executable(${TARGET} wxemem.cpp)
install(TARGETS ${TARGET} RUNTIME)

# psapi: K32EnumProcesses, K32GetProcessMemoryInfo, K32GetModuleBaseName, K32EnumDeviceDrivers
# advapi32: service control manager (EnumServicesStatusEx, OpenSCManager, ...)
# ntdll: NtQuerySystemInformation
target_link_libraries(${TARGET} PRIVATE psapi advapi32 ntdll)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (MSVC)
    target_compile_definitions(${TARGET} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()
