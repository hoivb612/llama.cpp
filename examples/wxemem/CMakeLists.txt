set(TARGET wxemem)

add_executable(${TARGET} wxemem.cpp)
install(TARGETS ${TARGET} RUNTIME)

if (WIN32)
    # psapi: K32EnumProcesses, K32GetProcessMemoryInfo, K32GetModuleBaseName, K32EnumDeviceDrivers
    # advapi32: service control manager (EnumServicesStatusEx, OpenSCManager, ...)
    # ntdll: NtQuerySystemInformation
    target_link_libraries(${TARGET} PRIVATE psapi advapi32 ntdll)
endif()
# On Linux the collectors read /proc and shell out to systemctl; no extra libs.

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (MSVC)
    target_compile_definitions(${TARGET} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()
