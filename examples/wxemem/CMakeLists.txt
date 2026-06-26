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

# ---------------------------------------------------------------------------
# Standalone Windows OS utilities (CPU topology + memory-bandwidth probes).
# These are self-contained Win32 diagnostics used to investigate NUMA / CCX
# behaviour (see topo.c and bwtest.c headers). Windows-only.
#   topo   - dumps LP -> core -> CCX(L3) -> NUMA topology
#   bwtest - measures aggregate read bandwidth vs thread placement
# ---------------------------------------------------------------------------
if (WIN32)
    add_executable(topo topo.c)
    install(TARGETS topo RUNTIME)
    target_compile_definitions(topo PRIVATE
        _WIN32_WINNT=0x0601
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )

    add_executable(bwtest bwtest.c)
    install(TARGETS bwtest RUNTIME)
    target_compile_definitions(bwtest PRIVATE
        _WIN32_WINNT=0x0601
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()
