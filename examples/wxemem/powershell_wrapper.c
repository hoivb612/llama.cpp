#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <string.h>

/*
 * powershell.exe wrapper.
 *
 * Launches "pwsh.exe -command <args>" forwarding every argument that was
 * passed to this program, then returns pwsh's exit code.
 */

static const wchar_t *skip_first_arg(const wchar_t *cmd)
{
    const wchar_t *p = cmd;

    /* Skip leading whitespace. */
    while (*p == L' ' || *p == L'\t')
        p++;

    /* Skip the program name (argv[0]) honouring quotes. */
    if (*p == L'"') {
        p++;
        while (*p && *p != L'"')
            p++;
        if (*p == L'"')
            p++;
    } else {
        while (*p && *p != L' ' && *p != L'\t')
            p++;
    }

    /* Skip whitespace separating argv[0] from the rest. */
    while (*p == L' ' || *p == L'\t')
        p++;

    return p;
}

int wmain(void)
{
    const wchar_t *full = GetCommandLineW();
    const wchar_t *rest = skip_first_arg(full);

    const wchar_t *prefix = L"pwsh.exe -command ";
    size_t len = wcslen(prefix) + wcslen(rest) + 1;

    wchar_t *cmdline = (wchar_t *)malloc(len * sizeof(wchar_t));
    if (!cmdline) {
        fwprintf(stderr, L"powershell wrapper: out of memory\n");
        return 1;
    }

    if (*rest)
        _snwprintf(cmdline, len, L"%s%s", prefix, rest);
    else
        _snwprintf(cmdline, len, L"pwsh.exe");

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessW(NULL, cmdline, NULL, NULL, TRUE,
                        0, NULL, NULL, &si, &pi)) {
        fwprintf(stderr, L"powershell wrapper: failed to launch pwsh.exe (error %lu)\n",
                 GetLastError());
        free(cmdline);
        return 1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    free(cmdline);

    return (int)exit_code;
}
