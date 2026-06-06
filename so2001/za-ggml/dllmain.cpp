//------------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//
//------------------------------------------------------------------------------

#include <windows.h>

BOOL
APIENTRY
DllMain(
    _In_ HMODULE hModule,
    _In_ DWORD  dwReason,
    _In_opt_ LPVOID /* lpReserved */
    )
{
    switch (dwReason)
    {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hModule);
        break;

    case DLL_PROCESS_DETACH:
        break;

    default:
        break;
    }

    return TRUE;
}

