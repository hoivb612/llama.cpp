#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
#ifndef _GAMING_XBOX    
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
#endif // _GAMING_XBOX

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

#ifndef _GAMING_XBOX    
    SetErrorMode(old_mode);
#endif // _GAMING_XBOX

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
#ifndef _GAMING_XBOX    
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
#endif // _GAMING_XBOX

    void * p = (void *) GetProcAddress(handle, name);

#ifndef _GAMING_XBOX    
    SetErrorMode(old_mode);
#endif // _GAMING_XBOX

    return p;
}

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
