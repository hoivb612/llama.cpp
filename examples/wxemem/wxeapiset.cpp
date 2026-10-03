// Read-only API-set namespace diagnostics for Windows x86/x64.
// Standalone build (in a VS developer prompt):
//   cl /nologo /std:c++17 /EHsc /W4 /MT wxeapiset.cpp
// Usage:
//   wxeapiset ext-ms-win-gdi-font-l1-1-0.dll [--importer user32.dll]
//   wxeapiset --list [--importer user32.dll]
//   wxeapiset --bin C:\path\binary.dll
//   wxeapiset --closure C:\path\binary.exe > dependencies.txt
//   wxeapiset --closure-delay C:\path\binary.exe > potential-dependencies.txt
//   wxeapiset --self-test
//
// Detects v6, v7, and v7 with a v6 compatibility header at runtime.
// V6 supports lookup/list/importer overrides through a bounded schema parser.
// V7 uses ntdll!ApiSetGetImplementationHost and ApiSetQuerySchema, without
// adding a load-time dependency on those exports. Hybrid v7 can list the v6
// compatibility names, resolving each through v7. Pure v7 enumeration and v7
// importer overrides are not exposed by these APIs and are explicitly rejected.
// Uses internal PEB layouts/native APIs, not a stable public Windows API.
// Hosts are schema mappings, not proof of file presence, loadability, exports,
// or functional implementation. No target DLL is loaded or executed.
// --bin enumerates normal/delay modules and named/ordinal symbols without
// executing the input. Uses the tool's PEB, even for a cross-architecture file.
// --closure walks normal imports; --closure-delay also walks potential delay imports.
// Both search CWD then PATH only.
// Stdout is a sorted list of resolved dependency paths (excluding the root).
// Diagnostics/summary go to stderr; nonzero exit means the list is incomplete.
// Does not model LoadLibrary, export forwarders, SxS, KnownDLLs or loader policy.
// Exit codes: 0 = mapped/listed, 1 = absent/no host, 2 = error.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#include <intrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct NamespaceHeader {
    uint32_t version, size, flags, count, entryOffset, hashOffset, hashFactor;
};
struct NamespaceEntry {
    uint32_t flags, nameOffset, nameLength, hashedLength, valueOffset, valueCount;
};
struct ValueEntry {
    uint32_t flags, nameOffset, nameLength, valueOffset, valueLength;
};
struct V7HeaderPrefix {
    uint8_t major, minor, flags, runLevel;
    uint32_t size, reserved0, reserved1;
    uint16_t headerSize, headerOffset;
};
struct SchemaInfo {
    unsigned major;
    unsigned minor;
    bool hybrid;
};
static_assert(sizeof(NamespaceHeader) == 28 && sizeof(NamespaceEntry) == 24 &&
              sizeof(ValueEntry) == 20 && sizeof(V7HeaderPrefix) == 20 &&
              sizeof(wchar_t) == 2, "Unexpected layout");

struct Mapping {
    std::wstring importer;
    std::wstring host;
};
struct Contract {
    std::wstring name;
    std::wstring lookupKey;
    std::vector<Mapping> mappings;
};

std::wstring lower(std::wstring value) {
    for (auto &ch : value) {
        if (ch >= L'A' && ch <= L'Z') ch += L'a' - L'A';
    }
    return value;
}

std::wstring normalize(std::wstring value) {
    value = lower(value);
    if (value.size() >= 4 && value.compare(value.size() - 4, 4, L".dll") == 0) {
        value.resize(value.size() - 4);
    }
    return value;
}

bool validContract(const std::wstring &name) {
    if (name.compare(0, 4, L"api-") != 0 && name.compare(0, 4, L"ext-") != 0) return false;
    return name.size() > 4 && std::all_of(name.begin(), name.end(), [](wchar_t ch) {
        return (ch >= L'a' && ch <= L'z') || (ch >= L'0' && ch <= L'9') ||
               ch == L'-' || ch == L'~';
    });
}

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void range(const std::vector<uint8_t> &bytes, uint32_t offset, uint32_t count, size_t width) {
    require(offset <= bytes.size() && count <= (bytes.size() - offset) / width,
            "Malformed API-set schema: range outside namespace");
}

template<class T>
T read(const std::vector<uint8_t> &bytes, uint32_t offset) {
    range(bytes, offset, 1, sizeof(T));
    T result{};
    std::memcpy(&result, bytes.data() + offset, sizeof(T));
    return result;
}

std::wstring readString(const std::vector<uint8_t> &bytes, uint32_t offset, uint32_t length) {
    require(length % sizeof(wchar_t) == 0, "Malformed API-set schema: odd UTF-16 length");
    range(bytes, offset, length, 1);
    std::wstring result(length / sizeof(wchar_t), L'\0');
    if (length) std::memcpy(result.data(), bytes.data() + offset, length);
    require(result.find(L'\0') == std::wstring::npos, "Malformed API-set schema: embedded NUL");
    return result;
}

std::vector<Contract> parse(const std::vector<uint8_t> &bytes) {
    const auto header = read<NamespaceHeader>(bytes, 0);
    require(header.version == 6, "Unsupported API-set schema version (only v6 is supported)");
    require(header.size == bytes.size(), "Malformed API-set schema: size mismatch");
    range(bytes, header.entryOffset, header.count, sizeof(NamespaceEntry));
    std::vector<Contract> contracts;
    for (uint32_t i = 0; i < header.count; ++i) {
        const auto entry = read<NamespaceEntry>(
            bytes, header.entryOffset + i * static_cast<uint32_t>(sizeof(NamespaceEntry)));
        require(entry.hashedLength && entry.hashedLength <= entry.nameLength &&
                entry.hashedLength % 2 == 0, "Malformed API-set schema: invalid hashed length");
        Contract contract;
        contract.name = lower(readString(bytes, entry.nameOffset, entry.nameLength));
        require(validContract(contract.name), "Unsupported API-set contract name");
        contract.lookupKey = contract.name.substr(0, entry.hashedLength / sizeof(wchar_t));
        range(bytes, entry.valueOffset, entry.valueCount, sizeof(ValueEntry));
        for (uint32_t j = 0; j < entry.valueCount; ++j) {
            const auto value = read<ValueEntry>(
                bytes, entry.valueOffset + j * static_cast<uint32_t>(sizeof(ValueEntry)));
            contract.mappings.push_back({
                lower(readString(bytes, value.nameOffset, value.nameLength)),
                readString(bytes, value.valueOffset, value.valueLength)});
        }
        contracts.push_back(std::move(contract));
    }
    return contracts;
}

void readMemory(uintptr_t address, void *destination, size_t size) {
    SIZE_T copied = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void *>(address),
                           destination, size, &copied) || copied != size) {
        throw std::runtime_error("Cannot read active API-set map; Win32 error " +
                                 std::to_string(GetLastError()));
    }
}

SchemaInfo identify(const std::vector<uint8_t> &prefix) {
    const auto word = read<uint32_t>(prefix, 0);
    if ((word & 0xff) == 7) {
        const auto header = read<V7HeaderPrefix>(prefix, 0);
        return {header.major, header.minor, false};
    }
    if (word == 6) {
        const auto header = read<NamespaceHeader>(prefix, 0);
        if (header.entryOffset > sizeof(NamespaceHeader) &&
            prefix.size() > sizeof(NamespaceHeader) && prefix[sizeof(NamespaceHeader)] == 7) {
            const auto v7 = read<V7HeaderPrefix>(prefix, sizeof(NamespaceHeader));
            return {v7.major, v7.minor, true};
        }
        return {6, 0, false};
    }
    throw std::runtime_error("Unsupported API-set schema version " + std::to_string(word & 0xff));
}

uintptr_t activeMap() {
    // Private PEB.ApiSetMap offsets. Fail at compile time on other architectures.
#if defined(_M_X64)
    const uintptr_t peb = static_cast<uintptr_t>(__readgsqword(0x60));
    constexpr uintptr_t mapOffset = 0x68;
#elif defined(_M_IX86)
    const uintptr_t peb = static_cast<uintptr_t>(__readfsdword(0x30));
    constexpr uintptr_t mapOffset = 0x38;
#else
#error wxeapiset currently supports only Windows x86 and x64
#endif
    uintptr_t map = 0;
    readMemory(peb + mapOffset, &map, sizeof(map));
    require(map != 0, "The current process has no API-set map");
    return map;
}

SchemaInfo identifyLive(uintptr_t map) {
    uint32_t word = 0;
    readMemory(map, &word, sizeof(word));
    if ((word & 0xff) == 7) {
        std::vector<uint8_t> prefix(sizeof(V7HeaderPrefix));
        readMemory(map, prefix.data(), prefix.size());
        return identify(prefix);
    }
    require(word == 6, "Unsupported API-set schema major version (expected 6 or 7)");
    std::vector<uint8_t> prefix(sizeof(NamespaceHeader));
    readMemory(map, prefix.data(), prefix.size());
    const auto header = read<NamespaceHeader>(prefix, 0);
    if (header.entryOffset > sizeof(NamespaceHeader) &&
        header.size >= sizeof(NamespaceHeader) + sizeof(V7HeaderPrefix)) {
        prefix.resize(sizeof(NamespaceHeader) + sizeof(V7HeaderPrefix));
        readMemory(map, prefix.data(), prefix.size());
    }
    return identify(prefix);
}

std::vector<uint8_t> snapshotV6(uintptr_t map) {
    NamespaceHeader header{};
    readMemory(map, &header, sizeof(header));
    if (header.version != 6) {
        throw std::runtime_error("Unsupported API-set schema version " +
                                 std::to_string(header.version) + " (only v6 is supported)");
    }
    // A diagnostic allocation bound, not a claim about the Windows format limit.
    require(header.size >= sizeof(header) && header.size <= 64 * 1024 * 1024,
            "Invalid or unexpectedly large API-set namespace");
    std::vector<uint8_t> bytes(header.size);
    readMemory(map, bytes.data(), bytes.size());
    return bytes;
}

using ResolveHost = LONG (NTAPI *)(PCSTR, PBOOLEAN, PUNICODE_STRING);
using QuerySchema = LONG (NTAPI *)(PCSTR, PULONG);

struct NativeResult {
    bool resolved;
    std::wstring host;
    std::optional<ULONG> availability;
};

std::string statusHex(LONG status) {
    char buffer[16]{};
    std::snprintf(buffer, sizeof(buffer), "0x%08lX", static_cast<ULONG>(status));
    return buffer;
}

NativeResult queryNative(const std::wstring &name, ResolveHost resolve, QuerySchema query) {
    require(resolve != nullptr, "V7 requires ntdll!ApiSetGetImplementationHost; export unavailable");
    require(name.size() <= 32766 && validContract(name), "Invalid or excessively long API-set name");
    std::string ansi;
    ansi.reserve(name.size());
    for (const auto ch : name) ansi.push_back(static_cast<char>(ch)); // Validated ASCII above.
    BOOLEAN resolved = FALSE;
    UNICODE_STRING host{};
    const LONG status = resolve(ansi.c_str(), &resolved, &host);
    if (status < 0) {
        throw std::runtime_error("ApiSetGetImplementationHost failed: NTSTATUS " + statusHex(status));
    }
    require(host.Length % sizeof(wchar_t) == 0 && host.Length <= host.MaximumLength &&
            (!host.Length || (resolved && host.Buffer)), "Native API returned an invalid host string");
    NativeResult result{resolved != FALSE, {}, std::nullopt};
    if (host.Length) {
        result.host.resize(host.Length / sizeof(wchar_t));
        // The returned string is borrowed from the process schema, not heap-owned.
        readMemory(reinterpret_cast<uintptr_t>(host.Buffer), result.host.data(), host.Length);
        require(result.host.find(L'\0') == std::wstring::npos, "Native host string contains an embedded NUL");
    }
    if (query) {
        ULONG availability = 0;
        const LONG queryStatus = query(ansi.c_str(), &availability);
        if (queryStatus < 0) {
            throw std::runtime_error("ApiSetQuerySchema failed: NTSTATUS " + statusHex(queryStatus));
        }
        result.availability = availability;
    }
    return result;
}

template<class T>
T nativeExport(HMODULE module, const char *name) {
    const auto address = GetProcAddress(module, name);
    T function = nullptr;
    static_assert(sizeof(function) == sizeof(address), "Unexpected function pointer size");
    std::memcpy(&function, &address, sizeof(function));
    return function;
}

const wchar_t *availabilityText(ULONG value) {
    switch (value) {
    case 0: return L"implemented";
    case 0xf0: return L"query unsuccessful";
    case 0xf1: return L"not in schema";
    case 0xf2: return L"not hosted";
    case 0xf3: return L"disabled by contract";
    case 0xf4: return L"disabled by run level";
    case 0xf5: return L"disabled by feature state";
    case 0xf6: return L"hash failure";
    default: return L"unknown query result";
    }
}

int printNative(const std::wstring &name, bool list, const std::wstring &importer) {
    require(!list, "V7 --list is unavailable: native APIs cannot enumerate contracts; v7 entries use hashes");
    require(importer.empty(), "V7 --importer is unavailable: native host-query API has no importer parameter");
    const auto module = GetModuleHandleW(L"ntdll.dll");
    require(module != nullptr, "Cannot find the already loaded ntdll.dll");
    const auto result = queryNative(name,
        nativeExport<ResolveHost>(module, "ApiSetGetImplementationHost"),
        nativeExport<QuerySchema>(module, "ApiSetQuerySchema"));
    const bool mapped = result.resolved && !result.host.empty();
    std::wprintf(L"Resolver: ntdll!ApiSetGetImplementationHost (default host)\n");
    std::wprintf(L"%ls -> %ls\n", name.c_str(),
                 mapped ? result.host.c_str() : L"[no host resolved]");
    if (result.availability) {
        std::wprintf(L"Schema availability: %ls (0x%08lX)\n",
                     availabilityText(*result.availability), *result.availability);
    } else {
        std::wprintf(L"Schema availability: not checked (ApiSetQuerySchema export unavailable)\n");
    }
    return mapped ? 0 : 1;
}

const Contract *lookup(const std::vector<Contract> &contracts, const std::wstring &name) {
    // V6 hashes the name through the last '-' (excluding the revision suffix).
    const auto dash = name.find_last_of(L'-');
    if (dash == std::wstring::npos) return nullptr;
    const auto key = name.substr(0, dash);
    const Contract *match = nullptr;
    for (const auto &contract : contracts) {
        if (contract.lookupKey == key) {
            require(match == nullptr, "Ambiguous API-set lookup key");
            match = &contract;
        }
    }
    return match;
}

const Mapping *selectHost(const Contract &contract, const std::wstring &importer) {
    const Mapping *fallback = nullptr;
    const Mapping *selected = nullptr;
    for (const auto &mapping : contract.mappings) {
        if (mapping.importer.empty()) {
            require(fallback == nullptr, "Ambiguous default host");
            fallback = &mapping;
        } else if (!importer.empty() && mapping.importer == importer) {
            require(selected == nullptr, "Ambiguous importer-specific host");
            selected = &mapping;
        }
    }
    return selected ? selected : fallback;
}

bool printContract(const Contract &contract, const std::wstring &importer) {
    const auto selected = selectHost(contract, importer);
    const bool resolved = selected && !selected->host.empty();
    std::wprintf(L"%ls.dll -> %ls\n", contract.name.c_str(),
                 resolved ? selected->host.c_str() : L"[no applicable host]");
    for (const auto &mapping : contract.mappings) {
        std::wprintf(L"  %ls: %ls%ls\n",
                     mapping.importer.empty() ? L"(default)" : mapping.importer.c_str(),
                     mapping.host.empty() ? L"[empty host]" : mapping.host.c_str(),
                     &mapping == selected ? L" [selected]" : L"");
    }
    return resolved;
}

struct ImportModule {
    std::wstring name;
    bool delayed;
    std::vector<std::wstring> symbols;
};

class PeImports {
    const std::vector<uint8_t> &bytes;
    std::vector<IMAGE_SECTION_HEADER> sections;
    uint32_t headersSize = 0;
    uint64_t imageBase = 0;
    bool pe64 = false;

    template<class T>
    T at(size_t offset) const {
        require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset,
                "Malformed PE: truncated structure");
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }

    struct Span { size_t offset, size; };

    Span span(uint64_t rva) const {
        require(rva <= UINT32_MAX, "Malformed PE: RVA overflow");
        if (rva < headersSize) {
            require(headersSize <= bytes.size(), "Malformed PE: truncated headers");
            return {static_cast<size_t>(rva), headersSize - static_cast<size_t>(rva)};
        }
        std::optional<Span> found;
        for (const auto &section : sections) {
            if (rva < section.VirtualAddress) continue;
            const auto delta = rva - section.VirtualAddress;
            if (delta >= std::max(section.Misc.VirtualSize, section.SizeOfRawData)) continue;
            require(!found, "Malformed PE: overlapping sections");
            require(delta < section.SizeOfRawData, "Malformed PE: RVA has no file-backed data");
            const uint64_t offset = section.PointerToRawData + delta;
            const auto length = section.SizeOfRawData - static_cast<size_t>(delta);
            require(offset <= bytes.size() && length <= bytes.size() - offset,
                    "Malformed PE: truncated section data");
            found = Span{static_cast<size_t>(offset), length};
        }
        require(found.has_value(), "Malformed PE: unmapped RVA");
        return *found;
    }

    template<class T>
    T fromRva(uint64_t rva) const {
        const auto data = span(rva);
        require(sizeof(T) <= data.size, "Malformed PE: structure crosses raw-data boundary");
        return at<T>(data.offset);
    }

    std::wstring stringAt(uint64_t rva) const {
        const auto data = span(rva);
        std::wstring value;
        for (size_t i = 0; i < data.size; ++i) {
            const auto ch = bytes[data.offset + i];
            if (!ch) {
                require(!value.empty(), "Malformed PE: empty import name");
                return value;
            }
            require(ch >= 32 && ch < 127, "Malformed PE: non-printable import name");
            value.push_back(ch);
            require(value.size() <= 32766, "Malformed PE: import name too long");
        }
        throw std::runtime_error("Malformed PE: unterminated import name");
    }

    uint64_t addressToRva(uint64_t address, bool isRva) const {
        if (isRva) return address;
        require(address >= imageBase, "Malformed PE: delay-import VA precedes image base");
        return address - imageBase;
    }

    void thunks(ImportModule &module, uint64_t table, bool isRva) const {
        require(table != 0, "Malformed PE: missing import name table");
        const auto data = span(table);
        const size_t width = pe64 ? 8 : 4;
        for (size_t i = 0; i < data.size / width; ++i) {
            const uint64_t value = pe64 ? at<uint64_t>(data.offset + i * width) :
                                         at<uint32_t>(data.offset + i * width);
            if (!value) return;
            const uint64_t ordinalFlag = pe64 ? IMAGE_ORDINAL_FLAG64 : IMAGE_ORDINAL_FLAG32;
            if (value & ordinalFlag) {
                require((value & ~(ordinalFlag | 0xffffULL)) == 0, "Malformed PE: invalid ordinal thunk");
                module.symbols.push_back(L"#" + std::to_wstring(value & 0xffff));
            } else {
                const auto name = addressToRva(value, isRva);
                (void)fromRva<uint16_t>(name);
                module.symbols.push_back(stringAt(name + sizeof(uint16_t)));
            }
        }
        throw std::runtime_error("Malformed PE: unterminated import thunk table");
    }

    void directory(IMAGE_DATA_DIRECTORY dir, bool delayed) {
        if (!dir.VirtualAddress && !dir.Size) return;
        require(dir.VirtualAddress && dir.Size, "Malformed PE: incomplete import directory");
        const size_t width = delayed ? 32 : sizeof(IMAGE_IMPORT_DESCRIPTOR);
        require(dir.Size >= width, "Malformed PE: short import directory");
        for (uint64_t offset = 0; offset + width <= dir.Size; offset += width) {
            uint64_t name = 0, table = 0;
            bool isRva = true;
            if (delayed) {
                struct DelayDescriptor {
                    uint32_t attributes, name, handle, iat, names, bound, unload, timestamp;
                };
                const auto d = fromRva<DelayDescriptor>(dir.VirtualAddress + offset);
                if (!(d.attributes | d.name | d.handle | d.iat | d.names | d.bound | d.unload | d.timestamp)) return;
                require((d.attributes & ~1U) == 0, "Unsupported PE delay-import attributes");
                isRva = (d.attributes & 1) != 0;
                require(d.name && d.names, "Malformed PE: missing delay-import name or table");
                name = addressToRva(d.name, isRva);
                table = addressToRva(d.names, isRva);
            } else {
                const auto d = fromRva<IMAGE_IMPORT_DESCRIPTOR>(dir.VirtualAddress + offset);
                if (!(d.OriginalFirstThunk | d.TimeDateStamp | d.ForwarderChain | d.Name | d.FirstThunk)) return;
                require(d.Name != 0, "Malformed PE: missing import module name");
                require(d.OriginalFirstThunk || !d.TimeDateStamp,
                        "Cannot enumerate a bound import without its original name table");
                name = d.Name;
                table = d.OriginalFirstThunk ? d.OriginalFirstThunk : d.FirstThunk;
            }
            ImportModule module{stringAt(name), delayed, {}};
            thunks(module, table, isRva);
            modules.push_back(std::move(module));
        }
        throw std::runtime_error("Malformed PE: unterminated import descriptor table");
    }

public:
    uint16_t machine = 0;
    std::vector<ImportModule> modules;

    explicit PeImports(const std::vector<uint8_t> &input, bool includeDelay = true) : bytes(input) {
        const auto dos = at<IMAGE_DOS_HEADER>(0);
        require(dos.e_magic == IMAGE_DOS_SIGNATURE && dos.e_lfanew >= 0, "Not a DOS/PE binary");
        const auto nt = static_cast<size_t>(dos.e_lfanew);
        require(at<uint32_t>(nt) == IMAGE_NT_SIGNATURE, "Not a PE binary");
        const auto file = at<IMAGE_FILE_HEADER>(nt + 4);
        machine = file.Machine;
        const auto optional = nt + 4 + sizeof(file);
        const auto magic = at<uint16_t>(optional);
        require(magic == IMAGE_NT_OPTIONAL_HDR32_MAGIC || magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC,
                "Unsupported PE optional-header format");
        pe64 = magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        require((machine == IMAGE_FILE_MACHINE_I386 && !pe64) ||
                (machine == IMAGE_FILE_MACHINE_AMD64 && pe64), "Only x86/x64 PE binaries are supported");
        const size_t directoriesOffset = pe64 ? offsetof(IMAGE_OPTIONAL_HEADER64, DataDirectory) :
                                                offsetof(IMAGE_OPTIONAL_HEADER32, DataDirectory);
        require(file.SizeOfOptionalHeader >= directoriesOffset, "Malformed PE: short optional header");
        const auto count = at<uint32_t>(optional + directoriesOffset - 4);
        require(count <= (file.SizeOfOptionalHeader - directoriesOffset) / sizeof(IMAGE_DATA_DIRECTORY),
                "Malformed PE: truncated data directories");
        headersSize = at<uint32_t>(optional + offsetof(IMAGE_OPTIONAL_HEADER32, SizeOfHeaders));
        imageBase = pe64 ? at<uint64_t>(optional + offsetof(IMAGE_OPTIONAL_HEADER64, ImageBase)) :
                           at<uint32_t>(optional + offsetof(IMAGE_OPTIONAL_HEADER32, ImageBase));
        const auto sectionTable = optional + file.SizeOfOptionalHeader;
        require(sectionTable <= headersSize && headersSize <= bytes.size() &&
                file.NumberOfSections <= (headersSize - sectionTable) / sizeof(IMAGE_SECTION_HEADER),
                "Malformed PE: truncated section table");
        for (size_t i = 0; i < file.NumberOfSections; ++i) {
            sections.push_back(at<IMAGE_SECTION_HEADER>(sectionTable + i * sizeof(IMAGE_SECTION_HEADER)));
        }
        for (const auto index : {IMAGE_DIRECTORY_ENTRY_IMPORT, IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT}) {
            if (index == IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT && !includeDelay) continue;
            if (static_cast<uint32_t>(index) < count) {
                directory(at<IMAGE_DATA_DIRECTORY>(optional + directoriesOffset +
                          index * sizeof(IMAGE_DATA_DIRECTORY)), index == IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT);
            }
        }
    }
};

std::vector<uint8_t> readBinary(const wchar_t *path) {
    FILE *raw = nullptr;
    require(_wfopen_s(&raw, path, L"rb") == 0, "Cannot open input binary for reading");
    const std::unique_ptr<FILE, decltype(&std::fclose)> file(raw, std::fclose);
    require(_fseeki64(file.get(), 0, SEEK_END) == 0, "Cannot seek input binary");
    const auto length = _ftelli64(file.get());
    require(length > 0 && length <= 512LL * 1024 * 1024, "Input binary must be between 1 byte and 512 MiB");
    require(_fseeki64(file.get(), 0, SEEK_SET) == 0, "Cannot rewind input binary");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    require(std::fread(bytes.data(), 1, bytes.size(), file.get()) == bytes.size(), "Cannot read input binary");
    return bytes;
}

int printBinary(const wchar_t *path, uintptr_t map, const SchemaInfo &schema) {
    const auto bytes = readBinary(path);
    const PeImports pe(bytes);
    std::wstring importer = path;
    importer = lower(importer.substr(importer.find_last_of(L"\\/") + 1));
    std::wprintf(L"Binary: %ls | Machine: %ls\n", path, pe.machine == IMAGE_FILE_MACHINE_I386 ? L"x86" : L"x64");
    const bool sameArch = (pe.machine == IMAGE_FILE_MACHINE_AMD64) == (sizeof(void *) == 8);
    if (!sameArch) {
        std::wprintf(L"WARNING: binary and tool architectures differ. Resolutions use this tool's PEB;\n"
                     L"run the matching-architecture tool for the intended process view.\n");
    }
    std::wprintf(L"Direct imports only; no recursive dependency, export, or DLL loading checks.\n");
    std::vector<Contract> contracts;
    if (schema.major == 6) {
        contracts = parse(snapshotV6(map));
        std::wprintf(L"V6 importer context: %ls (input filename; rename can affect overrides).\n", importer.c_str());
    } else {
        std::wprintf(L"V7 uses DEFAULT hosts only; importer-specific overrides are not evaluated.\n");
    }
    size_t symbols = 0, contractCount = 0, missing = 0, normal = 0, delayed = 0;
    bool errors = false;
    for (const auto &module : pe.modules) {
        module.delayed ? ++delayed : ++normal;
        symbols += module.symbols.size();
        std::wprintf(L"\n[%ls] %ls (%zu symbols)\n",
                     module.delayed ? L"delay" : L"normal", module.name.c_str(), module.symbols.size());
        const auto name = normalize(module.name);
        if (name.compare(0, 4, L"api-") == 0 || name.compare(0, 4, L"ext-") == 0) {
            ++contractCount;
            try {
                require(validContract(name), "Invalid imported API-set contract name");
                if (schema.major == 7) {
                    if (printNative(name, false, {}) != 0) ++missing;
                } else {
                    const auto contract = lookup(contracts, name);
                    if (!contract) {
                        std::wprintf(L"  -> [absent from active schema]\n");
                        ++missing;
                    } else if (!printContract(*contract, importer)) {
                        ++missing;
                    }
                }
            } catch (const std::exception &error) {
                std::fflush(stdout);
                std::fprintf(stderr, "wxeapiset: resolving %ls: %s\n", module.name.c_str(), error.what());
                errors = true;
            }
        } else {
            std::wprintf(L"  -> %ls [ordinary DLL; not checked]\n", module.name.c_str());
        }
        for (const auto &symbol : module.symbols) std::wprintf(L"    %ls\n", symbol.c_str());
    }
    std::wprintf(L"\nImport summary: %zu normal modules, %zu delay modules, %zu symbols, "
                 L"%zu contract imports, %zu unresolved contracts.\n",
                 normal, delayed, symbols, contractCount, missing);
    return errors ? 2 : missing ? 1 : 0;
}

std::runtime_error win32Error(const char *operation, DWORD error = GetLastError()) {
    return std::runtime_error(std::string(operation) + ": Win32 error " + std::to_string(error));
}

std::wstring fullPath(const std::wstring &path) {
    const auto fail = [&](DWORD error) {
        std::fwprintf(stderr, L"wxeapiset: cannot normalize path: \"%ls\"\n", path.c_str());
        throw win32Error("GetFullPathName", error);
    };
    const auto size = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
    if (!size) fail(GetLastError());
    std::wstring result(size, L'\0');
    const auto copied = GetFullPathNameW(path.c_str(), size, result.data(), nullptr);
    if (!copied) fail(GetLastError());
    if (copied >= size) fail(ERROR_INSUFFICIENT_BUFFER);
    result.resize(copied);
    return result;
}

struct CloseHandleDeleter {
    void operator()(void *handle) const { CloseHandle(handle); }
};

std::optional<std::wstring> existingFile(const std::wstring &path) {
    const auto raw = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (raw == INVALID_HANDLE_VALUE) {
        const auto error = GetLastError();
        if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return std::nullopt;
        throw win32Error("Cannot inspect dependency", error);
    }
    const std::unique_ptr<void, CloseHandleDeleter> handle(raw);
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(raw, &info)) throw win32Error("GetFileInformationByHandle");
    if (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) return std::nullopt;
    const auto size = GetFinalPathNameByHandleW(raw, nullptr, 0, FILE_NAME_NORMALIZED);
    if (!size) throw win32Error("GetFinalPathNameByHandle");
    std::wstring result(static_cast<size_t>(size) + 1, L'\0');
    const auto copied = GetFinalPathNameByHandleW(raw, result.data(), static_cast<DWORD>(result.size()),
                                                 FILE_NAME_NORMALIZED);
    if (!copied || copied >= result.size()) throw win32Error("GetFinalPathNameByHandle");
    result.resize(copied);
    return result;
}

std::wstring displayPath(const std::wstring &path) {
    if (path.compare(0, 8, L"\\\\?\\UNC\\") == 0) return L"\\\\" + path.substr(8);
    if (path.compare(0, 4, L"\\\\?\\") == 0) return path.substr(4);
    return path;
}

struct PathLess {
    bool operator()(const std::wstring &left, const std::wstring &right) const {
        const auto result = CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE);
        if (!result) throw win32Error("CompareStringOrdinal");
        return result == CSTR_LESS_THAN;
    }
};

class DependencySearch {
    std::vector<std::wstring> directories;
public:
    DependencySearch() {
        directories.push_back(fullPath(L"."));
        SetLastError(ERROR_SUCCESS);
        const auto size = GetEnvironmentVariableW(L"PATH", nullptr, 0);
        if (!size) {
            const auto error = GetLastError();
            if (error == ERROR_ENVVAR_NOT_FOUND || error == ERROR_SUCCESS) return;
            throw win32Error("Read PATH", error);
        }
        std::wstring path(size, L'\0');
        SetLastError(ERROR_SUCCESS);
        const auto copied = GetEnvironmentVariableW(L"PATH", path.data(), size);
        if (!copied && GetLastError() == ERROR_SUCCESS) return;
        if (!copied || copied >= size) throw win32Error("Read PATH");
        path.resize(copied);
        size_t start = 0;
        do {
            const auto end = path.find(L';', start);
            auto part = path.substr(start, end == std::wstring::npos ? end : end - start);
            const auto first = part.find_first_not_of(L" \t\r\n");
            if (first == std::wstring::npos) {
                part.clear();
            } else {
                part = part.substr(first, part.find_last_not_of(L" \t\r\n") - first + 1);
            }
            if (part.size() >= 2 && part.front() == L'"' && part.back() == L'"') {
                part = part.substr(1, part.size() - 2);
            }
            if (part.find_first_not_of(L" \t\r\n") == std::wstring::npos) part.clear();
            if (part.find(L'"') != std::wstring::npos) {
                std::fwprintf(stderr, L"wxeapiset: malformed quoted PATH entry: %ls\n", part.c_str());
                throw std::runtime_error("Malformed quoted PATH entry");
            }
            if (!part.empty()) directories.push_back(fullPath(part));
            if (end == std::wstring::npos) break;
            start = end + 1;
        } while (start <= path.size());
    }

    std::optional<std::wstring> find(const std::wstring &name) const {
        // Do not substitute Windows loader search rules for the requested CWD/PATH order.
        require(!name.empty() && name != L"." && name != L".." &&
                name.find_first_of(L"\\/:*?\"<>|") == std::wstring::npos,
                "Dependency must be a module base name for CWD/PATH search");
        for (const auto &directory : directories) {
            const auto candidate = directory + L"\\" + name;
            try {
                if (auto found = existingFile(candidate)) return found;
            } catch (const std::exception &error) {
                std::fwprintf(stderr, L"wxeapiset: search candidate: %ls\n", candidate.c_str());
                throw std::runtime_error(error.what());
            }
        }
        return std::nullopt;
    }
};

class ClosureResolver {
    const SchemaInfo schema;
    std::vector<Contract> contracts;
    ResolveHost nativeResolve = nullptr;
    QuerySchema nativeQuery = nullptr;
public:
    ClosureResolver(uintptr_t map, const SchemaInfo &info) : schema(info) {
        if (schema.major == 6) {
            contracts = parse(snapshotV6(map));
        } else {
            const auto module = GetModuleHandleW(L"ntdll.dll");
            require(module != nullptr, "Cannot find the already loaded ntdll.dll");
            nativeResolve = nativeExport<ResolveHost>(module, "ApiSetGetImplementationHost");
            nativeQuery = nativeExport<QuerySchema>(module, "ApiSetQuerySchema");
        }
    }

    NativeResult resolve(const std::wstring &input, const std::wstring &importer) const {
        std::wstring host = input;
        std::set<std::wstring> seen;
        std::optional<ULONG> availability;
        for (;;) {
            const auto name = normalize(host);
            if (name.compare(0, 4, L"api-") != 0 && name.compare(0, 4, L"ext-") != 0) {
                return {true, host, availability};
            }
            require(validContract(name), "Invalid imported API-set contract name");
            require(seen.insert(name).second && seen.size() <= 128, "API-set host redirection cycle/limit");
            if (schema.major == 7) {
                const auto result = queryNative(name, nativeResolve, nativeQuery);
                if (result.availability && *result.availability != 0) availability = result.availability;
                if (!result.resolved || result.host.empty()) return {false, {}, result.availability};
                host = result.host;
            } else {
                const auto contract = lookup(contracts, name);
                const auto mapping = contract ? selectHost(*contract, importer) : nullptr;
                if (!mapping || mapping->host.empty()) return {false, {}, std::nullopt};
                host = mapping->host;
            }
        }
    }
};

int printClosure(const wchar_t *path, uintptr_t map, const SchemaInfo &schema, bool includeDelay) {
    const DependencySearch search;
    const std::wstring input = path;
    const auto root = input.find_first_of(L"\\/:") == std::wstring::npos ?
                      search.find(input) : existingFile(fullPath(input));
    require(root.has_value(), "Root binary not found (CWD then PATH, or supplied explicit path)");
    const ClosureResolver resolver(map, schema);
    std::set<std::wstring, PathLess> visited{*root}, dependencies;
    std::vector<std::wstring> pending{*root};
    size_t missing = 0, errors = 0, unavailable = 0;
    uint16_t machine = 0;
    std::fwprintf(stderr, L"wxeapiset: closure using %ls process PEB, schema %u.%u%ls; CWD then PATH.\n"
                  L"%ls Excludes dynamic loads/export forwarders;\n"
                  L"does not reproduce KnownDLLs/SxS/Windows loader policy or verify exports/loadability.\n",
                  sizeof(void *) == 8 ? L"x64" : L"x86", schema.major, schema.minor,
                  schema.hybrid ? L" (v6 compatibility header)" : L"",
                  includeDelay ? L"Includes normal and potential delay dependencies." :
                                 L"Normal imports only; excludes delay dependencies.");
    if (schema.major == 7) {
        std::fwprintf(stderr, L"V7 default hosts only; importer-specific overrides are not evaluated.\n");
    }
    for (size_t index = 0; index < pending.size(); ++index) {
        const auto current = pending[index];
        try {
            const auto bytes = readBinary(current.c_str());
            const PeImports pe(bytes, includeDelay);
            if (index == 0) {
                machine = pe.machine;
                if ((machine == IMAGE_FILE_MACHINE_AMD64) != (sizeof(void *) == 8)) {
                    std::fwprintf(stderr, L"WARNING: root binary is %ls, tool is %ls. Continuing with the tool's "
                                  L"PEB schema and filesystem view, not the root binary's process view.\n",
                                  machine == IMAGE_FILE_MACHINE_AMD64 ? L"x64" : L"x86",
                                  sizeof(void *) == 8 ? L"x64" : L"x86");
                }
            } else if (pe.machine != machine) {
                throw std::runtime_error(std::string("Architecture mismatch: dependency is ") +
                    (pe.machine == IMAGE_FILE_MACHINE_AMD64 ? "x64" : "x86") +
                    ", root binary requires " + (machine == IMAGE_FILE_MACHINE_AMD64 ? "x64" : "x86"));
            }
            const auto importer = lower(current.substr(current.find_last_of(L'\\') + 1));
            for (const auto &module : pe.modules) {
                try {
                    const auto resolved = resolver.resolve(module.name, importer);
                    if (!resolved.resolved || resolved.host.empty()) {
                        ++missing;
                        std::fwprintf(stderr, L"UNRESOLVED [%ls] %ls <- %ls\n",
                                      module.delayed ? L"delay" : L"normal", module.name.c_str(),
                                      displayPath(current).c_str());
                        continue;
                    }
                    if (resolved.availability && *resolved.availability != 0) {
                        ++unavailable;
                        std::fwprintf(stderr, L"UNAVAILABLE %ls: %ls (0x%08lX) <- %ls\n",
                                      module.name.c_str(), availabilityText(*resolved.availability),
                                      *resolved.availability, displayPath(current).c_str());
                    }
                    const auto found = search.find(resolved.host);
                    if (!found) {
                        ++missing;
                        std::fwprintf(stderr, L"MISSING [%ls] %ls (import %ls) <- %ls\n",
                                      module.delayed ? L"delay" : L"normal", resolved.host.c_str(),
                                      module.name.c_str(), displayPath(current).c_str());
                        continue;
                    }
                    if (visited.insert(*found).second) {
                        require(visited.size() <= 10000, "Dependency traversal limit exceeded (10000 files)");
                        dependencies.insert(*found);
                        pending.push_back(*found);
                    }
                } catch (const std::exception &error) {
                    ++errors;
                    std::fwprintf(stderr, L"ERROR resolving %ls <- %ls: %hs\n",
                                  module.name.c_str(), displayPath(current).c_str(), error.what());
                }
            }
        } catch (const std::exception &error) {
            ++errors;
            std::fwprintf(stderr, L"ERROR reading %ls: %hs\n", displayPath(current).c_str(), error.what());
            if (index == 0) break;
        }
    }
    for (const auto &dependency : dependencies) std::wprintf(L"%ls\n", displayPath(dependency).c_str());
    const int result = errors ? 2 : (missing || unavailable) ? 1 : 0;
    std::fwprintf(stderr, L"Closure %ls: %zu dependency files (root excluded), %zu missing/unresolved edges, "
                  L"%zu unavailable contracts, %zu errors.\n",
                  result ? L"INCOMPLETE" : L"complete for the stated import/search model",
                  dependencies.size(), missing, unavailable, errors);
    return result;
}

void usage() {
    std::wprintf(
        L"Usage: wxeapiset <api-/ext- contract[.dll]> [--importer module.dll]\n"
        L"       wxeapiset --list [--importer module.dll]\n"
        L"       wxeapiset --bin <EXE/DLL path>\n"
        L"       wxeapiset --closure <EXE/DLL path>\n"
        L"       wxeapiset --closure-delay <EXE/DLL path>\n"
        L"       wxeapiset --self-test\n"
        L"Auto-detects active API-set v6/v7 (including v7 with a v6 header).\n"
        L"V6: parsed lookup/list/importer mappings. V7: native default-host query.\n"
        L"Hybrid v7 --list enumerates compatibility names only, resolved by v7.\n"
        L"Pure v7 --list and all v7 --importer queries are unavailable.\n"
        L"Does not load host DLLs.\n"
        L"--bin lists direct normal/delay DLLs and named/ordinal symbols, resolving contracts.\n"
        L"--closure recursively walks normal imports only (no delay-load imports).\n"
        L"--closure-delay recursively walks both normal and potential delay-load imports.\n"
        L"Both closure modes search CWD, then PATH only.\n"
        L"PATH entries allow surrounding whitespace/quotes; blank entries are ignored.\n"
        L"Stdout: unique sorted dependency paths, root excluded. Diagnostics: stderr.\n"
        L"Closure allows cross-architecture input with a warning; dependencies must match the root.\n"
        L"Closure does not model dynamic loads,\n"
        L"export forwarders, KnownDLLs, SxS, or actual Windows loader search policy.\n"
        L"Uses the tool's PEB, not an offline schema; prefer matching x86/x64 architecture.\n"
        L"Normal WOW64 filesystem redirection applies: x86 System32 paths access SysWOW64;\n"
        L"use Sysnative from an x86 process when inspecting native System32 binaries.\n"
        L"Mapping does not establish file presence or functional availability.\n"
        L"Without --importer, shows the default mapping and all alternatives.\n"
        L"Exit codes: 0=mapped/listed, 1=absent/no host/incomplete closure, 2=error.\n");
}

LONG NTAPI fakeResolve(PCSTR name, PBOOLEAN resolved, PUNICODE_STRING host) {
    static wchar_t value[] = L"v7host.dll";
    *resolved = FALSE;
    *host = {};
    if (std::strcmp(name, "api-test-error") == 0) return static_cast<LONG>(0xc000000dUL);
    if (std::strcmp(name, "api-test-empty") == 0) { *resolved = TRUE; return 0; }
    if (std::strcmp(name, "api-test-missing") == 0) return 0;
    *resolved = TRUE;
    host->Buffer = value;
    host->Length = sizeof(value) - sizeof(wchar_t);
    host->MaximumLength = host->Length;
    if (std::strcmp(name, "api-test-malformed") == 0) ++host->Length;
    return 0;
}

LONG NTAPI fakeQuery(PCSTR name, PULONG result) {
    if (std::strcmp(name, "api-test-query-error") == 0) return static_cast<LONG>(0xc000000dUL);
    *result = std::strcmp(name, "api-test-disabled") == 0 ? 0xf5 : 0;
    return 0;
}

void peSelfTest() {
    for (const bool pe64 : {false, true}) {
        std::vector<uint8_t> bytes(0x600);
        auto put = [&](size_t offset, const auto &value) {
            std::memcpy(bytes.data() + offset, &value, sizeof(value));
        };
        IMAGE_DOS_HEADER dos{};
        dos.e_magic = IMAGE_DOS_SIGNATURE;
        dos.e_lfanew = 0x80;
        put(0, dos);
        put(0x80, uint32_t{IMAGE_NT_SIGNATURE});
        IMAGE_FILE_HEADER file{};
        file.Machine = pe64 ? IMAGE_FILE_MACHINE_AMD64 : IMAGE_FILE_MACHINE_I386;
        file.NumberOfSections = 1;
        file.SizeOfOptionalHeader = pe64 ? sizeof(IMAGE_OPTIONAL_HEADER64) : sizeof(IMAGE_OPTIONAL_HEADER32);
        put(0x84, file);
        auto fillHeader = [](auto &header) {
            header.ImageBase = 0x10000000;
            header.SizeOfHeaders = 0x200;
            header.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
            header.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {0x1000, 40};
            header.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT] = {0x1040, 64};
        };
        if (pe64) {
            IMAGE_OPTIONAL_HEADER64 header{};
            header.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
            fillHeader(header);
            put(0x98, header);
        } else {
            IMAGE_OPTIONAL_HEADER32 header{};
            header.Magic = IMAGE_NT_OPTIONAL_HDR32_MAGIC;
            fillHeader(header);
            put(0x98, header);
        }
        IMAGE_SECTION_HEADER section{};
        section.VirtualAddress = 0x1000;
        section.Misc.VirtualSize = 0x400;
        section.SizeOfRawData = 0x400;
        section.PointerToRawData = 0x200;
        put(0x98 + file.SizeOfOptionalHeader, section);
        IMAGE_IMPORT_DESCRIPTOR descriptor{};
        descriptor.Name = 0x1080;
        descriptor.OriginalFirstThunk = 0x1100;
        put(0x200, descriptor);
        put(0x240, uint32_t{1});
        put(0x244, uint32_t{0x10c0});
        put(0x250, uint32_t{0x1100});
        const char normalName[] = "api-test-fixture-l1-1-0.dll";
        const char delayedName[] = "ordinary.dll";
        const char symbolName[] = "ExampleFunction";
        std::memcpy(bytes.data() + 0x280, normalName, sizeof(normalName));
        std::memcpy(bytes.data() + 0x2c0, delayedName, sizeof(delayedName));
        std::memcpy(bytes.data() + 0x342, symbolName, sizeof(symbolName));
        if (pe64) {
            put(0x300, uint64_t{0x1140});
            put(0x308, uint64_t{IMAGE_ORDINAL_FLAG64 | 17});
        } else {
            put(0x300, uint32_t{0x1140});
            put(0x304, uint32_t{IMAGE_ORDINAL_FLAG32 | 17});
        }
        const PeImports imports(bytes);
        require(imports.modules.size() == 2 && !imports.modules[0].delayed && imports.modules[1].delayed &&
                imports.modules[0].name == L"api-test-fixture-l1-1-0.dll" &&
                imports.modules[1].name == L"ordinary.dll" &&
                imports.modules[0].symbols == std::vector<std::wstring>{L"ExampleFunction", L"#17"} &&
                imports.modules[1].symbols == imports.modules[0].symbols, "PE import fixture failed");
        // Legacy delay descriptors and name thunks use VAs rather than RVAs.
        put(0x240, uint32_t{0});
        put(0x244, uint32_t{0x100010c0});
        put(0x250, uint32_t{0x10001180});
        if (pe64) put(0x380, uint64_t{0x10001140});
        else put(0x380, uint32_t{0x10001140});
        require(PeImports(bytes).modules[1].symbols == std::vector<std::wstring>{L"ExampleFunction"},
                "PE legacy VA delay import test failed");
        for (int test = 0; test < 6; ++test) {
            auto broken = bytes;
            if (test == 0) broken.resize(30);
            if (test == 1) broken[0] = 0;
            if (test == 2) {
                const uint32_t badRva = 0xfffffff0;
                std::memcpy(broken.data() + 0x20c, &badRva, sizeof(badRva));
            }
            if (test == 3) {
                std::memset(broken.data() + 0x342, 'x', broken.size() - 0x342);
            }
            if (test == 4) {
                const uint32_t badTable = 0x13ff;
                std::memcpy(broken.data() + 0x200, &badTable, sizeof(badTable));
            }
            if (test == 5) broken[0x240] = 2;
            bool rejected = false;
            try { (void)PeImports(broken); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "Malformed PE accepted");
        }
        std::memset(bytes.data() + 0x200, 0, 40);
        std::memset(bytes.data() + 0x240, 0, 64);
        require(PeImports(bytes).modules.empty(), "PE empty imports test failed");
    }
}

void selfTest() {
    peSelfTest();
    std::vector<uint8_t> bytes(sizeof(NamespaceHeader) + sizeof(NamespaceEntry) + 2 * sizeof(ValueEntry));
    auto append = [&](const std::wstring &s) {
        const auto offset = static_cast<uint32_t>(bytes.size());
        const auto start = reinterpret_cast<const uint8_t *>(s.data());
        bytes.insert(bytes.end(), start, start + s.size() * sizeof(wchar_t));
        return offset;
    };
    const std::wstring name = L"ext-ms-test-example-l1-1-0";
    NamespaceEntry entry{0, append(name), static_cast<uint32_t>(name.size() * 2),
                         static_cast<uint32_t>(name.find_last_of(L'-') * 2),
                         sizeof(NamespaceHeader) + sizeof(NamespaceEntry), 2};
    ValueEntry fallback{0, 0, 0, append(L"host.dll"), 16};
    ValueEntry alias{0, append(L"client.dll"), 20, append(L"special.dll"), 22};
    NamespaceHeader header{6, static_cast<uint32_t>(bytes.size()), 0, 1, sizeof(NamespaceHeader), 0, 0};
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + header.entryOffset, &entry, sizeof(entry));
    std::memcpy(bytes.data() + entry.valueOffset, &fallback, sizeof(fallback));
    std::memcpy(bytes.data() + entry.valueOffset + sizeof(fallback), &alias, sizeof(alias));
    require(identify(bytes).major == 6 && !identify(bytes).hybrid, "V6 detection test failed");
    const auto contracts = parse(bytes);
    const auto found = lookup(contracts, normalize(L"EXT-MS-TEST-EXAMPLE-L1-1-0.DLL"));
    require(found && selectHost(*found, L"")->host == L"host.dll", "Default lookup test failed");
    require(selectHost(*found, L"client.dll")->host == L"special.dll", "Importer test failed");
    require(selectHost(*found, L"other.dll")->host == L"host.dll", "Fallback test failed");
    require(lookup(contracts, L"ext-ms-test-example-l1-1-9") == found, "V6 revision test failed");
    require(!lookup(contracts, L"ext-ms-test-missing-l1-1-0"), "Absent contract test failed");
    Contract empty{L"ext-ms-test-l1-1-0", L"ext-ms-test-l1-1", {{L"", L""}}};
    require(selectHost(empty, L"")->host.empty(), "Empty host test failed");
    empty.mappings.clear();
    require(!selectHost(empty, L""), "No values test failed");
    empty.mappings.push_back({L"client.dll", L"special.dll"});
    require(!selectHost(empty, L""), "Alias-only test failed");
    for (int test = 0; test != 5; ++test) {
        auto broken = bytes;
        if (test == 0) broken.resize(8);
        if (test == 1) broken[0] = 7;
        if (test == 2) {
            auto bad = entry;
            bad.nameOffset = UINT32_MAX;
            std::memcpy(broken.data() + header.entryOffset, &bad, sizeof(bad));
        }
        if (test == 3) {
            auto bad = entry;
            bad.nameLength |= 1;
            std::memcpy(broken.data() + header.entryOffset, &bad, sizeof(bad));
        }
        if (test == 4) {
            auto bad = entry;
            bad.valueCount = UINT32_MAX;
            std::memcpy(broken.data() + header.entryOffset, &bad, sizeof(bad));
        }
        bool rejected = false;
        try { (void)parse(broken); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "Malformed schema accepted");
    }
    V7HeaderPrefix v7{7, 0, 0x19, 4, 4096, 0, 0, 120, 0};
    std::vector<uint8_t> v7Bytes(sizeof(v7));
    std::memcpy(v7Bytes.data(), &v7, sizeof(v7));
    require(identify(v7Bytes).major == 7 && !identify(v7Bytes).hybrid, "V7 detection test failed");
    auto hybrid = bytes;
    auto compat = header;
    compat.entryOffset = sizeof(header) + 120;
    v7.headerOffset = sizeof(header);
    std::memcpy(hybrid.data(), &compat, sizeof(compat));
    std::memcpy(hybrid.data() + sizeof(header), &v7, sizeof(v7));
    require(identify(hybrid).major == 7 && identify(hybrid).hybrid, "Hybrid detection test failed");
    const auto native = queryNative(L"api-test-example~group", fakeResolve, fakeQuery);
    require(native.resolved && native.host == L"v7host.dll" && native.availability == 0UL,
            "Native host query test failed");
    require(queryNative(L"api-test-empty", fakeResolve, fakeQuery).host.empty(), "Native empty host test failed");
    require(!queryNative(L"api-test-missing", fakeResolve, fakeQuery).resolved, "Native absent test failed");
    require(queryNative(L"api-test-disabled", fakeResolve, fakeQuery).availability == 0xf5UL,
            "Native disabled query test failed");
    require(!queryNative(L"api-test-example", fakeResolve, nullptr).availability,
            "Missing availability API test failed");
    for (const auto testName : {L"api-test-error", L"api-test-malformed", L"api-test-query-error"}) {
        bool rejected = false;
        try { (void)queryNative(testName, fakeResolve, fakeQuery); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "Invalid native result accepted");
    }
    bool rejected = false;
    try { (void)queryNative(L"api-test-example", nullptr, nullptr); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "Missing native resolver accepted");
    v7Bytes[0] = 8;
    rejected = false;
    try { (void)identify(v7Bytes); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "Unknown schema version accepted");
    std::wprintf(L"Self-test passed: PE32/PE32+ imports/bounds, v6/v7/hybrid detection, native queries/errors, v6 resolution.\n");
}

} // namespace

int wmain(int argc, wchar_t **argv) {
    try {
        if (argc == 2 && std::wstring(argv[1]) == L"--help") { usage(); return 0; }
        if (argc == 2 && std::wstring(argv[1]) == L"--self-test") { selfTest(); return 0; }
        const bool closureDelay = argc >= 2 && std::wstring(argv[1]) == L"--closure-delay";
        if (argc >= 2 && (std::wstring(argv[1]) == L"--closure" || closureDelay)) {
            if (argc != 3 || !*argv[2]) {
                std::fwprintf(stderr, L"Usage: wxeapiset %ls <EXE/DLL path>\n", argv[1]);
                return 2;
            }
            const auto map = activeMap();
            return printClosure(argv[2], map, identifyLive(map), closureDelay);
        }
        const bool binary = argc >= 2 && std::wstring(argv[1]) == L"--bin";
        if (binary) {
            if (argc != 3 || !*argv[2]) { usage(); return 2; }
            const auto map = activeMap();
            const auto schema = identifyLive(map);
            std::wprintf(L"Source: current process PEB.ApiSetMap | Architecture: %ls | Schema: %u.%u%ls\n",
                         sizeof(void *) == 8 ? L"x64" : L"x86", schema.major, schema.minor,
                         schema.hybrid ? L" (v6 compatibility header)" : L"");
            return printBinary(argv[2], map, schema);
        }
        if (argc != 2 && argc != 4) { usage(); return 2; }
        const bool list = std::wstring(argv[1]) == L"--list";
        const auto query = normalize(argv[1]);
        require(list || validContract(query), "Expected an api-/ext- contract or --list");
        std::wstring importer;
        if (argc == 4) {
            require(std::wstring(argv[2]) == L"--importer", "Expected --importer module.dll");
            importer = lower(argv[3]);
            require(!importer.empty() && importer.find_first_of(L"\\/:") == std::wstring::npos,
                    "Importer must be a module base name, not a path");
        }
        const auto map = activeMap();
        const auto schema = identifyLive(map);
        std::wprintf(L"Source: current process PEB.ApiSetMap | Architecture: %ls | Schema: %u.%u%ls\n",
                     sizeof(void *) == 8 ? L"x64" : L"x86", schema.major, schema.minor,
                     schema.hybrid ? L" (v6 compatibility header)" : L"");
        if (schema.major == 7) {
            if (list && schema.hybrid) {
                require(importer.empty(), "V7 --importer is unavailable: native host-query API has no importer parameter");
                const auto contracts = parse(snapshotV6(map));
                std::wprintf(L"Listing v6 compatibility names with native v7 resolution.\n"
                             L"This inventory does not enumerate v7-only aliases/groups.\n");
                for (const auto &contract : contracts) printNative(contract.name, false, {});
                std::wprintf(L"Compatibility contracts: %zu\n", contracts.size());
                return 0;
            }
            return printNative(query, list, importer);
        }
        const auto contracts = parse(snapshotV6(map));
        std::wprintf(L"Importer: %ls | Mapping only; host loading not checked.\n",
                     importer.empty() ? L"(default)" : importer.c_str());
        if (list) {
            for (const auto &contract : contracts) printContract(contract, importer);
            std::wprintf(L"Contracts: %zu\n", contracts.size());
            return 0;
        }
        const auto contract = lookup(contracts, query);
        if (!contract) {
            std::wprintf(L"%ls -> [absent from active schema]\n", query.c_str());
            return 1;
        }
        std::wprintf(L"Query: %ls\n", query.c_str());
        return printContract(*contract, importer) ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "wxeapiset: %s\n", error.what());
        return 2;
    }
}
