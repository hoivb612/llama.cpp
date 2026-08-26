# AMD `amdkmdag.sys` — `DAL3` NonPaged Pool Consumption: Technical Report

**Prepared for discussion with the AMD graphics driver team.**
**Status:** Not believed to be a leak. Evidence indicates a *deliberate*, one-time,
bulk allocation performed by the Display Core allocator at graphics PnP start, carrying
AMD's own built-in allocation-tracking instrumentation.

---

## 1. Summary

A production system showed **246.2 MB of NonPaged pool** attributed to the single pool
tag **`DAL3`**, across **3,416 outstanding allocations** (~72 KB average), with **0 bytes
paged**. On this system that single tag is the largest NonPaged consumer and roughly
matches the entire NonPaged growth observed when the graphics device starts.

| Metric | Value |
|---|---|
| Tag | `DAL3` (`0x334C4144`) |
| NonPaged bytes | 246.2 MB |
| Paged bytes | 0 |
| Outstanding allocations | 3,416 |
| Average allocation size | ~72 KB |
| Driver | `amdkmdag.sys` (80.2 MB image, no public PDB) |
| Image base | `0x140000000` |

Static analysis of the driver binary shows this tag is emitted by **exactly one**
allocation wrapper. It is a central Display Abstraction Layer / Display Core (DC)
allocator used by thousands of runtime call sites, each passing its own size. The 246 MB
is therefore the *aggregate* working memory of the display pipeline (topology, links,
streams, planes, DSC, DMUB, mode/timing tables, color/LUT, MPO surfaces, HDCP/OPM
context), not a single runaway allocation.

We are **not** asserting a leak. We are asking AMD to help us understand and, if possible,
reduce this footprint — and we found that the allocator already contains AMD's own
stack-capture tracking that would answer the "what is it" question directly if enabled.

---

## 2. How the data was collected

- **Live pool accounting:** an in-house tool (`wxemem`) enumerates pool tags via the
  system pool-tag information class and prints top NonPaged tags by bytes and outstanding
  allocation count. This is where `DAL3 = 246.2 MB / 3,416 allocs` comes from.
- **Snapshot diffing:** an in-house tool (`wxememdiff`) diffs two `wxemem` JSON captures,
  including a per-tag NonPaged section (NEW/GONE tags, top growers/shrinkers). Used to
  observe that NonPaged pool jumps by ~+263 MB at graphics PnP start, matching `DAL3`.
- **Static binary analysis:** `objdump` (PE/COFF x64) disassembly of `amdkmdag.sys`, plus
  an in-house PE scanner (`poolscan.py`) that finds `ExAllocatePoolWithTag`-style call
  sites and their tag immediates.

No AMD symbols (PDB) were available; all findings below are derived from the shipped
binary and are reproducible from it.

---

## 3. Static-analysis findings

### 3.1 Exactly one static call site for the tag

`poolscan.py` reports a single static reference that supplies the `DAL3` immediate to a
pool allocation:

```
DAL3   count=1   tag=DAL3   ExAllocatePoolWithTag   @ RVA 0x16910e8  (VMA 0x1416910e8)
```

The reported "constant size" for that site is a runtime value (multi-GB garbage), which is
itself evidence that **the size is computed at runtime by the caller**, not a fixed
immediate. So one wrapper, many callers, many sizes → a central allocator.

The wrapper lives in section **`PAGED3IC`** (a pageable code section), but it allocates
**NonPaged** pool — consistent with a helper that is called at PASSIVE_LEVEL to hand back
memory that is later touched at high IRQL (VBlank / power transitions), which is why the
*data* must be NonPaged.

### 3.2 The allocator wrapper (annotated disassembly)

VMA `0x141691060`–`0x14169116e`. Register/behavior annotations added by us:

```asm
; void* DalAllocPool(SIZE_T size /*ecx*/, ULONG poolClass /*edx*/, ...)
141691060: prologue; save rbp/rsi/rdi; sub rsp,0x20
14169106f: mov  esi, ecx                 ; esi = original requested SIZE
141691071: test edx, edx                 ; edx = pool-class selector
141691073: je   .default                 ;   0 -> global default pool type
141691075: dec  edx; je .class1          ;   1 -> edi = 1        (NonPagedPool, legacy)
14169107a: dec  edx; je .class2          ;   2 -> edi = 0x204
14169107d: cmp  edx,1; je .class3        ;   3 -> edi = 5
.class1:   mov  edi, 1
.class3:   mov  edi, 5
.class2:   mov  edi, 0x204
.default:  mov  edi, [0x1412ed170]        ; global default pool-type value

14169109f: cmpl [0x144ac5000], 0          ; *** global "tracking enabled" flag ***
1416910a6: je   .noheader
1416910a8: add  ecx, 0x50                 ; if tracking on, add 0x50 (80B header) to size
.noheader:
1416910ab: mov  rax, [0x144ad5010]        ; *** registered custom-allocator hook ptr ***
1416910b7: mov  ebp, ecx                  ; ebp = final size (incl header)
1416910b9: test rax, rax
1416910bc: je   .fallback                 ; no hook registered -> ExAllocatePoolWithTag
1416910be: mov  r8d, 0x334C4144           ; 'DAL3'
1416910c4: mov  edx, ebp                  ; size
1416910c6: mov  ecx, 0x40                 ; hook flags = 0x40
1416910cb: call rax                       ; custom_allocator(0x40, size, 'DAL3')
1416910d0: mov  rbx, rax; test rax,rax; jne .track   ; success -> header/track path

.fallback:
1416910d8: mov  ecx, edi                  ; pool type
1416910da: mov  r8d, 0x334C4144           ; 'DAL3'
1416910e0: bts  ecx, 0xa                  ; set bit 10 in pool type (NX/execute flag)
1416910e4: mov  rdx, rbp                  ; size
1416910e7: call [0x1407cc6d0]             ; ExAllocatePoolWithTag(PoolType, size, 'DAL3')
           mov  rbx, rax
           ... (NULL checks) ...

.track:                                   ; only when tracking flag @0x144ac5000 != 0
141691118: cmpl [0x144ac5000], 0
14169111f: je   .return_raw
141691121: xor  r9d, r9d                  ; RtlCaptureStackBackTrace arg4 BackTraceHash = NULL
141691124: mov  [rbx+0x4c], edi           ; header+0x4C = pool type / flags
141691127: mov  r8,  rbx                  ; arg3 BackTrace buffer = block start
14169112a: mov  [rbx+0x44], esi           ; header+0x44 = original size
14169112d: mov  edx, 0x8                  ; arg2 FramesToCapture = 8
141691132: mov  dword [rbx+0x48], 0x334C4144   ; header+0x48 = 'DAL3' signature
141691139: mov  ecx, 0x2                  ; arg1 FramesToSkip = 2
14169113e: mov  dword [rbx+0x40], 0x1234ABCD    ; header+0x40 = guard magic
141691145: call [0x1407cc6f8]             ; RtlCaptureStackBackTrace(2, 8, block, NULL)
141691151: lea  rax, [rbx+0x50]           ; return pointer = block + 0x50 (past header)
.return_raw:
141691157: mov  rax, rbx                  ; (tracking off) return raw block
14169116e: ret
```

### 3.3 Resolved imports (IAT)

From the PE import table (`ntoskrnl.exe`):

| IAT VMA | Import |
|---|---|
| `0x1407cc6d0` | `ExAllocatePoolWithTag` (fallback allocator) |
| `0x1407cc6f8` | **`RtlCaptureStackBackTrace`** (allocation stack capture) |

### 3.4 Key data addresses

| VMA | Meaning (inferred) |
|---|---|
| `0x144ac5000` | Global **allocation-tracking enable** flag (0 = off) |
| `0x144ad5010` | **Registered custom-allocator** function pointer (0 = use ExAllocatePoolWithTag) |
| `0x1412ed170` | Global default pool-type value used when class selector = 0 |
| `0x334C4144` | The tag `DAL3`, hardcoded at the alloc call and in the tracking header |
| `0x1234ABCD` | Per-allocation header guard magic |

---

## 4. Interpretation

1. **This is one deliberate allocator, not a leak.** A single tagged wrapper, invoked by
   many DC subsystems with runtime sizes, allocated in bulk at device start. The
   NonPaged-pool step at graphics PnP (~+263 MB) matches the 246 MB `DAL3` figure, which is
   the signature of a **one-time startup allocation**, not slow runtime growth.

2. **NonPaged / NX is required by design.** Display programming runs at elevated IRQL
   (VBlank, power/mode transitions) where paging is illegal, so the backing store must be
   NonPaged. The `bts ecx,0xa` (execute/NX pool-type flag) and the pool-class selector show
   AMD deliberately choosing NonPaged pool variants per allocation class.

3. **AMD already instruments these allocations.** The allocator contains an **opt-in
   leak/attribution tracker**: when the global flag at `0x144ac5000` is non-zero it prepends
   an **80-byte (`0x50`) header** to every `DAL3` block containing:
   - an **8-frame return-address stack trace** of the allocator
     (`RtlCaptureStackBackTrace(FramesToSkip=2, FramesToCapture=8)`),
   - a `0x1234ABCD` guard magic,
   - the requested size,
   - a `DAL3` signature and pool-type/flags word.

   This means **AMD can identify the exact call stacks and size distribution behind the
   3,416 allocations from their own driver**, without third-party ETW, simply by enabling
   that flag (registry knob / checked build / debug switch) and dumping the headers.

4. **A registrable custom allocator exists.** If a hook is installed at `0x144ad5010`, DC
   memory bypasses `ExAllocatePoolWithTag` entirely (called with flags `0x40`, size, tag).
   This suggests AMD can already redirect/quota DC memory through an alternate allocator.

---

## 5. Questions for AMD

1. **Is 246 MB / 3,416 allocations the expected steady-state `DAL3` footprint** for a
   modern DC build, and what is the dominant contributor (mode/timing tables, DSC, MPO,
   DMUB, per-link/stream/plane context)?
2. **What controls the tracking flag at `0x144ac5000`** (the `RtlCaptureStackBackTrace`
   path)? Is there a supported registry value / debug build that turns it on so AMD (or we,
   with guidance) can dump per-allocation stacks and sizes?
3. **What is the custom-allocator hook at `0x144ad5010`?** Is there a supported way to
   register an alternate DC allocator or impose a quota/budget on DC pool?
4. **Which DC features materially reduce `DAL3`** on a fixed display config — e.g. MPO
   (multi-plane overlay), DSC, virtual displays — and what are the supported registry
   controls under the display class key
   (`HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\<NNNN>`)?
5. **For a headless / compute-only deployment**, is there an **MCDM (compute-only) or
   display-less driver variant** that omits the DC/DAL path entirely and reclaims this pool?
6. **Does `DAL3` grow across sleep/resume or repeated mode-set cycles** in your validation?
   (Our next step is to measure this with `wxememdiff`; if it grows, we will file a leak
   report with the tag deltas.)

---

## 6. What we will provide / measure next

- **Leak-vs-steady classification (in progress):** `wxemem` JSON snapshots at
  (a) post-boot, (b) after mode/hotplug changes, (c) after N sleep/resume cycles, diffed
  with `wxememdiff` on the `DAL3` row. Flat = by-design; climbing = leak candidate.
- **Optional live stack attribution (if AMD cannot enable their own tracker):** ETW pool
  tracing with stackwalk, filtered to tag `DAL3`:
  ```
  wpr -start GeneralProfile -start Pool
  ...reproduce (device disable/enable, or boot-trace to catch PnP start)...
  wpr -stop dal3.etl
  ```
  Analyze in WPA → Pool graph → filter Tag = `DAL3` → group by stack.
  **Timing note:** pool ETW records only allocations that occur *while tracing*. Because
  `DAL3` is allocated once at graphics PnP start, the driver must be (re)initialized while
  the trace is live — either by disabling/enabling the display adapter, or by boot-tracing
  so the trace is armed before graphics PnP runs.

---

## 7. Reproduction (from the shipped binary)

```
# Single call site + tag immediate
python poolscan.py amdkmdag.sys | findstr DAL3

# Allocator disassembly (ImageBase 0x140000000)
objdump -d --start-address=0x141691060 --stop-address=0x141691170 amdkmdag.sys

# Resolve IAT thunks used by the wrapper
objdump -p amdkmdag.sys | findstr /C:"ExAllocatePoolWithTag" /C:"RtlCaptureStackBackTrace"
```

Tag encoding: `'DAL3'` little-endian = `44 41 4C 33` = `0x334C4144`.
