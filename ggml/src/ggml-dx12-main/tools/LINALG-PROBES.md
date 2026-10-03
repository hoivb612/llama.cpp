# LinAlg conformance probes

Small self-checking shaders that answer one question each about a D3D12 LinAlg
implementation. They exist because the capability tier does not tell you what a
driver actually does: on Intel Xe3 the tier is reported, the PSO builds, and the
multiply returns wrong results for a shape the driver never implemented.

Run these on any LinAlg-capable part to find out which defects it shares. They
change no production routing and need no llama.cpp build.

For the broad performance and capability matrix, use
[linalg-bench](linalg-bench/README.md) instead. These probes are the narrow
counterpart: one defect each, exact integer expected values, no timing.

## What each probe answers

| probe | question | Intel Arc B390 (Xe3, driver 32.0.101.8992) |
|---|---|---|
| `linalg_probe_f16_intel` | is f16 x f16 -> f16 at 8x16x16 correct, and does GetCoordinate agree with Store | correct, and lane=column element=row |
| `linalg_probe_i8_desc` | is s8 x s8 -> s32 at 8x32x16 correct when operands come from a descriptor | correct |
| `linalg_probe_i8_rows` | is it correct when operands come from groupshared | misleading - miscoded, see `linalg_probe_gs_typed` |
| `linalg_probe_i8_align` | does the Align argument on a descriptor load do anything | **inert - stride truncates to 4 bytes regardless** |
| `linalg_repro_groupshared_load` | SUPERSEDED - measured a type conversion, not a load | ignore, see TUNING section 35 |
| `linalg_probe_gs_typed` | does groupshared Load work with valid typed arrays | **yes - f16 and s8 both exact** |
| `linalg_repro_i8_coord_intel` | does GetCoordinate agree with Store for s8 | agrees, 0/128 disagreements |

`linalg_repro_getcoordinate` is the older AMD-facing form of the last one; see
[AMD_LinAlg_Driver_Bug.md](../../../../AMD_LinAlg_Driver_Bug.md).

The GetCoordinate answer is already known to be vendor-specific: wrong on AMD
RDNA4, correct on Intel Xe3. Do not assume any row of that table carries over.

## Requirements

- Windows Developer Mode on, for experimental shader models
- DXC with LinAlg support (SM 6.10). Known good: 1.10.2605.24
- Agility SDK preview. Known good: 1.721.3-preview
- A VS x64 developer prompt

## Build

Both commands are one-time. Substitute your own DXC and Agility paths.

```bat
set DXC=C:\dxc\1.10.2605.24
set AGILITY=C:\AgilitySDK\1.721.3-preview\build\native

%DXC%\bin\x64\dxc.exe -T cs_6_10 -E main -enable-16bit-types ^
    -I %DXC%\inc\hlsl ^
    -Fo probe.cso ggml\src\ggml-dx12\shaders\<probe>.hlsl

cl /nologo /std:c++17 /EHsc /O2 ggml\src\ggml-dx12\tools\linalg_repro_host.cpp ^
    /I %AGILITY%\include /link d3d12.lib dxgi.lib dxguid.lib
```

`dxguid.lib` is required for `CLSID_D3D12SDKConfiguration`; without it the link
fails with LNK2019.

## Run

```
linalg_repro_host.exe probe.cso <agility-bin-x64-dir> <sdk-version> [adapter] [raw-count]
```

The detailed dump assumes a 16x16 output window, which most probes do not use.
Pass a raw-count to see their cells; 1100 covers every probe here.

```bat
linalg_repro_host.exe probe.cso %AGILITY%\bin\x64 721 0 1100
```

Each probe writes its sections at fixed indices and documents them in its own
header comment. Read that header before judging the numbers - several probes
scale their output by 16 so the integer readback keeps a fractional digit.

## Worked example

`linalg_repro_groupshared_load` multiplies two constant matrices, A filled with
3 and B filled with 1 over K=32, so every cell must be 96.

```
mismatching cells: 0 (expected 0)
  [ 100] = 516128     <- the matrix result
  [ 300] = 50529027   <- control: the same groupshared bytes read normally
```

516128 is 32 * 127 * 127, meaning both operands were read as 0x7F. The control
reads 0x03030303, so the data is present and the barrier is correct; only
`Matrix::Load` fails to see it. A conforming part prints 96 at index 100.

Note that `mismatching cells: 0` is not the verdict here - that counter belongs
to the 16x16 dump. Judge each probe by its own documented cells.

## Reporting

Worth capturing for any new part: the adapter string and driver version, the
`LinAlg tier` line the host prints, and the cell values each probe documents.
The backend prints its own view of the same capabilities at load:

```
LinAlg: yes tier=16 no-wave16x16 wave8x16x16
```

`wave16x16` is f16 x f16 -> f32 at 16x16x16, which most of the shipped LinAlg
paths need. `wave8x16x16` is the f16-accumulator shape Intel Xe3 has instead,
which the flag-264 GEMM uses. A part reporting a tier but neither shape will
fall back off the LinAlg paths, which is intended: that combination is what
silently returned wrong results before the shape query was added.
