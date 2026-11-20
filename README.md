# zluda-gfx906 — ZLUDA per Radeon VII (gfx906) su Windows

**2S · 2Solution** · fork di [vosen/ZLUDA](https://github.com/vosen/ZLUDA) · [English below](#english)

> Il file README originale di ZLUDA è in [README-ZLUDA.md](README-ZLUDA.md).
> ZLUDA è opera di Andrzej Janik e dei suoi contributori. Questo fork lo porta su una GPU AMD
> Vega 20 sotto Windows, dove AMD non supporta più ufficialmente la scheda.

## Cosa fa

ZLUDA traduce a runtime le chiamate CUDA in chiamate HIP/ROCm: un programma scritto per CUDA
gira sulla GPU AMD senza essere ricompilato. Questo fork lo porta sulla **Radeon VII (gfx906,
wave64)** con la runtime **HIP 5.7** del driver Windows. Le DLL che produce:

| DLL prodotta | crate | cosa traduce |
|---|---|---|
| `nvcuda.dll` | `zluda` | driver API. Porta a wave64: il warp CUDA da 32 lane resta confinato nella sua metà della wavefront da 64 (shuffle con `ds_bpermute`, `laneid`, `lanemask_*`, ballot, vote, activemask), e così riduzioni e look-back di cub (sort, scan) danno il risultato giusto; `bfind` in tutte le forme (correzione del parser PTX); modulo supplementare per i kernel elementwise `float` mancanti, ponte sullo stream nullo |
| `cudnn64_8.dll` | `zluda_dnn8` → `zluda_dnn` | cuDNN 8 verso MIOpen: tutti gli 86 simboli che importa `torch_cuda.dll` |
| `cublas64_11.dll`, `cublasLt64_11.dll` | `zluda_blas`, `zluda_blaslt`, `zluda_blaslt12` | BLAS verso rocBLAS / hipBLASLt (incluse le varianti `_64`) |
| `cudart64_110.dll`, `cudart64_12.dll` | `zluda_cudart` | runtime API |
| `cufft64_10/11/12.dll`, `curand64_10.dll`, `cusolver64_11.dll` | `zluda_fft` (di upstream, su hipFFT), `zluda_curand`, `zluda_cusolver` | FFT, generatori casuali, solver |
| `nvrtc64_*.dll`, `nvJitLink.dll`, `nvjpeg`, `nccl` | `zluda_nvrtc`, `zluda_nvjitlink`, `zluda_nvjpeg`, `zluda_nccl` | NVRTC e nvJitLink: compilazione a runtime via hipRTC. nvJPEG e NCCL: stub che rispondono «non supportato» invece di far cadere il processo |

Il pacchetto delle [Releases](https://github.com/2s-2solution/zluda-gfx906/releases) contiene
**sei** di queste DLL, le sole provate:
- `nvcuda.dll`;
- `cublas64_11.dll`, `cublasLt64_11.dll`;
- `cudart64_110.dll`;
- `cudnn64_8.dll`;
- `nvrtc64_112_0.dll`.

Dalla 2026.11.14 contiene anche `clang\clang.exe` (clang 24 della stessa LLVM di ZLUDA, vedi
«Il clang dei kernel»).

Le altre (cuFFT, cuRAND, cuSOLVER, `cudart64_12`, nvJitLink, nvJPEG, NCCL) si compilano dai
sorgenti ma **non sono nel pacchetto** e non sono state validate.

## Requisiti

Componenti 2S, ciascuno nel proprio repository e con il proprio pacchetto:

- **[rocm-marconi](https://github.com/2s-2solution/rocm-marconi)**, obbligatorio, versione del 12
  novembre 2026 o successiva (provato con la 2026.11.14): quella che traduce l'ABI HIP 7.x verso la
  runtime 5.7. È il kit HIP
  5.7 e porta:
  - lo shim `amdhip64_7_shim.dll`;
  - rocBLAS con la libreria gfx906;
  - hipRTC 5.7.

  Il clang con cui ZLUDA compila i kernel tradotti non viene più da Marconi: deve venire dalla
  stessa LLVM che ZLUDA porta dentro di sé (la 24, vedi sotto), e il pacchetto lo porta in
  `zluda\clang\clang.exe`.
- **[miopen-gfx906](https://github.com/2s-2solution/miopen-gfx906)**, obbligatorio: la MIOpen
  corretta per gfx906, con il proprio pacchetto da installare dopo il kit. La `cudnn64_8.dll` di
  ZLUDA importa `MIOpen.dll` e la trova nel PATH, dove la mette l'installatore di miopen-gfx906.
  L'installatore di ZLUDA controlla che ci sia (`MIOPEN_GFX906_ROOT`), e la verifica controlla che
  la `MIOpen.dll` caricata sia proprio quella.

Provati su una sola macchina:
- AMD Radeon VII (Vega 20, gfx906, 16 GB HBM2). La famiglia gfx900 (Vega 10) è compatibile a
  livello di ISA, ma non è stata provata con questo fork.
- Windows 10 Pro 22H2 (Windows 11 non provato), driver AMD Adrenalin 31.0.21923.11000 (runtime HIP
  5.7 in `System32\amdhip64.dll`).
- Per compilare servono Rust stabile, CMake, Ninja, Python e Visual Studio Build Tools 2022. L'HIP
  SDK serve solo per rigenerare `ptx/lib/zluda_ptx_impl*.bc` (header HIP e `ocml.bc`); a runtime
  non serve.

## Installazione rapida

1. Installare prima il kit [rocm-marconi](https://github.com/2s-2solution/rocm-marconi/releases),
   poi [miopen-gfx906](https://github.com/2s-2solution/miopen-gfx906/releases).
2. Dalla pagina **[Releases](https://github.com/2s-2solution/zluda-gfx906/releases)** scaricare
   `zluda-gfx906-<versione>.exe`, verificarne lo SHA256 pubblicato accanto ed estrarlo.
3. Nella cartella estratta lanciare:
   - `installa.cmd -SenzaPyTorch` per installare solo ZLUDA;
   - oppure `installa.cmd -Python <python.exe>` per collegare anche un ambiente PyTorch
     2.3.1+cu118 (vedi sotto).

ZLUDA si installa in `%LOCALAPPDATA%\2S\zluda`, viene aggiunto al Path dell'utente e parte
`verifica.cmd`. La verifica prova un kernel PTX tradotto, una sgemm cuBLAS e un handle cuDNN,
tutti contro la CPU, e PyTorch se è collegato. `disinstalla.cmd` rimette tutto com'era.
Istruzioni complete nel `LEGGIMI.md` del pacchetto. I pacchetti binari non sono nel repository
git.

## Come si compila

```bat
git clone --recurse-submodules -b gfx906-windows https://github.com/2s-2solution/zluda-gfx906
cd zluda-gfx906
:: se il clone è stato fatto senza --recurse-submodules: servono ext\llvm-project ed ext\HiGHS,
:: altrimenti llvm_zluda e highs-sys non compilano
git submodule update --init
cargo build --profile release-lto -p zluda -p zluda_blas -p zluda_blaslt -p zluda_cudart -p zluda_dnn8 -p zluda_nvrtc
```

I `-p` servono: un `cargo build --profile release-lto` senza argomenti compila solo i membri
predefiniti del workspace e **non** produce cuBLAS, cudart né NVRTC. Le DLL escono in
`target\release-lto`. I nomi con cui le cercano i programmi CUDA 11 sono:
- `cublas64_11.dll` (da `cublas.dll`);
- `cublasLt64_11.dll` (da `cublaslt.dll`);
- `cudart64_110.dll` e `cudnn64_8.dll` (già con questo nome);
- `nvrtc64_112_0.dll` (da `nvrtc.dll`).

Non servono librerie d'importazione: dal merge di upstream del 13 novembre 2026 i binding HIP
usano `raw-dylib`. Tutte le funzioni HIP vanno allo shim `amdhip64_7_shim.dll` di Marconi, caricato
in differita. Fa eccezione la registrazione dei kernel di cudart (`__hipRegisterFatBinary/Function`),
che resta sulla runtime 5.7 (`amdhip64.dll`). Il nome `amdhip64_7.dll` si evita apposta: un driver
recente può metterne una vera in `System32`, che verrebbe trovata prima del PATH.

**Il clang dei kernel.** ZLUDA traduce il PTX in IR LLVM con la LLVM di `ext\llvm-project`, la
ROCm LLVM 10.1 (LLVM 24). Il codice macchina lo produce un clang esterno, che deve venire dalla
stessa LLVM: l'IR testuale della 24 non lo legge il clang 21 di Marconi. Lo si costruisce dallo
stesso sottomodulo, con la CRT statica:

```bat
cmake -S ext\llvm-project\llvm -B build_clang -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF ^
  -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF
ninja -C build_clang clang
```

`build_clang\bin\clang.exe` va in `clang\clang.exe` accanto a `nvcuda.dll`. Lo stesso clang,
insieme a `llvm-dis` e `llvm-as`, rigenera `ptx/lib/zluda_ptx_impl.bc` e
`zluda_ptx_impl_constrained.bc` (comandi in testa a `ptx/lib/zluda_ptx_impl.cpp`; target gfx906
wave64, code object 5, header HIP 7.x e `ocml.bc` di ROCm 7.x).

**Build pubblicabile, senza percorsi personali nei binari.** Rust scrive nelle DLL i percorsi
del sorgente, del registro cargo e di rustup, cioè la cartella utente. Per le DLL delle Releases
questi percorsi sono rimappati e il PDB è citato senza percorso. Da un prompt *x64 Native Tools*
di Visual Studio 2022:

```bat
set "RUSTFLAGS=-Ctarget-feature=+crt-static --remap-path-prefix=%CD%=zluda --remap-path-prefix=%USERPROFILE%\.cargo\registry\src=cargo --remap-path-prefix=%USERPROFILE%\.rustup=rustup -Clink-arg=/PDBALTPATH:%%_PDB%%"
cargo build --profile release-lto --target x86_64-pc-windows-msvc -p zluda -p zluda_blas -p zluda_blaslt -p zluda_cudart -p zluda_dnn8 -p zluda_nvrtc
```

Il comando va in un file `.cmd`, come sopra; digitandolo direttamente nel prompt si scrive
`%_PDB%` con un solo `%`. Poi si verifica con `strings` o simili che le DLL non contengano
`C:\Users\`.

**Percorsi cercati a runtime.**

- clang: ZLUDA cerca la variabile `ZLUDA_CLANG`, poi `clang\clang.exe` accanto a `nvcuda.dll`.
  È necessario: senza, il caricamento dei moduli fallisce con un messaggio esplicito. Il codegen
  interno di LLVM si attiva solo con `ZLUDA_CODEGEN_INTERNO=1`: provato il 13 novembre 2026, fa
  cadere la runtime 5.7 al primo kernel.
- Librerie ROCm caricate a runtime (rocBLAS, hipBLASLt, MIOpen, rocSPARSE, hipFFT): accanto alla
  DLL di ZLUDA, poi per nome (il modulo già caricato o il `PATH`), poi `%MARCONI_ROOT%\bin`, e solo
  per ultimo `%HIP_PATH%\bin`. `HIP_PATH` può indicare un HIP SDK 6.x o 7.x, le cui librerie sono
  per un'altra runtime.
- Shim HIP (`amdhip64_7_shim.dll`): si carica per nome, con l'ordine di ricerca normale di
  Windows (il modulo già caricato, la cartella dell'eseguibile, System32, il `PATH`), **non**
  dalla cartella di `nvcuda.dll`. Se nel `PATH` c'è prima uno shim più vecchio, ZLUDA si lega a
  quello e perde la traduzione dell'ABI: per esempio `cuPointerGetAttributes` cade dentro
  `hipDrvPointerGetAttributes` della runtime 5.7. Il bin del kit Marconi va quindi tenuto prima di
  ogni altra cartella che contenga uno shim. Con PyTorch basta copiare lo shim in `torch\lib`,
  perché PyTorch precarica tutte le DLL di quella cartella.
- Cache dei moduli compilati: `ZLUDA_CACHE_DIR` se impostata, altrimenti
  `%LOCALAPPDATA%\zluda\ComputeCache`. Quest'ultima si trova con l'API Known Folder, che ignora la
  variabile `LOCALAPPDATA`: per tenere separata un'installazione di prova serve `ZLUDA_CACHE_DIR`.
  La chiave della cache contiene il backend del codegen.
- Livello d'ottimizzazione del clang esterno: **-O0** predefinito, quello validato in produzione;
  `ZLUDA_OPT_LIVELLO=O3` attiva -O3, non validato.
- hipRTC 5.7 (`zluda/src/impl/cu_link.rs`, `zluda_nvjitlink/src/impl.rs`):
  `C:\Program Files\AMD\ROCm\5.7\bin\hiprtc0507.dll`, poi `%MARCONI_ROOT%\bin\hiprtc0507.dll`, poi
  `hiprtc0507.dll` dal `PATH`. nvJitLink prova anche, prima del `PATH`,
  `C:\rocm_5_7_expansion_pack\bin`: è la cartella storica della macchina di sviluppo e si ignora
  se non esiste.
- File di diagnostica: solo con `ZLUDA_DIAG` impostata, nella cartella `ZLUDA_DIAG_DIR`
  (predefinita `%TEMP%\zluda_diag`). Senza `ZLUDA_DIAG`, ZLUDA scrive su disco solo la cache dei
  moduli (`%LOCALAPPDATA%\ZLUDA`).
- `zluda/src/impl/zluda_aten_float_vec4.hsaco` contiene i kernel elementwise `float`
  supplementari. È un binario gfx906 il cui sorgente `.hip` non è stato conservato: accanto c'è
  il disassemblato completo, `zluda_aten_float_vec4.s`.
- Nella `nvcuda.dll` restano, come testo dei messaggi d'asserzione di LLVM, i percorsi della
  cartella in cui è stato compilato LLVM (`...\ext\llvm-project\...`). Non contengono la cartella
  utente e non si usano a runtime. Per toglierli si compila LLVM con `-ffile-prefix-map` (o
  `/d1trimfile:` con MSVC).
- `cuEventRecordWithFlags` con `flags != 0` restituisce `NOT_SUPPORTED`: HIP 5.7 non ha
  `hipEventRecordWithFlags`.
- **ABI HIP 7.x contro runtime 5.7.** I binding di ZLUDA seguono gli header 7.x, ma a eseguire è la
  runtime 5.7 del driver. Le funzioni che passano un tipo di memoria o un `HIP_MEMCPY3D` (copie 2D/3D,
  attributi dei puntatori) vanno quindi **sempre** allo shim di rocm-marconi, che traduce.
  - Le versioni asincrone delle copie e la `cudaPointerGetAttributes` di cudart si risolvono per
    indirizzo in `amdhip64_7_shim.dll`. Importate per nome, il linker le prendeva dalla 5.7.
  - Serve il kit rocm-marconi del 12 novembre 2026 o successivo.
  - Prima di questa correzione `cuPointerGetAttribute` dava «host» per la memoria GPU, e
    `cuMemcpy3D` restituiva successo con i dati sbagliati.
- Gli attributi booleani dei puntatori (`IS_MANAGED`, `MAPPED`, `SYNC_MEMOPS`...) si azzerano su 4
  byte prima di chiederli: la 5.7 ne scrive uno solo. Senza, la runtime CUDA statica di PyTorch
  leggeva il resto del valore dalla pila e vedeva la memoria bloccata come «managed», quindi
  `is_pinned()` restituiva `False`.

⚠️ Usare il profilo `release-lto`: dopo una build con un altro profilo si rischia di misurare il
binario vecchio. Dopo ogni ricompilazione bisogna svuotare la cache dei moduli di ZLUDA. Le DLL
vanno poi copiate dove le carica il programma e, dopo la copia, se ne controlla l'impronta: una
DLL in uso non si sovrascrive e l'errore è silenzioso.

## Esempio d'uso e banco di prova: PyTorch

Le wheel ufficiali **PyTorch 2.3.1+cu118** sono il programma CUDA con cui questo fork è stato
provato. PyTorch carica le DLL CUDA dalla propria cartella `torch\lib`, non dal Path, quindi le
cinque DLL di ZLUDA vanno messe lì. Con `installa.cmd -Python <python.exe>` lo fa il pacchetto,
conservando gli originali.

Con la MIOpen di miopen-gfx906 le due variabili `MIOPEN_DEBUG_...` non servono (vedi i limiti);
restano necessarie solo con una MIOpen senza le correzioni. Le altre sono facoltative:

```python
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")                   # facoltativa: la v7 è più veloce (9 contro 26 ms)
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
os.environ.setdefault("TORCH_CUBLAS_USE_LT", "0")
import torch
```

Risultati misurati con PyTorch (agosto–ottobre 2026, Radeon VII):

| prova | esito |
|---|---|
| simboli cuDNN che PyTorch importa | **86 / 86** implementati |
| convoluzioni, API cuDNN v7 e v8 | 13 / 13 e 13 / 13 (scarto 2–6·10⁻⁷ dalla CPU) |
| addestramento di una CNN, 5 passi | pesi uguali alla CPU entro 5,9·10⁻⁸ |
| RNN / LSTM / GRU | 10 / 10 |
| spatial transformer | 4 / 4 |
| `torch.backends.cudnn.benchmark = True` | corretto (4·10⁻⁷) |
| passo CNN a regime (8×3×32×32) | **9 ms** (230 ms prima della cache degli algoritmi) |
| matmul fp16 4096² via rocBLAS | 11,5 TFLOPS a stream singolo |

Chi può ricompilare PyTorch ha un'alternativa più veloce: [la-ruota](https://github.com/2s-2solution/la-ruota),
cioè PyTorch compilato nativo per ROCm, senza traduzione, è circa **5 volte più veloce** sullo
stesso carico. ZLUDA resta utile per i programmi che arrivano solo in versione CUDA.

## Documentazione

- [docs/CUDNN_MIOPEN_gfx906.md](docs/CUDNN_MIOPEN_gfx906.md): come `zluda_dnn` è arrivato a
  coprire gli 86 simboli cuDNN che PyTorch importa, cioè il metodo, le RNN, lo spatial
  transformer e l'API backend v8 (resoconto del 2 agosto 2026).

## Limiti noti

- **MIOpen sbaglia in silenzio il gradiente sui pesi** con due solver:
  - `ConvAsmBwdWrW3x3`: conv 3×3 con 3 canali d'ingresso, cioè il primo strato di ogni rete RGB;
  - `ConvAsmImplicitGemmV4R1DynamicWrw`.

  Con una MIOpen non corretta gli scarti misurati sono 0,96 e 1,2 (relativi alla CPU), e
  l'addestramento diverge senza alcun errore. Se ne esce in due modi:
  - spegnendo i due solver con `MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3=0` e
    `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1=0`;
  - oppure usando la MIOpen di [miopen-gfx906](https://github.com/2s-2solution/miopen-gfx906), che
    li corregge.

  Con quest'ultima, provata sotto ZLUDA l'11 novembre 2026 con i solver imposti e senza le
  variabili, gli scarti sono 8,1·10⁻⁷ e 2,0·10⁻⁶.
- cuBLASLt: `cublasLt64_11.dll` carica hipBLASLt (`hipblaslt.dll`) solo al primo uso, ma
  hipBLASLt non è in nessun pacchetto, quindi le chiamate cuBLASLt non funzionano. Con PyTorch
  si usa `TORCH_CUBLAS_USE_LT=0`.
- Funzioni cuDNN non implementate: restituiscono `NOT_SUPPORTED` e PyTorch ripiega da sé.
  - attivazione SWISH;
  - LSTM con proiezione (`proj_size > 0`);
  - batch norm con attivazione fusa;
  - spatial transformer fp16;
  - grafi cuDNN v8 con più operazioni.
- cuFFT, cuRAND, cuSOLVER e cuSPARSE non sono nel pacchetto: con PyTorch restano quelle di NVIDIA,
  e le funzioni che le usano sulla GPU (per esempio `torch.fft`) non funzionano. cuFFT (su hipFFT)
  e cuSPARSE (su rocSPARSE) sono ora quelli di upstream, non provati su gfx906.
- `torch.topk` fa cadere il processo con una divisione intera per zero: la runtime 5.7 non risponde
  a `MAX_BLOCKS_PER_MULTIPROCESSOR` né a `MAX_REGISTERS_PER_MULTIPROCESSOR`, ZLUDA li riporta come 0
  e PyTorch ci divide. Era così anche nella 2026.11.12 (trovato il 13 novembre 2026).
- Le modifiche di upstream al livello cuDNN (`zluda_dnn`, per esempio `cudnnGetProperty`) non sono
  ancora integrate: il livello cuDNN è quello validato 86/86.
- Alcune specializzazioni elementwise `float` assenti nella wheel (es. `pow`) richiedono un
  ripiego su CPU.
- `torch.compile` / Inductor non sono supportati.
- Superati: i difetti di agosto 2026 su sort/argsort oltre n=4096, `torch.unique` e il backward di
  `nn.Embedding` con meno di 2048 indici non si ripresentano. Nella prova estesa del 14 nov 2026
  (28 operazioni contro la CPU) sono giusti sort fino a n=200000, argsort stabile int64 con
  n=50000, cumsum int64 con n=100000, `unique` con conteggi e il backward di Embedding con 100 e
  5000 indici.
- Prestazioni: i kernel elementwise tradotti dal PTX raggiungevano circa 23 GB/s, contro i
  489 GB/s di una copia pura sulla stessa scheda, e un lancio costava circa 46 µs.
- Più stream non danno velocità: sulla runtime 5.7 del driver i kernel su stream diversi non si sono
  mai sovrapposti nelle nostre prove. L'ordine fra stream però non è garantito (la RNN multi-stream
  di MIOpen aveva una corsa vera): le dipendenze fra stream vanno dichiarate con gli eventi.
- Prove eseguite su una sola scheda; nessuna garanzia su altre GPU o driver.

## Licenza

Come ZLUDA: a scelta **Apache-2.0** ([LICENSE-APACHE](LICENSE-APACHE)) o **MIT**
([LICENSE-MIT](LICENSE-MIT)). Le modifiche di questo fork sono distribuite alle stesse
condizioni; vedi [NOTICE](NOTICE).

---

<a name="english"></a>
# English

**zluda-gfx906** is a fork of [vosen/ZLUDA](https://github.com/vosen/ZLUDA) (by Andrzej Janik and
contributors; original README in [README-ZLUDA.md](README-ZLUDA.md)). It runs unmodified CUDA
programs on an **AMD Radeon VII (gfx906)** under **Windows**, on top of the driver's **HIP 5.7**
runtime.

**What it adds:**
- a wave64 port of the PTX translator: each 32-lane CUDA warp is confined to its half of the
  64-lane wavefront (shuffles with `ds_bpermute`, `laneid`, `lanemask_*`, ballot, vote,
  activemask), which makes reductions and cub's look-back (sort, scan) correct;
- a PTX parser fix: all forms of `bfind`, not only `bfind.shiftamt.u32`;
- a supplementary code object for missing `float` elementwise kernels;
- a null-stream bridge;
- CUDA libraries mapped to AMD ones:
  - `cudnn64_8.dll` to MIOpen (all 86 cuDNN symbols imported by `torch_cuda.dll`);
  - cuBLAS/cuBLASLt to rocBLAS and hipBLASLt;
  - cudart 11/12, cuFFT (upstream's, on hipFFT), cuRAND, cuSOLVER;
  - NVRTC through hipRTC;
  - stubs for nvJitLink, nvJPEG and NCCL that return "not supported" instead of crashing.

The release package ships the six tested DLLs: `nvcuda`, `cublas64_11`, `cublasLt64_11`,
`cudart64_110`, `cudnn64_8`, `nvrtc64_112_0`; since 2026.11.14 also `clang\clang.exe` (clang 24 from
ZLUDA's own LLVM).

**Requirements:**
- [rocm-marconi](https://github.com/2s-2solution/rocm-marconi), mandatory: the HIP 5.7 shim,
  rocBLAS and hipRTC. The clang that compiles the translated kernels must come from ZLUDA's own
  LLVM 24 and ships in the package as `zluda\clang\clang.exe`;
- [miopen-gfx906](https://github.com/2s-2solution/miopen-gfx906), mandatory: the corrected
  MIOpen, installed after the kit. ZLUDA's `cudnn64_8.dll` imports `MIOpen.dll` and finds it on
  the PATH; the installer checks `MIOPEN_GFX906_ROOT`, and the built-in check verifies which
  `MIOpen.dll` is loaded.

Tested only on a Radeon VII, Windows 10 Pro 22H2, Adrenalin driver 31.0.21923.11000.

**Quick install:**
1. Install rocm-marconi first, then miopen-gfx906.
2. Download `zluda-gfx906-<version>.exe` from the
   **[Releases](https://github.com/2s-2solution/zluda-gfx906/releases)** page, check its
   published SHA256 and extract it.
3. Run `installa.cmd -SenzaPyTorch` for ZLUDA alone, or `installa.cmd -Python <python.exe>` to
   also hook up a PyTorch 2.3.1+cu118 environment.

The built-in check runs a translated PTX kernel, a cuBLAS sgemm and a cuDNN handle against the
CPU, plus PyTorch when it is hooked up. `disinstalla.cmd` reverts everything.

**Build:** same prerequisites as upstream ZLUDA (Rust, CMake, Ninja, Python, VS Build Tools 2022).
- Clone with `--recurse-submodules` (or run `git submodule update --init`): `ext/llvm-project` and
  `ext/HiGHS` are required.
- Build with
  `cargo build --profile release-lto -p zluda -p zluda_blas -p zluda_blaslt -p zluda_cudart -p zluda_dnn8 -p zluda_nvrtc`.
  Without the `-p` flags only the default workspace members are built, so cuBLAS, cudart and
  NVRTC are missing.
- No import libraries are needed: the HIP bindings use `raw-dylib` and go to the rocm-marconi shim
  (`amdhip64_7_shim.dll`, delay-loaded); cudart's kernel registration stays on the 5.7 runtime.
- Build clang from the same `ext/llvm-project` (`-DLLVM_ENABLE_PROJECTS=clang`, static CRT) and put
  it in `clang\clang.exe` next to `nvcuda.dll`; see the Italian section for the commands.
- For redistributable DLLs, see the Italian section: `--remap-path-prefix` and `/PDBALTPATH:%_PDB%`.

**Runtime lookups:**
- clang: `ZLUDA_CLANG`, then `clang\clang.exe` next to `nvcuda.dll`; without it module loading fails
  with an explicit message (the in-process codegen is opt-in, `ZLUDA_CODEGEN_INTERNO=1`, and crashed
  the 5.7 runtime on the first kernel);
- ROCm libraries: next to the ZLUDA DLL, then by name (already loaded or `PATH`), then
  `%MARCONI_ROOT%\bin`, and `%HIP_PATH%\bin` only last;
- HIP shim (`amdhip64_7_shim.dll`): loaded by name with the normal Windows search order (already
  loaded module, executable folder, System32, `PATH`), **not** from `nvcuda.dll`'s folder. An older
  shim earlier on the `PATH` wins and the ABI translation is lost (e.g. `cuPointerGetAttributes`
  crashes inside the 5.7 runtime's `hipDrvPointerGetAttributes`): keep the Marconi kit's `bin`
  ahead of any other folder with a shim. With PyTorch, copying the shim into `torch\lib` is
  enough, because PyTorch preloads every DLL in that folder;
- module cache: `ZLUDA_CACHE_DIR`, else `%LOCALAPPDATA%\zluda\ComputeCache` (found through the Known
  Folder API, which ignores the `LOCALAPPDATA` variable);
- default optimisation level -O0 (validated); `ZLUDA_OPT_LIVELLO=O3` is opt-in and unvalidated;
- hipRTC 5.7: the HIP SDK path, then `%MARCONI_ROOT%\bin`, then the `PATH`;
- calls that pass a memory type or `HIP_MEMCPY3D` always go through the rocm-marconi shim, which
  translates the HIP 7.x ABI to the driver's 5.7 runtime (requires rocm-marconi 2026.11.12 or later).
  Before this fix, GPU memory was reported as host memory and `cuMemcpy3D` returned success with
  wrong data;
- boolean pointer attributes are zeroed on 4 bytes, because 5.7 writes 1. Before, PyTorch saw
  pinned memory as managed and `is_pinned()` was `False`;
- diagnostic files are written only with `ZLUDA_DIAG`, into `ZLUDA_DIAG_DIR` (default
  `%TEMP%\zluda_diag`);
- the supplementary `zluda_aten_float_vec4.hsaco` has no surviving source; its full disassembly
  is next to it.

**Usage example and test bench, PyTorch:** the official PyTorch 2.3.1+cu118 wheels load the CUDA
DLLs from `torch\lib`, so ZLUDA's five libraries must be copied there; the package does this with
`-Python`. Measured (Aug–Oct 2026):
- PyTorch convolutions 13/13 on both cuDNN v7 and v8 paths, 2–6e-7 from the CPU;
- 5-step CNN training matches the CPU within 5.9e-8;
- RNN/LSTM/GRU 10/10, spatial transformer 4/4;
- steady-state CNN step 9 ms;
- fp16 4096² matmul 11.5 TFLOPS.

The native build [la-ruota](https://github.com/2s-2solution/la-ruota) is about 5× faster.

**Known limits:**
- An uncorrected MIOpen silently returns wrong weight gradients (`ConvAsmBwdWrW3x3`, V4R1 WrW:
  relative errors 0.96 and 1.2) unless `MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3=0` and
  `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1=0` are set. With the miopen-gfx906 MIOpen the
  variables are not needed: under ZLUDA, forced solvers give 8.1e-7 and 2.0e-6 (11 Nov 2026).
- cuBLASLt calls do not work: hipBLASLt (`hipblaslt.dll`, delay-loaded) is not packaged. With
  PyTorch use `TORCH_CUBLAS_USE_LT=0`.
- SWISH, LSTM projections, fused batch-norm activations, fp16 spatial transformer and multi-op
  cuDNN v8 graphs return `NOT_SUPPORTED`.
- cuFFT, cuRAND, cuSOLVER and cuSPARSE are not packaged; cuFFT and cuSPARSE now come from upstream
  and are untested on gfx906.
- `torch.topk` crashes with an integer division by zero: the 5.7 runtime does not answer
  `MAX_BLOCKS_PER_MULTIPROCESSOR` nor `MAX_REGISTERS_PER_MULTIPROCESSOR`, ZLUDA reports 0 and PyTorch
  divides by it. Already the case in 2026.11.12 (found on 13 Nov 2026).
- Upstream's cuDNN changes are not merged yet; the cuDNN layer is the validated 86/86 one.
- Fixed since August 2026: in the extended test of 14 Nov 2026 (28 operations against the CPU)
  sort up to n=200000, stable int64 argsort (n=50000), int64 cumsum (n=100000), `unique` with
  counts and `nn.Embedding` backward (100 and 5000 indices) are all correct.
- Multiple streams give no speed-up (kernels on different streams never overlapped in our tests),
  but ordering across streams is not guaranteed, so cross-stream dependencies need events.
- Translated elementwise kernels reached about 23 GB/s, against 489 GB/s for a plain copy; a
  launch cost about 46 µs.
- How the cuDNN layer was built: [docs/CUDNN_MIOPEN_gfx906.md](docs/CUDNN_MIOPEN_gfx906.md) (Italian).
- Some `float` elementwise kernels fall back to the CPU.
- No `torch.compile`.
- Tested on a single card only.

**License:** Apache-2.0 or MIT, at your option, as upstream ZLUDA. See [NOTICE](NOTICE).

**Upstream base:** `vosen/ZLUDA` commit `eb89ae3b` ("Add an executable picker to the Windows
launcher (#676)", 11 Nov 2026), merged into branch `gfx906-windows` on 13 Nov 2026; the port
itself started from `aa47713c` ("Use LLVM for optimized i8 MMAs (#571)").
