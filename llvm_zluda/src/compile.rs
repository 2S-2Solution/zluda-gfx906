use crate::utils::{Context, Message, PassBuilderOptions, TargetMachine};
use crate::LLVMZludaParseCommandLineOptions;
use crate::{ffi::LLVMZludaLinkWithLLD, utils::Module};
use llvm_sys::{
    core::*,
    target::{
        LLVMInitializeAMDGPUAsmPrinter, LLVMInitializeAMDGPUTarget, LLVMInitializeAMDGPUTargetInfo,
        LLVMInitializeAMDGPUTargetMC,
    },
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMGetTargetFromTriple,
        LLVMRelocMode,
    },
    transforms::pass_builder::LLVMRunPasses,
};
use llvm_sys::{LLVMLinkage, LLVMVisibility};
use std::ffi::{CStr, CString};
use std::fs;
use std::sync::OnceLock;
use tempfile::NamedTempFile;

const OCKL_MODULE: &[u8] = include_bytes!("device-libs/ockl.bc");
const OCML_MODULE: &[u8] = include_bytes!("device-libs/ocml.bc");

// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
const CONSTANT_ADDRESS_SPACE: u32 = 4;

fn load_module(ctx: &Context, bc: &[u8], name: &std::ffi::CStr) -> Result<Module, String> {
    let module = Module::try_from_bitcode(ctx, bc, Some(name))
        .ok_or(("Failed to parse bitcode").to_string())?;
    module.verify()?;
    Ok(module)
}

// TODO: see if there's a way to reduce duplication with attributes.rs
fn add_constant(context: &Context, module: &Module, name: &CStr, attribute: u32) {
    let attribute_type = unsafe { LLVMInt32TypeInContext(context.get()) };
    let global = unsafe {
        LLVMAddGlobalInAddressSpace(
            module.get(),
            attribute_type,
            name.as_ptr(),
            CONSTANT_ADDRESS_SPACE,
        )
    };
    unsafe { LLVMSetLinkage(global, LLVMLinkage::LLVMExternalLinkage) };
    unsafe { LLVMSetVisibility(global, LLVMVisibility::LLVMHiddenVisibility) };
    unsafe { LLVMSetInitializer(global, LLVMConstInt(attribute_type, attribute as u64, 0)) };
    unsafe { LLVMSetGlobalConstant(global, 1) };
}

fn path_to_cstring(path: &std::path::Path) -> Result<CString, String> {
    let path_str = path
        .to_str()
        .ok_or(("path is not valid as str").to_string())?;
    CString::new(path_str).map_err(|_| ("path includes invalid null byte").to_string())
}

fn get_isa_version_from_gcn_arch(gcn_arch: &str) -> Result<u32, String> {
    // VEGA-FIX T11:55 23 giu: gfx906:sramecc+:xnack- ha feature suffix dopo `:`.
    // Strip prima del replace+parse altrimenti fail.
    let arch_only = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    let base: u32 = arch_only
        .replace("gfx", "")
        .parse()
        .map_err(|_| (format!("could not get ISA version from gcn_arch (parsed='{}', stripped='{}')", gcn_arch, arch_only)))?;
    let stepping = base % 10;
    let minor = (base / 10) % 10;
    let major = base / 100;
    Ok(major * 1000 + minor * 100 + stepping)
}

fn create_oclc_constants(ctx: &Context, gcn_arch: &str) -> Result<Module, String> {
    let module = Module::new(ctx, c"oclc_constants");

    // used by ockl
    add_constant(ctx, &module, c"__oclc_wavefrontsize64", 0);
    add_constant(ctx, &module, c"__oclc_wavefrontsize_log2", 5);
    add_constant(ctx, &module, c"__oclc_ABI_version", 500);
    add_constant(
        ctx,
        &module,
        c"__oclc_ISA_version",
        get_isa_version_from_gcn_arch(gcn_arch)?,
    );

    // used by ocml
    add_constant(ctx, &module, c"__oclc_unsafe_math_opt", 0);
    add_constant(ctx, &module, c"__oclc_correctly_rounded_sqrt32", 1);
    add_constant(ctx, &module, c"__oclc_finite_only_opt", 0);
    Ok(module)
}

pub fn compile(
    ctx: &Context,
    gcn_arch: &str,
    main: Module,
    ptx_impl: &[u8],
    attributes: Module,
    compiler_hook: Option<&dyn Fn(&Vec<u8>, String)>,
) -> Result<Vec<u8>, String> {
    init_globals()?;

    let linked = Module::new(ctx, c"llvm-link");

    let ptx_impl = load_module(ctx, ptx_impl, c"ptx_impl.bc")?;
    let ockl = load_module(ctx, OCKL_MODULE, c"ockl.bc")?;
    let ocml = load_module(ctx, OCML_MODULE, c"ocml.bc")?;

    let oclc_constants = create_oclc_constants(ctx, gcn_arch)?;

    linked.link(main)?;
    linked.link(attributes)?;
    linked.link(oclc_constants)?;
    linked.link(ptx_impl)?;
    linked.link(ockl)?;
    linked.link(ocml)?;

    linked.verify()?;

    if let Some(hook) = compiler_hook {
        // Run compiler hook on human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("linked.ll"));
    }

    let triple = c"amdgcn-amd-amdhsa";
    // VEGA-FIX T12:00 23 giu: strip feature suffix (gfx906:sramecc+:xnack- → gfx906)
    // altrimenti LLVM AMDGPU target rejecta processor name.
    let arch_only = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    let cpu = std::ffi::CString::new(arch_only).map_err(|_| ("invalid gcn_arch").to_string())?;
    // VEGA-FIX T12:00: gfx906 è wave64-only hardware (no wave32 native su GCN5).
    // ZLUDA upstream usa "-wavefrontsize64,+cumode" per RDNA2/3 (wave32 default).
    // Per gfx906 dobbiamo ABILITARE wave64 esplicito.
    let features_gfx906 = c"+wavefrontsize64,+cumode";
    let features_default = c"-wavefrontsize64,+cumode";
    let features = if arch_only.starts_with("gfx9") && !arch_only.starts_with("gfx94") {
        features_gfx906  // GCN5 (Vega): gfx900, gfx902, gfx906, gfx909, gfx90c
    } else {
        features_default  // RDNA + CDNA: wave32 / wave64 selectable
    };

    let mut target = unsafe { std::mem::zeroed() };
    let mut err = std::ptr::null_mut();
    let status = unsafe { LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut err) };
    if status != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }

    // VEGA-FIX T12:15 23 giu: install fatal error handler per catturare
    // LLVM abort() (es. AMDGPU "Cannot select i7") senza crashare process.
    static FATAL_ERR_INIT: std::sync::Once = std::sync::Once::new();
    extern "C" fn vega_fatal_handler(reason: *const ::std::os::raw::c_char) {
        use std::io::Write;
        let msg = if reason.is_null() { "(null)".to_string() }
                  else { unsafe { CStr::from_ptr(reason).to_string_lossy().to_string() } };
        if let Ok(mut f) = crate::diag("zluda_llvm_fatal.log") {
            let _ = writeln!(f, "[LLVM FATAL] {}", msg);
        }
        // Non chiamo abort/exit qui — torno e lascio LLVM continuare se possibile.
        // ATTENZIONE: questo è comportamento NOT-SAFE LLVM-level ma evita Python death.
    }
    FATAL_ERR_INIT.call_once(|| unsafe {
        llvm_sys::error_handling::LLVMInstallFatalErrorHandler(Some(vega_fatal_handler));
    });
    // VEGA-FIX T12:15: downgrade Aggressive→Default codegen + O3→O2 passes per evitare
    // optimization passes che inferiscono i7 ValueType non-selezionabile su AMDGPU.
    let target_machine = TargetMachine::new(
        target,
        triple,
        &cpu,
        features,
        LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
        LLVMRelocMode::LLVMRelocDefault,
        LLVMCodeModel::LLVMCodeModelDefault,
    );

    // VEGA-FIX T12:35: aggiungere @llvm.used globale che references TUTTI i kernel
    // amdgpu_kernel altrimenti il target machine emit elimina via DCE/internalize.
    unsafe {
        let mut kernels: Vec<llvm_sys::prelude::LLVMValueRef> = Vec::new();
        let mut fn_ = LLVMGetFirstFunction(linked.get());
        while !fn_.is_null() {
            let cc = LLVMGetFunctionCallConv(fn_);
            // AMDGPU_KERNEL = 91 (CallingConv::AMDGPU_KERNEL)
            if cc == 91 {
                kernels.push(fn_);
            }
            fn_ = LLVMGetNextFunction(fn_);
        }
        // VEGA-DIAG 2 lug (Fable 5): conta funzioni totali vs kernel cc=91 per capire
        // perche' alcuni moduli torch compilano a 0 kernel (DCE elimina tutto).
        {
            use std::io::Write as _;
            let mut total = 0usize;
            let mut fd = LLVMGetFirstFunction(linked.get());
            while !fd.is_null() { total += 1; fd = LLVMGetNextFunction(fd); }
            if let Ok(mut lf) = crate::diag("zluda_kernelcount.log") {
                let _ = writeln!(lf, "[compile] total_fns={} amdgpu_kernels_cc91={}", total, kernels.len());
            }
        }
        if !kernels.is_empty() {
            let ptr_type = LLVMPointerType(LLVMVoidTypeInContext(ctx.get()), 0);
            let arr_type = LLVMArrayType2(ptr_type, kernels.len() as u64);
            let arr_const = LLVMConstArray2(ptr_type, kernels.as_mut_ptr(), kernels.len() as u64);
            let used_global = LLVMAddGlobal(linked.get(), arr_type, c"llvm.used".as_ptr());
            LLVMSetInitializer(used_global, arr_const);
            LLVMSetLinkage(used_global, LLVMLinkage::LLVMAppendingLinkage);
            LLVMSetSection(used_global, c"llvm.metadata".as_ptr());
        }
    }
    let pb_options = PassBuilderOptions::new();
    let error = unsafe {
        LLVMRunPasses(
            linked.get(),
            c"verify".as_ptr(),
            target_machine.get(),
            pb_options.get(),
        )
    };

    if !error.is_null() {
        let err_msg = unsafe { llvm_sys::error::LLVMGetErrorMessage(error) };
        let message = Message::new(unsafe { CStr::from_ptr(err_msg) });
        return Err(message.to_str().to_string());
    }

    if let Some(hook) = compiler_hook {
        // Run compiler hook on optimized human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("opt.ll"));

        // Running a disassembler would be a bit of a pain, so run codegen as assembly
        let assembly = target_machine
            .emit_to_memory_buffer(&linked.clone(), LLVMCodeGenFileType::LLVMAssemblyFile)?
            .to_vec();
        hook(&assembly, String::from("asm"))
    }

    // VEGA-FIX T13:00 23 giu: bypass emit_to_memory_buffer (che elimina kernel exports
    // via REQUIRED AMDGPU passes hardcoded) → usa clang 21 ESTERNO con
    // -mllvm -amdgpu-internalize-symbols=false per preservare TUTTI i kernel.
    let object_file = {
        // clang di Marconi: ZLUDA_CLANG, poi %MARCONI_ROOT%\compiler\bin\clang.exe, poi la cartella storica.
        let clang_buf = trova_clang();
        let clang_path = clang_buf.as_path();
        if clang_path.exists() {
            // Dump LLVM IR su tempfile + clang esterno → object con kernel preservati
            let ir_path = NamedTempFile::with_prefix("zluda_ir.ll")
                .map_err(|_| ("Failed to create IR tempfile").to_string())?
                .into_temp_path();
            let clang_obj_path = NamedTempFile::with_prefix("zluda_clang.o")
                .map_err(|_| ("Failed to create clang obj tempfile").to_string())?
                .into_temp_path();
            let mcpu_arg = format!("-mcpu={}", arch_only);
            // VEGA 3 lug: ciclo con resilienza stack-size. Se clang fallisce perche' funzioni
            // dead-code (foreach complex<double>) sforano il limite di stack gfx906, le stubbo
            // nel modulo e ricompilo. Max 8 giri (le funzioni oversized sono poche).
            let mut ir_text = String::from_utf8_lossy(linked.print_module_to_string().to_bytes()).into_owned();
            let mut obj_bytes: Option<Vec<u8>> = None;
            // [2S, 22 ott 2026] Ottimizzazione. Da giugno questa chiamata non passava -O (clang con IR
            // in ingresso => -O0) e la pipeline in-process e' solo "verify": tutti i kernel tradotti giravano a -O0,
            // ogni valore in scratch. In piu' le funzioni __zluda_ptx_impl_* (bfe, div, sqrt...) non si espandevano
            // in linea: i kernel hanno target-features diverse dalle loro, e senza -ignore-tti-inline-compatible
            // l'inliner le rifiuta; la chiamata rimasta fa assumere 16 KiB di stack per corsia (has_recursion=1).
            // Effetto misurato: somma fp32 a 26 GB/s contro 567 di copy_. Qui -O3 con le opzioni d'inlining di
            // ZLUDA originale; se un modulo non compila a un livello si scende al successivo (O3 -> O2 -> O1 -> O0).
            // ZLUDA_OPT_LIVELLO=O0|O1|O2|O3 fissa il livello di partenza (O0 = comportamento di prima).
            let livelli = ["-O3", "-O2", "-O1", "-O0"];
            let mut livello = livelli
                .iter()
                .position(|l| std::env::var("ZLUDA_OPT_LIVELLO").map(|v| format!("-{}", v.trim()) == *l).unwrap_or(false))
                .unwrap_or(livelli.len() - 1); // predefinito -O0 come la versione in uso; O3 solo con ZLUDA_OPT_LIVELLO
            for _attempt in 0..12 {
                fs::write(&ir_path, ir_text.as_bytes())
                    .map_err(|_| ("Failed to write IR").to_string())?;
                if crate::diag("zluda_clang_compile.log").is_ok() {
                    let _ = fs::write(crate::diag_dir().join("zluda_last_ir.ll"), ir_text.as_bytes());
                }
                let output = std::process::Command::new(clang_path)
                    .args(&[
                        "-x", "ir",
                        ir_path.to_str().ok_or("invalid IR path".to_string())?,
                        "-target", "amdgcn-amd-amdhsa",
                        &mcpu_arg,
                        "-nogpulib",  // VEGA-FIX T12:50: IR già ha device libs linked (ockl/ocml)
                        // -fgpu-rdc RIMOSSO (name-preservation L2): decora i kernel symbol.
                        "-fno-gpu-rdc",
                        livelli[livello],
                        "-mllvm", "-ignore-tti-inline-compatible",
                        "-mllvm", "-inline-threshold=2250",
                        "-mllvm", "-inlinehint-threshold=3250",
                        "-mllvm", "-amdgpu-internalize-symbols=false",
                        "-mllvm", "--amdgpu-attributor-enable=false",
                        "-mllvm", "-amdgpu-enable-lower-module-lds=false",  // L3: preserve module-LDS 1:1
                        "-mcode-object-version=4",  // amdhip64 v5.7 vuole COV=4
                        "-c", "-o",
                        clang_obj_path.to_str().ok_or("invalid obj path".to_string())?,
                    ])
                    .output()
                    .map_err(|e| format!("clang invocation failed: {}", e))?;
                if let Ok(mut f) = crate::diag("zluda_clang_compile.log") {
                    use std::io::Write;
                    let _ = writeln!(f, "[clang] attempt={} livello={} status={:?} stderr_len={}", _attempt, livelli[livello], output.status.code(), output.stderr.len());
                    if !output.stderr.is_empty() {
                        let _ = f.write_all(b"--- stderr ---\n");
                        let _ = f.write_all(&output.stderr);
                        let _ = f.write_all(b"\n--- end ---\n");
                    }
                }
                if output.status.success() {
                    obj_bytes = Some(fs::read(&clang_obj_path).map_err(|_| ("Failed to read clang output".to_string()))?);
                    break;
                }
                let stderr = String::from_utf8_lossy(&output.stderr);
                // resilienza: stubba (via testo IR) le funzioni dead-code che sforano lo stack
                let (new_ir, stubbed) = stub_oversized_functions_text(&ir_text, &stderr);
                if stubbed == 0 {
                    if livello + 1 < livelli.len() {
                        // nessuna funzione da stubbare: errore del livello d'ottimizzazione, scendo di uno
                        if let Ok(mut f) = crate::diag("zluda_clang_compile.log") {
                            use std::io::Write;
                            let _ = writeln!(f, "[clang] fallito a {}, riprovo a {}", livelli[livello], livelli[livello + 1]);
                        }
                        livello += 1;
                        continue;
                    }
                    return Err(format!("clang failed: {}", stderr));
                }
                ir_text = new_ir;
                if let Ok(mut f) = crate::diag("zluda_clang_compile.log") {
                    use std::io::Write;
                    let _ = writeln!(f, "[clang] stubbate {} funzioni oversized (stack), ricompilo", stubbed);
                }
            }
            obj_bytes.ok_or_else(|| "clang failed: stack-size irresolubile dopo stub".to_string())?
        } else {
            // Fallback: codegen in-process se clang esterno non disponibile
            target_machine
                .emit_to_memory_buffer(&linked, LLVMCodeGenFileType::LLVMObjectFile)?
                .to_vec()
        }
    };

    if let Some(hook) = compiler_hook {
        // Run compiler hook for object file
        hook(&object_file, String::from("o"));
    }

    let object_path = NamedTempFile::with_prefix("zluda.o")
        .map_err(|_| ("Failed to create temporary file").to_string())?
        .into_temp_path();
    let executable_path = NamedTempFile::with_prefix("zluda.elf")
        .map_err(|_| ("Failed to create temporary file").to_string())?
        .into_temp_path();

    fs::write(&object_path, &object_file)
        .map_err(|_| ("Failed to write object file").to_string())?;

    let object_path_cstr = path_to_cstring(&object_path)?;
    let executable_path_cstr = path_to_cstring(&executable_path)?;

    let mut err = std::ptr::null_mut();
    let result = unsafe {
        LLVMZludaLinkWithLLD(
            object_path_cstr.as_ptr(),
            executable_path_cstr.as_ptr(),
            &mut err,
        )
    };
    if result != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }

    let executable =
        fs::read(&executable_path).map_err(|_| ("Failed to read executable file").to_string())?;

    if let Some(hook) = compiler_hook {
        // Run compiler hook for final executable
        hook(&executable, String::from("elf"));
    }

    Ok(executable)
}

// VEGA 3 lug: resilienza del modulo alle funzioni che sforano il limite di stack AMDGPU.
// I kernel foreach `complex<double>` (multi_tensor_apply, da _foreach_pow complesso) vogliono
// >128KB di stack privato = oltre il limite HARD di gfx906, e clang fa fallire l'INTERO modulo
// (atomico) → cade anche il foreach float che serve. Queste funzioni sono dead-code per il
// nostro workload (mai tensori complessi). Le stubbiamo (ret void / unreachable, stack ~0) e
// ricompiliamo: il modulo compila, il foreach float funziona nativo.
// Ritorna quante funzioni sono state stubbate (0 = nessuna stack-size da gestire → errore reale).
// Manipolazione ROBUSTA via testo IR (niente chirurgia LLVM-C fragile). Sostituisce il CORPO
// delle funzioni oversized con `unreachable` (stack ~0). Matcha la chiusura funzione come `}`
// a inizio riga (le graffe interne — tipi {i32}, array — non sono mai a colonna 0 da sole).
// Ritorna (nuovo_IR, quante funzioni stubbate).
fn stub_oversized_functions_text(ir: &str, stderr: &str) -> (String, usize) {
    // raccogli i nomi mangled dall'errore
    let mut names: Vec<String> = Vec::new();
    for line in stderr.lines() {
        if !(line.contains("stack frame size") && line.contains("exceeds limit")) {
            continue;
        }
        let marker = "in function '";
        if let Some(s) = line.find(marker) {
            let rest = &line[s + marker.len()..];
            if let Some(e) = rest.find('\'') {
                names.push(rest[..e].to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    let mut out = ir.to_string();
    let mut count = 0usize;
    for name in &names {
        let needle = format!("@{}(", name);
        let at = match out.find(&needle) {
            Some(p) => p,
            None => continue,
        };
        // graffa che apre il corpo: primo "{\n" dopo la firma
        let brace = match out[at..].find("{\n") {
            Some(b) => at + b,
            None => continue,
        };
        let body_start = brace + 1; // subito dopo '{'
        // chiusura funzione: prima "\n}" (la '}' a inizio riga)
        let close = match out[body_start..].find("\n}") {
            Some(c) => body_start + c,
            None => continue,
        };
        // sostituisci il corpo con un blocco minimo: unreachable (valido per qualsiasi firma)
        out.replace_range(body_start..close, "\n  unreachable");
        count += 1;
    }
    (out, count)
}

fn init_globals() -> Result<(), String> {
    static INIT_AMDGPU: OnceLock<Result<(), Message>> = OnceLock::new();
    INIT_AMDGPU
        .get_or_init(|| {
            let common_options = vec![
                // Uncomment for LLVM debug
                //c"-debug",
                // Uncomment to save passes
                // c"-print-before-all",
                c"llvm_zluda",
                c"-ignore-tti-inline-compatible",
                // c"-amdgpu-early-inline-all=true",
                c"-amdgpu-internalize-symbols=false",  // VEGA-FIX T12:45: preserva kernel exports per torch cuModuleGetFunction
                c"-amdhsa-code-object-version=5",
            ]
            .into_iter();
            let opt_options = if cfg!(debug_assertions) {
                vec![]
            } else {
                vec![
                    // default inlining threshold times 10
                    c"-inline-threshold=2250",
                    c"-inlinehint-threshold=3250",
                ]
            };
            let llvm_args_ptrs: Vec<*const i8> = common_options
                .chain(opt_options)
                .map(|s| s.as_ptr())
                .collect();
            let mut err_msg = std::ptr::null_mut();
            let success = unsafe {
                LLVMZludaParseCommandLineOptions(
                    llvm_args_ptrs.len() as i32,
                    llvm_args_ptrs.as_ptr(),
                    &mut err_msg,
                )
            };
            if !success {
                return Err(Message::new(unsafe { CStr::from_ptr(err_msg) }));
            }
            unsafe { LLVMInitializeAMDGPUTargetInfo() };
            unsafe { LLVMInitializeAMDGPUTarget() };
            unsafe { LLVMInitializeAMDGPUTargetMC() };
            unsafe { LLVMInitializeAMDGPUAsmPrinter() };
            Ok(())
        })
        .as_ref()
        .map(|()| ())
        .map_err(|e| e.to_str().to_owned())
}

fn trova_clang() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ZLUDA_CLANG") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(r) = std::env::var("MARCONI_ROOT") {
        let p = std::path::Path::new(&r).join("compiler").join("bin").join("clang.exe");
        if p.exists() {
            return p;
        }
    }
    std::path::PathBuf::from(r"C:\rocm_5_7_expansion_pack\compiler\bin\clang.exe")
}
