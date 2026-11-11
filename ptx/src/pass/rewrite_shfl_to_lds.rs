// VEGA TASK D 30 giu 2026 T07:50  (worker 3/5)
//
// Pass opt-in (feature `wave64_lds_scratchpad`) che riscrive ogni occorrenza di
// `ShflSync { mode: bfly|idx|up|down }` in una chiamata a una funzione device
// helper `shfl_lds_<mode>_b32[_pred]` implementata in `lib/zluda_ptx_impl.cpp`.
//
// Razionale:
//   - L'implementazione attuale di `__zluda_ptx_impl_shfl_sync_*_b32_pred`
//     usa `__builtin_amdgcn_ds_bpermute` (LDS hardware permute, 1 istruzione,
//     no barriera esplicita).
//   - Su gfx906 wave64, quando PyTorch ATen invoca reduction butterfly composta
//     (log2(64) = 6 steps consecutivi) nel backward di `aten::native_layer_norm`,
//     lo scheduling cross-wave del ds_bpermute innesca TDR Windows WDDM 2s
//     (composite 709 - bisect B6).
//   - L'alternativa LDS-scratchpad write + __threadfence_block + __syncthreads
//     + read espone safepoint deterministici che il driver tratta come progresso.
//   - Costo: ~3-5x latenza per shfl singola, ma elimina la composite 709
//     backward LN2 strutturalmente.
//
// Trigger: questo pass NON ha effetto se la feature `wave64_lds_scratchpad`
// non e' attiva (function no-op).
//
// Il pass va inserito DOPO `replace_instructions_with_functions` e PRIMA di
// `hoist_globals`, per riscrivere le `Call { func = "__zluda_ptx_impl_shfl_sync_*" }`
// gia' create dal pass standard. Cosi' non duplichiamo logica di setup args.
//
// IMPORTANTE: i nomi nel resolver vengono prefissati con `__zluda_ptx_impl_`
// dal pass `replace_instructions_with_functions::get_or_declare_function`
// (vedi pass/mod.rs costante `ZLUDA_PTX_PREFIX`). Il match deve quindi
// avvenire sui nomi GIA' prefissati.
//
// Vedi sediment ZLUDA.md categoria TASK-D + REGOLE_TASSATIVE.md regola 17.

use super::*;

/// Mapping nome funzione (prefissato) standard -> nome funzione LDS scratchpad.
/// I 4 mode (bfly/idx/up/down) hanno entrambe le varianti _pred e plain (8 totali).
///
/// `#[allow(dead_code)]` perche' usata solo dietro `cfg(feature = "wave64_lds_scratchpad")`
/// e dai test #[cfg(test)] — senza feature attiva il compilatore vede solo i test.
#[allow(dead_code)]
fn lds_replacement_name(orig: &str) -> Option<&'static str> {
    match orig {
        // varianti con predicate (ritornano vec<2,u32>)
        "__zluda_ptx_impl_shfl_sync_bfly_b32_pred" => {
            Some("__zluda_ptx_impl_shfl_lds_bfly_b32_pred")
        }
        "__zluda_ptx_impl_shfl_sync_idx_b32_pred" => {
            Some("__zluda_ptx_impl_shfl_lds_idx_b32_pred")
        }
        "__zluda_ptx_impl_shfl_sync_up_b32_pred" => {
            Some("__zluda_ptx_impl_shfl_lds_up_b32_pred")
        }
        "__zluda_ptx_impl_shfl_sync_down_b32_pred" => {
            Some("__zluda_ptx_impl_shfl_lds_down_b32_pred")
        }
        // varianti plain (ritornano u32)
        "__zluda_ptx_impl_shfl_sync_bfly_b32" => Some("__zluda_ptx_impl_shfl_lds_bfly_b32"),
        "__zluda_ptx_impl_shfl_sync_idx_b32" => Some("__zluda_ptx_impl_shfl_lds_idx_b32"),
        "__zluda_ptx_impl_shfl_sync_up_b32" => Some("__zluda_ptx_impl_shfl_lds_up_b32"),
        "__zluda_ptx_impl_shfl_sync_down_b32" => Some("__zluda_ptx_impl_shfl_lds_down_b32"),
        _ => None,
    }
}

/// Riscrive ogni Call shfl_sync_* -> shfl_lds_* a livello di nome simbolo nel
/// resolver. Signature, argomenti, dispatch del linker LLVM restano identici —
/// cambia solo l'implementazione linkata in `zluda_ptx_impl.bc`.
#[cfg(feature = "wave64_lds_scratchpad")]
pub(super) fn run<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<ptx_parser::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<Vec<Directive2<ptx_parser::Instruction<SpirvWord>, SpirvWord>>, TranslateError> {
    use std::borrow::Cow;

    // Step 1: raccogli (id, new_name) per i function-id da rinominare.
    // Iterazione separata per evitare double-borrow su resolver.
    let mut to_rename: Vec<(SpirvWord, &'static str)> = Vec::new();
    for (id, entry) in resolver.ident_map.iter() {
        if let Some(name) = entry.name.as_ref() {
            if let Some(new_name) = lds_replacement_name(name.as_ref()) {
                to_rename.push((*id, new_name));
            }
        }
    }
    // Step 2: applica i rename in-place. Tutte le Call esistenti che
    // referenziano questi function-id chiameranno al link il nuovo simbolo
    // `__zluda_ptx_impl_shfl_lds_*_b32[_pred]` definito in zluda_ptx_impl.bc.
    for (id, new_name) in to_rename {
        if let Some(entry) = resolver.ident_map.get_mut(&id) {
            entry.name = Some(Cow::Owned(new_name.to_string()));
        }
    }
    Ok(directives)
}

/// No-op se la feature non e' attiva. Mantiene la chiamata in mod.rs sempre
/// presente, senza overhead a runtime (compile-time strip).
#[cfg(not(feature = "wave64_lds_scratchpad"))]
pub(super) fn run<'input>(
    _resolver: &mut GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<ptx_parser::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<Vec<Directive2<ptx_parser::Instruction<SpirvWord>, SpirvWord>>, TranslateError> {
    Ok(directives)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lds_replacement_name_known_modes_prefixed() {
        // Le 8 varianti che ci aspettiamo siano tradotte (nomi prefissati).
        assert_eq!(
            lds_replacement_name("__zluda_ptx_impl_shfl_sync_bfly_b32_pred"),
            Some("__zluda_ptx_impl_shfl_lds_bfly_b32_pred")
        );
        assert_eq!(
            lds_replacement_name("__zluda_ptx_impl_shfl_sync_idx_b32"),
            Some("__zluda_ptx_impl_shfl_lds_idx_b32")
        );
        assert_eq!(
            lds_replacement_name("__zluda_ptx_impl_shfl_sync_up_b32_pred"),
            Some("__zluda_ptx_impl_shfl_lds_up_b32_pred")
        );
        assert_eq!(
            lds_replacement_name("__zluda_ptx_impl_shfl_sync_down_b32"),
            Some("__zluda_ptx_impl_shfl_lds_down_b32")
        );
    }

    #[test]
    fn lds_replacement_name_non_prefixed_returns_none() {
        // Nomi non prefissati non devono matchare — il pass viene
        // eseguito DOPO replace_instructions_with_functions, quindi i nomi
        // sono SEMPRE prefissati con `__zluda_ptx_impl_`.
        assert_eq!(lds_replacement_name("shfl_sync_bfly_b32_pred"), None);
        assert_eq!(lds_replacement_name("shfl_sync_idx_b32"), None);
    }

    #[test]
    fn lds_replacement_name_unknown_returns_none() {
        assert_eq!(lds_replacement_name("memcpy"), None);
        assert_eq!(lds_replacement_name("__zluda_ptx_impl_memcpy"), None);
        assert_eq!(lds_replacement_name("__zluda_ptx_impl_shfl_sync_xyz_b32"), None);
        assert_eq!(lds_replacement_name(""), None);
    }
}
