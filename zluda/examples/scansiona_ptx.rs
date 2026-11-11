//! Analizza OGNI PTX contenuto nei fatbin di una DLL con il parser vero di ZLUDA.
//!
//!     cargo run -p zluda --example scansiona_ptx --profile release-lto -- <file.dll>
//!
//! [2S, 17 ott 2026] Nato dal difetto di torch.unique: un modulo PTX che
//! il parser non riconosce diventa, nella build di rilascio, un modulo VUOTO senza
//! alcun errore, e i suoi kernel spariscono. Era `bfind.shiftamt` in Embedding.cu.
//! Eseguire un carico trova solo i moduli che quel carico usa; questo strumento li
//! trova TUTTI, senza eseguire niente, con le stesse funzioni che usa ZLUDA
//! (dark_api per i fatbin e la decompressione, ptx_parser per l'analisi).
//!
//! Riporta i fatbin in cui NESSUN PTX si analizza (sarebbero moduli vuoti) e, per
//! istruzione, quante volte il parser non la riconosce.

use std::collections::BTreeMap;

use cuda_types::dark_api::{FatbinFileHeader, FatbinHeader};
use dark_api::fatbin::FatbinSubmodule;

fn prima_entry(ptx: &str) -> String {
    ptx.find(".entry ")
        .map(|i| {
            let resto = &ptx[i + 7..];
            let fine = resto.find(|c: char| c == '(' || c.is_whitespace()).unwrap_or(resto.len());
            resto[..fine.min(110)].to_string()
        })
        .unwrap_or_else(|| "(nessuna .entry)".to_string())
}

/// Dal Debug di un errore del parser estrae (variante, mnemonico dell'istruzione).
fn classifica(errore: &str) -> (String, String) {
    let variante = errore.split('(').next().unwrap_or(errore).to_string();
    let dentro = errore.splitn(2, '"').nth(1).unwrap_or("");
    let mnemonico = dentro
        .trim_start_matches(|c: char| c == '@' || c == '!' || c == '%' || c.is_alphanumeric() && false)
        .split_whitespace()
        .find(|t| !t.starts_with('@'))
        .unwrap_or("")
        .trim_end_matches(';')
        .to_string();
    (variante, mnemonico)
}

fn main() {
    let percorso = std::env::args().nth(1).expect("uso: scansiona_ptx <file.dll>");
    let byte = std::fs::read(&percorso).expect("lettura del file");
    // copia allineata a 8 byte: FatbinHeader e' #[repr(C, align(8))]
    let mut allineato: Vec<u64> = vec![0; (byte.len() + 7) / 8];
    unsafe {
        std::ptr::copy_nonoverlapping(byte.as_ptr(), allineato.as_mut_ptr() as *mut u8, byte.len());
    }
    let base = allineato.as_ptr() as *const u8;
    let magia = u32::from_le_bytes(FatbinHeader::MAGIC);

    let (mut n_fatbin, mut n_ptx, mut n_ptx_falliti, mut n_decomp_fallite) = (0usize, 0usize, 0usize, 0usize);
    let mut moduli_vuoti: Vec<(usize, String, String)> = Vec::new();
    let mut per_istruzione: BTreeMap<(String, String), usize> = BTreeMap::new();

    let mut off = 0usize;
    while off + 16 <= byte.len() {
        let m = u32::from_le_bytes(byte[off..off + 4].try_into().unwrap());
        if m != magia {
            off += 8;
            continue;
        }
        let intestazione = unsafe { &*(base.add(off) as *const FatbinHeader) };
        let totale = intestazione.header_size as usize + intestazione.files_size as usize;
        if intestazione.version != FatbinHeader::VERSION
            || intestazione.header_size < 16
            || off + totale > byte.len()
        {
            off += 8;
            continue;
        }
        n_fatbin += 1;

        let mut ptx_qui = 0usize;
        let mut ptx_buoni = 0usize;
        let mut esempio = String::new();
        let mut primo_errore = String::new();
        let sub = FatbinSubmodule::new(intestazione);
        let mut file = sub.get_files();
        while let Some(f) = unsafe { file.next() } {
            let f = match f {
                Ok(f) => f,
                Err(_) => break,
            };
            if f.header.kind != FatbinFileHeader::HEADER_KIND_PTX {
                continue;
            }
            let contenuto = match unsafe { f.get_or_decompress_content() } {
                Ok(c) => c,
                Err(_) => {
                    n_decomp_fallite += 1;
                    continue;
                }
            };
            let testo = match std::str::from_utf8(&contenuto) {
                Ok(t) => t.trim_end_matches('\0'),
                Err(_) => {
                    n_decomp_fallite += 1;
                    continue;
                }
            };
            n_ptx += 1;
            ptx_qui += 1;
            if esempio.is_empty() {
                esempio = prima_entry(testo);
            }
            let normalizzato;
            let testo = if ptx_parser::va_normalizzato(testo) {
                normalizzato = ptx_parser::normalizza(testo);
                normalizzato.as_str()
            } else {
                testo
            };
            match ptx_parser::parse_module_checked(testo) {
                Ok(_) => ptx_buoni += 1,
                Err(errori) => {
                    n_ptx_falliti += 1;
                    let mut visti = std::collections::BTreeSet::new();
                    for e in &errori {
                        let s = format!("{:?}", e);
                        let chiave = classifica(&s);
                        if visti.insert(chiave.clone()) {
                            *per_istruzione.entry(chiave).or_insert(0) += 1;
                        }
                        if primo_errore.is_empty() {
                            primo_errore = s.chars().take(110).collect();
                        }
                    }
                }
            }
        }
        if ptx_qui > 0 && ptx_buoni == 0 {
            moduli_vuoti.push((off, esempio, primo_errore));
        }
        off += totale.max(8);
        off = (off + 7) & !7;
    }

    println!("file: {}", percorso);
    println!("fatbin trovati: {}   PTX analizzati: {}   PTX non analizzabili: {}   decompressioni fallite: {}",
        n_fatbin, n_ptx, n_ptx_falliti, n_decomp_fallite);
    println!();
    println!("FATBIN SENZA NESSUN PTX ANALIZZABILE (sotto ZLUDA diventano moduli VUOTI): {}", moduli_vuoti.len());
    for (o, e, err) in &moduli_vuoti {
        println!("  @0x{:08x}  {}\n      {}", o, e, err);
    }
    println!();
    println!("istruzioni non riconosciute (numero di PTX in cui compaiono):");
    for ((variante, mnemonico), n) in &per_istruzione {
        println!("  {:5}  {:24} {}", n, variante, mnemonico);
    }
}
