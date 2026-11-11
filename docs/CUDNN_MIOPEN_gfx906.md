# cuDNN → MIOpen in ZLUDA su gfx906

**2 agosto 2026 — 2S · 2Solution, Radeon VII.** È il resoconto di come `zluda_dnn` (la
`cudnn64_8.dll` di ZLUDA) è arrivato a coprire tutti gli 86 simboli cuDNN che PyTorch importa.
Viene dal resoconto «MIOpen e cuDNN su gfx906»: la parte su MIOpen (cause, shim, il difetto WrW
3×3) sta in [miopen-gfx906](https://github.com/2s-2solution/miopen-gfx906), `docs/MIOPEN_gfx906.md`.

È un documento storico. I banchi citati erano script della macchina di prova: quelli segnati
«sì» nella tabella del § 8 stanno in `banchi/` di miopen-gfx906, gli altri non sono pubblicati.
Il modo attuale di provare ZLUDA è la verifica del pacchetto delle Releases (vedi il README).
PyTorch 2.3.1+cu118 è il programma CUDA usato come banco di prova.

---

## 1. Dove siamo

| percorso | stato |
|---|---|
| MIOpen chiamato direttamente | **19 prove su 19** |
| Convoluzioni da PyTorch (avanti, indietro dati, indietro pesi) | **corrette**, 2-6e-07 |
| `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AvgPool2d` | **corretti** |
| Addestramento di una CNN, 5 passi | pesi identici alla CPU a **5.9e-08** |
| `torch.backends.cudnn.benchmark = True` | **funziona** (3.5e-07) |
| RNN / LSTM / GRU | **10 prove su 10** |
| Spatial transformer | **4 su 4** |
| Simboli cuDNN che PyTorch chiede | **86 su 86 implementati** |

Prima del 2 agosto: MIOpen non funzionava affatto, e PyTorch non riusciva
nemmeno a inizializzare cuDNN.

Tempi della CNN di prova (8x3x32x32, due conv + due batch norm):

| | ms |
|---|---|
| primo passo (compilazioni dei kernel MIOpen) | ~6400 |
| secondo passo (ricerca algoritmi per ogni forma nuova) | ~180 |
| **a regime, dal terzo** | **9** |

---

## 4. Il traduttore cuDNN (`zluda_dnn`)

### 4.1 Il crate si costruisce, contrariamente a quanto avevo scritto

Nel `Cargo.toml` di `zluda_dnn` avevo annotato che il crate non era
riattivabile come cdylib per una collisione di simboli. Falso: i due crate
separati che dicevo servissero **esistono già accanto** —`zluda_dnn8` e
`zluda_dnn9` — e riesportano un solo ramo ciascuno con `#[no_mangle]`. Avevo
provato ad aggiungere `crate-type` al crate sbagliato.

```
cargo build -p zluda_dnn8 --profile release-lto
```

Riesce al primo colpo. La DLL prodotta va copiata **in due posti**:

```
<cartella di ZLUDA>\cudnn64_8.dll
<venv>\Lib\site-packages\torch\lib\cudnn64_8.dll
```

PyTorch carica la propria copia: aggiornare solo la prima non serve a niente.
**Verificare sempre la copia con l'impronta**: una DLL in uso non si sovrascrive
e l'errore è silenzioso.

### 4.2 Il metodo: leggere gli import, non indovinare

`torch_cuda.dll` chiede **86 simboli** a `cudnn64_8.dll`. Incrociandoli con le
liste cablate in `lib.rs` si ottiene l'elenco esatto di quelli che contano.
Lo strumento era un parser PE senza dipendenze (`pe.py`, non pubblicato); `dumpbin /imports`
di Visual Studio dà lo stesso elenco.

E `diag_cudnn_sequenza.py`, che rifà la sequenza di PyTorch chiamata per
chiamata stampando ogni codice di ritorno: dice quale passo cede, invece di
lasciar dedurre da un «cuDNN error» generico.

### 4.3 Cosa mancava, in ordine di scoperta

| # | cosa | perché fermava tutto |
|---|---|---|
| a | `cudnnGetProperty` | PyTorch legge major/minor/patch da qui, **non** da `cudnnGetVersion`. Sullo stub leggeva memoria non inizializzata: «compiled against (8,7,0), found (0,128,0)» |
| b | `VERSIONE_CUDNN` col formato sbagliato | le 8.x sono `major*1000 + minor*100 + patch`; avevo scritto il formato delle 9.x (`*10000`) |
| c | `cudnnSetStream` / `cudnnGetStream` | PyTorch lega l'handle allo stream dentro `getCudnnHandle()`, prima di ogni altra cosa |
| d | la famiglia **Nd** (6 funzioni) | PyTorch non chiama **mai** le varianti 2d/4d che il traduttore aveva |
| e | ricerca algoritmo e spazio di lavoro all'indietro (4) | PyTorch usava una dimensione mai scritta → access violation |
| f | la Find prima della backward | MIOpen: «No invoker was registered for convolution backward. Was find executed?» |
| g | batch norm, varianti `Ex` (5) | PyTorch usa solo quelle |
| h | le tre ricerche `...AlgorithmEx` | `cudnn.benchmark = True` faceva cadere il processo |

Dettagli su alcune:

**(d) La famiglia Nd.** `cudnnSetTensorNdDescriptor`, `...NdDescriptorEx`,
`cudnnSetFilterNdDescriptor`, `cudnnSetConvolutionNdDescriptor`,
`cudnnSetConvolutionGroupCount`, `cudnnGetConvolutionNdForwardOutputDim`.
Le controparti MIOpen esistono tutte, corrispondenza 1:1. Scritte in stile
"Family B" (passaggio diretto) perché le liste di dimensioni sono array: la
conversione automatica `FromCuda` le tratterebbe come riferimenti a un singolo
intero.

**(f) La Find registra l'invoker.** MIOpen non permette di scegliere
l'algoritmo a tavolino: è la ricerca a registrare l'invoker per quella forma.
Le impl esistenti passavano un algoritmo fisso e MIOpen rifiutava
(`convolutionocl.cpp:1258`).

**(g) Batch norm Ex.** Con `bnOps = CUDNN_BATCHNORM_OPS_BN` sono equivalenti
alle non-Ex già presenti, e si delega. Per le modalità con attivazione fusa si
restituisce `NOT_SUPPORTED` invece di fingere: MIOpen non ha un equivalente e
PyTorch ripiega sul proprio percorso.

### 4.4 La ricerca va messa in cache

Fatta girare a ogni chiamata all'indietro costava **230 ms per passo** su una
rete minuscola. `MIOPEN_FIND_MODE` non cambia nulla (provate tutte e quattro le
modalità: 8-9 s sui cinque passi in ogni caso).

Aggiunte due mappe in `ContextCache` (`algo_cache_bwd_dati`,
`algo_cache_bwd_pesi`) sul modello di quella che il percorso in avanti aveva
già. L'invoker resta registrato nell'handle, quindi saltare la ricerca dalla
seconda volta in poi è lecito.

| | prima | dopo |
|---|---|---|
| passo a regime | 230 ms | **9 ms** |

---

## 6. Variabili d'ambiente

Nel codice di prova si impostavano prima dell'import di torch: vanno lette una volta sola, alla
prima operazione che le riguarda. Oggi le due di MIOpen le imposta il kit
[rocm-marconi](https://github.com/2s-2solution/rocm-marconi) per l'utente.

```python
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
os.environ.setdefault("MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1", "0")
```

| variabile | senza di essa |
|---|---|
| `TORCH_CUDNN_V8_API_DISABLED=1` | il 2 agosto: «CUDNN_BACKEND_TENSOR_DESCRIPTOR: cudnnCreate Descriptor Failed», perché PyTorch 2.3 preferisce l'API backend v8, allora non implementata. Dopo il § 10.3 è facoltativa: la v7 resta più veloce |
| `MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3=0` | nessun errore, gradienti sbagliati sul primo strato RGB |
| `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1=0` | nessun errore, gradienti sbagliati (misurato l'11 nov 2026 con la MIOpen del kit: scarto 1,2) |

rocBLAS trova la sua libreria attraverso `ROCBLAS_TENSILE_LIBPATH`, che imposta il kit
(`<kit>\bin\rocblas\library`).

---

## 7. Inventario dei file toccati

### Binari

Sulla macchina di prova erano state sostituite `cudnn64_8.dll` (nella cartella di ZLUDA e in
`torch\lib`) e i due shim di MIOpen (`hiprtc07013.dll`, `amd_cmgr3.dll`). Oggi le DLL di ZLUDA
arrivano dal pacchetto delle Releases, che ne controlla l'impronta dopo la copia; gli shim di
MIOpen sono in miopen-gfx906. Se le convoluzioni tornano a fallire, **la prima cosa da
controllare sono le impronte delle DLL caricate**: una copia vecchia rimessa al suo posto non dà
errori.

### Sorgenti modificati

| file | cosa |
|---|---|
| `zluda_dnn\src\impl.rs` | ~20 funzioni nuove, la cache all'indietro, la correzione di `VERSIONE_CUDNN` |
| `zluda_dnn\src\lib.rs` | le nuove funzioni cablate nella lista Family B |
| `zluda_dnn\Cargo.toml` | corretta la nota sbagliata sul cdylib |
| `zluda_nvrtc\src\lib.rs` | corretto il commento sbagliato su hipRTC 7.x (solo commento, nessun cambio di comportamento) |

---

## 8. Banchi di prova

Quelli segnati «sì» sono in `banchi/` di [miopen-gfx906](https://github.com/2s-2solution/miopen-gfx906);
gli altri erano script della macchina di prova, citati per completezza e **non pubblicati**.
Per ZLUDA oggi la prova è `verifica.cmd` del pacchetto delle Releases.

| banco | nel repo | cosa verifica | atteso |
|---|---|---|---|
| `prova_miopen_completo.py` | sì | MIOpen diretto: conv in 6 forme + fp16, i due all'indietro, pooling, 5 attivazioni, batch norm | **19/19** |
| `diag_wgrad_forme.py` | sì | gradiente sui pesi al variare della forma | tutte OK |
| `prova_conv_torch.py` | no | conv da PyTorch, avanti e indietro, più 4 strati `nn` | tutte OK |
| `prova_cnn_cinque_passi.py` | no | addestramento di una CNN, GPU contro CPU | **TUTTO CORRETTO** |
| `diag_cudnn_sequenza.py` | no | la sequenza cuDNN un passo alla volta, con i codici di ritorno | 15 passi ok |
| `prova_miopen_diretto.py` | no | una singola convoluzione MIOpen via ctypes | corretta |
| `prova_rnn_torch.py` | no | RNN, LSTM e GRU, avanti e indietro | **10/10** |
| `prova_spaziale_torch.py` | no | griglia affine e campionamento bilineare | **4/4** |

`prova_miopen_completo.py` prende la cartella di MIOpen come argomento (o `MIOPEN_BIN`); con
la MIOpen senza patch va lanciato con
`MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3=0`, gli altri se lo impostano da soli.

---

## Diagnostica cuDNN

La sequenza cuDNN un passo alla volta (`diag_cudnn_sequenza.py`): quale chiamata cede.

---

## 10. Copertura: 86 simboli su 86

Al 2 agosto 2026 **tutti** i simboli che PyTorch chiede a `cudnn64_8.dll` sono
implementati e verificati. Si era partiti da 27.

| gruppo | quanti | come |
|---|---|---|
| convoluzione v7, batch norm, descrittori Nd, stream | 41 | corrispondenza MIOpen diretta |
| descrittori di attivazione e pooling, lettura descrittori, conv+bias+attivazione | 8 | corrispondenza diretta |
| dropout | 5 | corrispondenza diretta |
| CTC loss | 6 | corrispondenza diretta |
| RNN / LSTM / GRU | 13 | corrispondenza diretta **piu' traduzione della disposizione dei pesi** |
| spatial transformer | 7 | **kernel HIP nostri**, MIOpen non ce l'ha |
| API backend v8 | 6 | **motore a grafo nostro**, tradotto sulla v7 |

### 10.1 RNN: due incompatibilita' che non danno errore

Le funzioni RNN v6 di cuDNN sono state rimosse in cuDNN 9 ed esistono solo nel
ramo 8 — ma i tipi ci sono in entrambi, quindi si scrivono come le altre.

**L'ordine delle porte non coincide.**

| | cuDNN | MIOpen |
|---|---|---|
| LSTM | ingresso, dimenticanza, **nuova memoria, uscita** | ingresso, dimenticanza, **uscita, nuova memoria** |
| GRU | **azzeramento, aggiornamento**, nuova memoria | **aggiornamento, azzeramento**, nuova memoria |

Passare l'indice cosi' com'e' darebbe una rete che si addestra senza errori e
calcola con le porte scambiate.

**La disposizione complessiva differisce da due strati in su.**

```
cuDNN:  [strato0: matrici, polarizzazioni][strato1: matrici, polarizzazioni]...
MIOpen: [tutte le matrici di tutti gli strati][tutte le polarizzazioni]
```

Con un solo strato coincidono; da due in su no. E PyTorch **verifica**: cammina i
parametri e asserisce `cur_offset == offset`.

Rimappare i soli indici quindi non basta, e rompe pure quel controllo. La
soluzione: a PyTorch si presenta la disposizione cuDNN (contigua), e al momento
di chiamare MIOpen il blocco viene ricopiato nell'ordine giusto con una serie di
copie da dispositivo a dispositivo. Per i gradienti sui pesi si fa il percorso
inverso, seminando l'appoggio col contenuto attuale — MIOpen **accumula**, e
sovrascrivere romperebbe l'accumulo.

Altre due cose trovate per strada, entrambe fatali e mute:

- **Dropout a probabilita' zero.** PyTorch crea sempre un descrittore di dropout
  per le RNN e, quando la probabilita' e' zero, lo imposta con `states = nullptr`
  e dimensione 0. cuDNN non fa nulla; MIOpen prova a inizializzare il generatore
  e lancia un kernel su **zero elementi**: «Failed to launch kernel: invalid
  argument». Un descrittore MIOpen appena creato e' gia' "nessun dropout", quindi
  in quel caso non c'e' niente da impostare.
- **Descrittori a 3 dimensioni.** PyTorch descrive i pesi RNN come tensore
  `[n,1,1]` dichiarato NCHW. Il setter di MIOpen basato sulla disposizione
  pretende che NCHW significhi 4 dimensioni e risponde «Unsupported layout». Fuori
  da quattro dimensioni si calcolano i passi a mano e si usa il setter generico.
- **Descrittori nulli.** GRU e RNN semplice non hanno stato di cella: PyTorch
  passa un descrittore NULLO per `cx`/`cy`. cuDNN lo tollera, MIOpen lo
  dereferenzia. Si sostituisce con quello dello stato nascosto, che ha la stessa
  forma; il puntatore ai dati resta nullo, ed e' quello che conta.

Verifica: `prova_rnn_torch.py`, **10 prove su 10** — LSTM a uno e due strati e
bidirezionale, GRU, RNN tanh e relu, avanti e indietro, tutte a ~2e-07.

### 10.2 Spatial transformer: kernel scritti da noi

MIOpen non ha nessuna delle sette funzioni. I quattro kernel (generazione della
griglia affine avanti e indietro, campionamento bilineare avanti e indietro) sono
scritti in HIP dentro `zluda_dnn` e compilati a runtime con hipRTC, caricata
dinamicamente: se il pack non c'e', le funzioni riferiscono NOT_SUPPORTED e il
resto della libreria continua a funzionare.

Convenzioni rispettate: coordinate in [-1,1] nell'ordine (x,y), vertici sui
centri dei pixel estremi, zero fuori dal bordo. Solo fp32: PyTorch qui passa
sempre fp32 e non si scrive codice non verificabile.

Verifica: `prova_spaziale_torch.py`, **4 su 4** — griglia, campionamento, e i
gradienti su ingresso e su theta, tutti sotto 8e-07.

### 10.3 API backend v8: un motore a grafo

E' un'API completamente diversa: si creano descrittori opachi di un TIPO, ci si
mettono ATTRIBUTI, si finalizzano, si esegue un piano. Il grafo che PyTorch
costruisce per una convoluzione contiene **una sola** operazione, e a quel punto
si sa esattamente quale chiamata MIOpen fare. Quello che c'e' e' quindi un
magazzino di attributi piu' un traduttore che ricostruisce la convoluzione e la
manda per la stessa strada della v7.

Tre convenzioni dell'API che non si indovinano, e che sbagliate danno tutte lo
stesso messaggio inutile — «GET was unable to find an engine»:

1. **Gli attributi di tipo "descrittore" si leggono per copia.** Il chiamante ha
   GIA' creato i descrittori di destinazione e passa il loro elenco: cuDNN deve
   copiarci dentro il contenuto, non restituire i propri puntatori.
   `cudnn_frontend` riempie un vettore temporaneo e poi tiene i **suoi**
   descrittori: scrivendoci un puntatore, quelli che tiene restano vuoti.
2. **Un elenco assente e' una lista vuota, non un errore.** `cudnn_frontend`
   interroga sempre note numeriche, note di comportamento e "manopole"; su un
   errore scarta la configurazione. Era l'attributo 302 chiesto come descrittore
   a far fallire tutto.
3. **`Finalize` CATTURA il contenuto dei descrittori riferiti.** Da quel momento
   il chiamante puo' distruggerli, e `cudnn_frontend` conta su questo: passa
   tensori temporanei e li distrugge appena costruita l'operazione. Tenendo solo
   i puntatori si leggeva memoria liberata — prima numeri sbagliati, poi access
   violation. Serve una copia profonda alla finalizzazione, e il piano dev'essere
   **autosufficiente**: al momento in cui viene finalizzato si risolve tutto e se
   ne tiene una copia per valore.

⚠ï¸ Attenzione a un errore di metodo che mi e' costato un giro: avevo messo una
traccia sui fallimenti, poi ho riscritto il corpo della funzione con una patch
che l'ha rimossa. Risultato: zero letture tracciate, e la conclusione sbagliata
che la funzione non venisse chiamata. Quando si strumenta e poi si riscrive,
ricontrollare che la strumentazione ci sia ancora.

Verifica: `prova_conv_torch.py` con `TORCH_CUDNN_V8_API_DISABLED=0`, **13 su 13**,
piu' l'addestramento della CNN corretto.

**La v7 resta il percorso predefinito** perche' e' piu' veloce: 9 ms per passo
contro 26 sulla stessa rete. Il motore a grafo ricostruisce i descrittori MIOpen
a ogni esecuzione, mentre la v7 tiene in cache l'algoritmo. Non e' piu' una
necessita', e' una scelta.

### 10.4 Stato verificato

| banco | esito |
|---|---|
| MIOpen diretto (`prova_miopen_completo.py`) | **19/19** |
| conv da PyTorch, percorso v7 | **13/13** |
| conv da PyTorch, percorso v8 | **13/13** |
| addestramento CNN | **corretto**, pesi a 5.9e-08 |
| RNN / LSTM / GRU | **10/10** |
| spatial transformer | **4/4** |
| `cudnn.benchmark = True` | **corretto**, 4.05e-07 |

### 10.5 Cosa resta davvero

Nulla, fra i simboli che PyTorch importa. Restano dei **limiti dichiarati**, che
si comportano correttamente (riferiscono NOT_SUPPORTED invece di dare numeri
sbagliati):

- attivazione **SWISH**: MIOpen non ce l'ha, PyTorch usa il proprio kernel
- **proiezione ricorrente** (LSTMP, `proj_size > 0`): MIOpen non ce l'ha
- batch norm con **attivazione fusa** (`bnOps` diverso da solo-normalizzazione)
- spatial transformer in **fp16**
- grafi v8 con **piu' di un'operazione** (le fusioni), pointwise, matmul

---

## 11. Convinzioni sbagliate, corrette

La stessa trappola può ripresentarsi: per questo restano scritte.

| cosa si credeva | com'è |
|---|---|
| hipRTC 7.x è rotta su gfx906 (rimozione da PAL) | falso: era il percorso di caricamento della DLL |
| il driver Adrenalin non carica i code object v5 | falso: la v5 si carica, il crash veniva da `sramecc` |
| `zluda_dnn` non si può riattivare come cdylib | falso: bastava costruire `zluda_dnn8` |

Le correzioni sono state portate anche nei commenti del codice (`zluda_nvrtc/src/lib.rs`,
`zluda_dnn/Cargo.toml`).
