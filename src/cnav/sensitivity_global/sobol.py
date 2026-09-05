"""
Indici di Sobol sul sottoinsieme di fattori scelto dopo Morris.

Cosa aggiunge rispetto a Morris
-------------------------------
Morris ordina, Sobol quantifica. Gli indici sono frazioni di varianza
di Y, quindi adimensionali e sommabili:

  S_i     indice del primo ordine. La frazione di varianza di Y che
          sparirebbe se il fattore i fosse fissato al suo valore vero.
          Sarebbe l'effetto del fattore "da solo"
  S_Ti    indice totale. La frazione di varianza che resterebbe se
          tutti gli altri fattori fossero fissati: contiene l'effetto
          diretto più tutte le interazioni in cui è coinvolto.
          Vale sempre S_Ti >= S_i
  S_Ti - S_i   la parte di influenza che passa dalle interazioni. Sui
          fattori con sigma alto nella nube di Morris ci si aspetta di
          trovarla diversa da zero: è la verifica quantitativa di
          quello che lo screening aveva solo segnalato

Somma di S_i vicino a 1 = modello sostanzialmente additivo. Somma
molto minore di 1 = la varianza vive nelle interazioni. Il modello qui
è fortemente non additivo (il sizing è un punto fisso, ogni
parametro entra nel MTOW che rientra in tutto il resto), quindi
aspettarsi la somma sotto 1 è normale e va scritto in tesi come
risultato, non come errore numerico.

IL GRUPPO "resto": perchè i fattori scartati restano campionati
------------------------------------------------------------------
Fare la Sobol solo sui fattori selezionati, con gli altri congelati al
nominale, risponderebbe a una domanda diversa: come si ripartisce la
varianza di un modello a cui ho tolto dei pezzi. La varianza totale
sarebbe più piccola di quella vera e gli indici sarebbero tutti
gonfiati.

Qui invece i fattori scartati continuano a essere campionati dalle
loro PDF, ma vengono trattati come un solo fattore di gruppo, chiamato
"resto": nella matrice AB le loro colonne si muovono tutte insieme.
Il costo passa da N*(k+2) con k = 15 a N*(g+2) con g = selezionati + 1,
la varianza totale al denominatore resta quella vera, e in più si
ottiene gratis S_T[resto], che è esattamente la verifica che la
guideline chiede: se lo screening ha buttato via solo roba
irrilevante, S_T[resto] deve venire piccolo. Se viene grosso, Morris
ha sbagliato a scartare e il sottoinsieme va allargato.

Stesso meccanismo, per la stessa ragione, di "calibration_block" in
factors.py: un gruppo è un fattore a tutti gli effetti, purchè sia
indipendente dagli altri.

I NaN
-----
Come in Morris, una configurazione infattibile da NaN. Qui però il
danno è peggiore: gli stimatori sono medie su coppie (f_A, f_AB), e
una coppia si perde tutta se manca un solo termine. I NaN vengono
esclusi coppia per coppia e il conteggio finisce nella colonna
n_valid, da leggere sempre insieme agli indici.

Attenzione all'interpretazione: i NaN non mancano a caso, mancano dove
i parametri sono sfavorevoli. Sotto il 5-10% di scarti la cosa è
trascurabile, ma sopra l'indice sta descrivendo la varianza
condizionata alla fattibilità, e in tesi va detto. La cura è spostare
il punto operativo dentro il dominio del sistema in
REPRESENTATIVE_MISSIONS, ma solo dopo aver letto l'avvertimento sulla
censura qui sotto: per i sistemi che vivono al bordo del proprio
dominio spostare il punto non risolve niente, sposta solo il problema
da una colonna all'altra.

Gli intervalli di confidenza
----------------------------
Bootstrap sulle righe del campione (n_boot ricampionamenti con
reimmissione degli indici 0..N-1, gli stessi per tutti i gruppi di uno
stesso output così i confronti fra fattori restano coerenti). Danno
la banda dovuta al campionamento finito, non l'errore del modello.
Servono a due cose concrete: capire se due fattori sono davvero
ordinati o solo rumore, e riconoscere gli S_i leggermente negativi,
che sono rumore attorno a zero e non un risultato (vengono lasciati
negativi apposta, non troncati: è il segnale che N è piccolo per
quel fattore).

LE CODE PESANTI, e perchè qui contano più del solito
-------------------------------------------------------
Gli indici di Sobol sono rapporti di varianze, e la varianza è una
statistica fragile: basta una manciata di campioni con Y enorme perchè
il denominatore sia deciso da loro e gli indici diventino instabili
(succede di vedere S_T > 1, che non ha senso e infatti è solo rumore).

Su questo modello il problema c'è. Vicino al confine di
fattibilità l'intensity non va a NaN di colpo: prima esplode, perche'
il punto fisso del sizing converge a un MTOW enorme. Sui punti
operativi al bordo del dominio (la batteria, il fuel cell con fan) il
rapporto fra massimo e mediana del campione arriva a uno o due ordini
di grandezza. Morris non ne soffriva, perchè mu* è una media di
valori assoluti; Sobol si.

tail_report() misura la cosa prima di stimare qualsiasi indice. Se un
output ha la coda lunga ci sono tre strade, in ordine di preferenza:

  1. spostare il punto operativo più dentro il dominio del sistema
     (in REPRESENTATIVE_MISSIONS). Questa è la strada giusta SOLO
     quando il punto era stato scelto ottimisticamente, cioè quando
     lì si sta misurando la fattibilità e non la sensibilità. NON lo
     è per i sistemi che vivono per costruzione al bordo del proprio
     dominio, come la batteria: lì allontanare il punto non accorcia
     la coda, la trasforma in NaN. I campioni peggiori smettono di
     convergere e spariscono dal calcolo, il numero migliora e il
     campione peggiora, e gli indici finiscono per descrivere la sola
     sottopopolazione fattibile. È un bias silenzioso, molto peggiore
     di una varianza rumorosa. Vedi l'avvertimento sulla censura nel
     docstring di tail_report
  2. transform="log" in sobol_indices: si decompone la varianza di
     log(Y) invece che di Y. La domanda cambia, e va scritto in tesi:
     non più "chi governa gli MJ/pax/nmi" ma "chi governa le
     variazioni RELATIVE dell'intensity". Su grandezze che spaziano
     ordini di grandezza è anche la domanda più sensata, ed è
     coerente con il confronto fra tecnologie che sta a monte della
     tesi, che è un confronto di rapporti
  3. alzare N. Aiuta, ma lentamente: con code pesanti la varianza
     campionaria converge male per costruzione

Non è consigliabile invece troncare o winsorizzare Y: quei campioni
non sono errori numerici, sono configurazioni fisicamente peggiori, e
buttarli via cambia la domanda senza dirlo.

Quanto deve valere N
--------------------
Regola pratica: N = 512 per una prova, 1024-4096 per il risultato da
tesi, e comunque una potenza di 2 (la sequenza di Sobol è bilanciata
sui blocchi di lunghezza 2^m; troncarla altrove peggiora la
convergenza). La verifica non è teorica ma empirica, è quella di
sobol_convergence: si ricalcolano gli indici su N crescenti e si
guarda se il ranking si è stabilizzato.
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

__all__ = [
    "RESIDUAL_GROUP",
    "SobolDesign",
    "sobol_design",
    "sobol_indices",
    "sobol_summary",
    "tail_report",
    "screening_check",
    "sobol_table",
    "sobol_convergence",
    "compare_with_morris",
    "plot_sobol_bars",
    "plot_all_sobol_bars",
    "plot_sobol_aggregate",
    "plot_sobol_convergence",
]

# nome del gruppo che raccoglie tutti i fattori non selezionati
RESIDUAL_GROUP = "resto"


# ---------------------------------------------------------------------
# Disegno
# ---------------------------------------------------------------------

@dataclass
class SobolDesign:
    """Il disegno A / B / AB di Saltelli, con i fattori raggruppati.

    U           (n_blocks*N, k) le matrici impilate nell'ordine
                A, B, AB_0, AB_1, ..., AB_{g-1}
    N           dimensione del campione base
    group_names i nomi dei gruppi, nell'ordine dei blocchi AB.
                L'ultimo è RESIDUAL_GROUP se qualche fattore è stato
                scartato
    group_columns  per ogni gruppo, gli indici delle colonne di U che
                gli appartengono
    factor_names   i nomi di tutti i k fattori dello spazio (serve solo
                a documentare il disegno)

    Uso tipico:

        design = sobol_design(space, selected=[...], N=1024)
        theta = space.theta_from_unit(design.U)
        Y = evaluate_outputs(theta, outputs)
        df = sobol_indices(design, Y)
    """
    U: np.ndarray
    N: int
    group_names: list
    group_columns: list
    factor_names: list = field(default_factory=list)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def n_blocks(self) -> int:
        return self.n_groups + 2          # A, B e un AB per gruppo

    @property
    def n_evaluations(self) -> int:
        return self.n_blocks * self.N

    def split(self, y, n: Optional[int] = None) -> tuple:
        """Da un vettore di risposte (una colonna di Y) ai blocchi.

        Ritorna (fA, fB, [fAB_0, ...]), ciascuno di lunghezza n.

        n < N tiene solo le prime n righe di ogni blocco: è così che
        sobol_convergence rifa' il conto su campioni più piccoli senza
        rivalutare il modello. Funziona perchè la sequenza di Sobol è
        estendibile, cioè le prime n righe sono a loro volta un
        campione bilanciato (se n è potenza di 2)
        """
        y = np.asarray(y, dtype=float).reshape(self.n_blocks, self.N)
        n = self.N if n is None else int(n)
        if not 1 <= n <= self.N:
            raise ValueError(f"n deve stare fra 1 e N={self.N}")
        return y[0, :n], y[1, :n], [y[2 + g, :n] for g in range(self.n_groups)]

    def cost_table(self) -> pd.DataFrame:
        """Il costo del disegno, da stampare prima di lanciare"""
        return pd.DataFrame([{
            "N": self.N,
            "gruppi": self.n_groups,
            "blocchi": self.n_blocks,
            "valutazioni": self.n_evaluations,
            "fattori_totali": len(self.factor_names),
            "gruppi_elencati": ", ".join(self.group_names),
        }])


def sobol_design(space, selected: list, N: int = 512, seed: int = 1,
                 residual_group: bool = True) -> SobolDesign:
    """Costruisce il disegno di Saltelli sui fattori selezionati.

    space     un FactorSpace (serve solo per space.names e space.k)
    selected  i nomi dei fattori da analizzare singolarmente, cioè
              quelli scelti a mano leggendo le nubi di Morris
    N         dimensione del campione base, potenza di 2
    residual_group  True: i fattori non selezionati restano campionati
              e si muovono insieme come gruppo RESIDUAL_GROUP.
              False: restano campionati ma non hanno un blocco AB, non
              si ottiene il loro indice e si perde la verifica dello
              screening. Da usare solo se selected è gia' l'insieme
              completo dei fattori

    Le matrici A e B vengono dalle prime k e dalle seconde k colonne di
    una sequenza di Sobol in 2k dimensioni: è il modo standard di
    ottenere due campioni indipendenti mantenendo la bassa discrepanza,
    generarne due con seed diversi darebbe stime più rumorose
    """
    names = list(space.names)
    k = len(names)

    if not selected:
        raise ValueError("selected è vuoto: la Sobol si fa su un sottoinsieme "
                         "scelto a mano dopo aver letto le nubi di Morris")
    doppioni = [nm for nm in set(selected) if selected.count(nm) > 1]
    if doppioni:
        raise ValueError(f"Fattori ripetuti in selected: {sorted(doppioni)}")
    ignoti = [nm for nm in selected if nm not in names]
    if ignoti:
        raise ValueError(f"Fattori sconosciuti: {ignoti}. Disponibili: {names}")

    if N & (N - 1):
        warnings.warn(f"N = {N} non è una potenza di 2: la sequenza di Sobol "
                      "perde il bilanciamento e gli indici convergono peggio",
                      RuntimeWarning)

    # i gruppi: uno per fattore selezionato, piu' il resto
    group_names = list(selected)
    group_columns = [[names.index(nm)] for nm in selected]

    rest = [j for j, nm in enumerate(names) if nm not in selected]
    if rest and residual_group:
        group_names.append(RESIDUAL_GROUP)
        group_columns.append(rest)
    elif rest and not residual_group:
        warnings.warn(
            f"{len(rest)} fattori non selezionati restano campionati ma senza "
            "blocco AB: la varianza totale è quella giusta, ma non si potrà "
            "verificare quanto pesa ciò che lo screening ha scartato",
            RuntimeWarning)

    sampler = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    with warnings.catch_warnings():
        # scipy avvisa se N non e' potenza di 2: l'avviso l'abbiamo gia' dato
        warnings.simplefilter("ignore", UserWarning)
        base = sampler.random(n=N)
    A, B = base[:, :k], base[:, k:]

    blocks = [A, B]
    for cols in group_columns:
        AB = A.copy()
        AB[:, cols] = B[:, cols]
        blocks.append(AB)

    return SobolDesign(U=np.vstack(blocks), N=N, group_names=group_names,
                       group_columns=group_columns, factor_names=names)


# ---------------------------------------------------------------------
# Stimatori
# ---------------------------------------------------------------------

def _estimate(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray) -> tuple:
    """(S_i, S_Ti, n_valid) per un gruppo, ignorando le coppie con NaN.

    S_i    stimatore di Saltelli 2010:  mean(f_B * (f_AB - f_A)) / V
    S_Ti   stimatore di Jansen:         mean((f_A - f_AB)^2) / (2V)

    Le risposte vengono centrate sulla media del campione A+B prima di
    stimare S_i: il numeratore è una covarianza, e con Y di media
    grande rispetto alla dispersione (le intensity stanno a decine di
    MJ/pax/nmi con spread di pochi) senza centrare si sottraggono
    numeri quasi uguali e si perdono cifre significative
    """
    ok = np.isfinite(fA) & np.isfinite(fB) & np.isfinite(fAB)
    n = int(ok.sum())
    if n < 8:
        return np.nan, np.nan, n

    a, b, ab = fA[ok], fB[ok], fAB[ok]
    f0 = 0.5 * (a.mean() + b.mean())
    a, b, ab = a - f0, b - f0, ab - f0

    V = np.var(np.concatenate([a, b]), ddof=1)
    if not np.isfinite(V) or V <= 0:
        return np.nan, np.nan, n

    S = float(np.mean(b * (ab - a)) / V)
    ST = float(np.mean((a - ab) ** 2) / (2.0 * V))
    return S, ST, n


def _bootstrap(fA, fB, fAB, n_boot: int, rng, idx=None) -> tuple:
    """Percentili 2.5 / 97.5 di S_i e S_Ti per ricampionamento delle
    righe. idx (n_boot, n) permette di riusare gli stessi
    ricampionamenti su tutti i gruppi di uno stesso output"""
    if n_boot <= 0:
        return (np.nan,) * 4

    n = len(fA)
    if idx is None:
        idx = rng.integers(0, n, size=(n_boot, n))

    S = np.full(n_boot, np.nan)
    ST = np.full(n_boot, np.nan)
    for r in range(n_boot):
        S[r], ST[r], _ = _estimate(fA[idx[r]], fB[idx[r]], fAB[idx[r]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qS = np.nanpercentile(S, [2.5, 97.5])
        qST = np.nanpercentile(ST, [2.5, 97.5])
    return float(qS[0]), float(qS[1]), float(qST[0]), float(qST[1])


def sobol_indices(design: SobolDesign, Y: pd.DataFrame, n_boot: int = 200,
                  seed: int = 0, n: Optional[int] = None,
                  transform: Optional[str] = None) -> pd.DataFrame:
    """Tabella tidy (output, fattore, S1, ST, interazione, ...).

    Y è la matrice delle risposte, con tante righe quante le righe di
    design.U e una colonna per punto operativo: esattamente quello che
    restituisce evaluate_outputs.

    Colonne:
      S1, ST            gli indici
      S1_lo/hi, ST_lo/hi   estremi bootstrap al 95%
      interazione       ST - S1, la quota che passa dalle interazioni
      n_valid           coppie usate su N (le altre avevano un NaN)
      frazione_valida   n_valid / N, da guardare prima degli indici

    n_boot=0 salta il bootstrap (molto più veloce, nessun intervallo)

    transform="log" decompone la varianza di log(Y) invece che di Y:
    da usare sugli output a coda lunga segnalati da tail_report. La
    colonna 'trasformazione' tiene traccia della scelta, così non si
    finisce per confrontare in tesi indici calcolati su scale diverse
    """
    if len(Y) != design.n_evaluations:
        raise ValueError(
            f"Y ha {len(Y)} righe, il disegno ne prevede {design.n_evaluations} "
            f"({design.n_blocks} blocchi x N={design.N}). Y va calcolata su "
            "design.U, senza riordinare o filtrare le righe")

    rng = np.random.default_rng(seed)
    n_eff = design.N if n is None else int(n)
    rows = []

    for col in Y.columns:
        fA, fB, fABs = design.split(
            _apply_transform(Y[col].to_numpy(dtype=float), transform), n=n_eff)
        # stessi ricampionamenti per tutti i gruppi di questo output
        idx = rng.integers(0, n_eff, size=(n_boot, n_eff)) if n_boot > 0 else None

        for g, name in enumerate(design.group_names):
            S, ST, n_valid = _estimate(fA, fB, fABs[g])
            S_lo, S_hi, ST_lo, ST_hi = _bootstrap(fA, fB, fABs[g], n_boot, rng, idx)
            rows.append({
                "output": col, "fattore": name,
                "S1": S, "S1_lo": S_lo, "S1_hi": S_hi,
                "ST": ST, "ST_lo": ST_lo, "ST_hi": ST_hi,
                "interazione": ST - S if np.isfinite(ST) and np.isfinite(S) else np.nan,
                "n_valid": n_valid, "frazione_valida": n_valid / n_eff,
                "N": n_eff, "trasformazione": transform or "nessuna",
            })

    return pd.DataFrame(rows)


def _apply_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """Trasforma la risposta prima di decomporne la varianza.

    None    nessuna trasformazione: si decompone la varianza di Y, e
            gli indici rispondono a "chi governa gli MJ/pax/nmi"
    "log"   si decompone la varianza di log(Y): gli indici rispondono
            a "chi governa le variazioni relative dell'intensity".
            Da usare sugli output a coda lunga (vedi tail_report), e
            da dichiarare in tesi, perchè la domanda cambia

    I valori non positivi diventano NaN: l'intensity è positiva per
    costruzione, se ne compare uno c'è un problema a monte e va visto,
    non trasformato
    """
    if transform is None:
        return y
    if transform != "log":
        raise ValueError("transform può essere None oppure 'log'")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y > 0, np.log(y), np.nan)


def tail_report(design: SobolDesign, Y: pd.DataFrame,
                soglia: float = 5.0) -> pd.DataFrame:
    """Quanto è lunga la coda di ciascun output. Da guardare prima di
    sobol_indices, insieme a nan_report.

    Il campione usato è quello dei blocchi A e B, cioè il campione
    onesto dalle PDF (i blocchi AB sono ibridi e non rappresentano una
    distribuzione vera).

    Due misure di coda, e il verdetto si basa sulla seconda:

      coda_max   massimo / mediana. Utile a occhio, ma la decide un
                 solo campione: un output con corpo perfettamente sano
                 può avere coda_max = 20 per via di un unico estratto
                 sfortunato
      coda_p99   99-esimo percentile / mediana. È il corpo della
                 distribuzione e non il suo caso peggiore, ed è quello
                 che decide davvero la varianza campionaria

    ATTENZIONE ALL'EFFETTO DI CENSURA. Le due colonne vanno lette
    insieme a frazione_nan, altrimenti si prende la decisione al
    contrario. Man mano che il punto operativo si allontana dal
    dominio di fattibilità, i campioni peggiori smettono di convergere
    e diventano NaN: spariscono dal calcolo, e la coda MIGLIORA
    numericamente proprio mentre il problema peggiora. Un output con
    coda corta e 15% di NaN non è ben condizionato, è troncato, e i
    suoi indici descrivono la sola sottopopolazione fattibile.

    Quindi: coda corta + pochi NaN = ok. Coda corta + molti NaN =
    censura, non salute. Coda lunga + pochi NaN = onesto ma a varianza
    fragile, ed è lì che serve transform="log"
    """
    rows = []
    for col in Y.columns:
        fA, fB, _ = design.split(Y[col].to_numpy())
        v = np.concatenate([fA, fB])
        v = v[np.isfinite(v)]
        if len(v) == 0:
            rows.append({"output": col, "mediana": np.nan, "p99": np.nan,
                         "massimo": np.nan, "coda_p99": np.nan, "coda_max": np.nan,
                         "cv": np.nan, "frazione_nan": 1.0,
                         "verdetto": "nessun campione valido"})
            continue

        med = float(np.median(v))
        p99 = float(np.percentile(v, 99))
        coda_p99 = p99 / med if med > 0 else np.inf
        coda_max = float(np.max(v)) / med if med > 0 else np.inf
        frazione_nan = 1.0 - len(v) / (2 * design.N)

        if frazione_nan > 0.10:
            verdetto = "censurato dai NaN"
        elif coda_p99 > soglia:
            verdetto = "coda lunga"
        else:
            verdetto = "ok"

        rows.append({
            "output": col,
            "mediana": med,
            "p99": p99,
            "massimo": float(np.max(v)),
            "coda_p99": coda_p99,
            "coda_max": coda_max,
            "cv": float(np.std(v, ddof=1) / np.mean(v)) if np.mean(v) else np.nan,
            "frazione_nan": frazione_nan,
            "verdetto": verdetto,
        })
    return pd.DataFrame(rows).sort_values("coda_p99", ascending=False).reset_index(drop=True)


def sobol_summary(df_sobol: pd.DataFrame) -> pd.DataFrame:
    """Una riga per output: le diagnostiche da guardare prima di
    interpretare i singoli indici.

      somma_S1       vicino a 1 = modello additivo in quei fattori.
                     Molto sotto 1 = la varianza sta nelle interazioni
      somma_ST       sempre >= somma_S1; quanto la supera misura il
                     peso complessivo delle interazioni (ogni
                     interazione è contata una volta per fattore
                     coinvolto, per questo può superare 1)
      ST_resto       l'indice totale del gruppo scartato: la verifica
                     dello screening
      S1_min         il più negativo fra gli S1: se è molto sotto
                     zero (diciamo -0.05) il campione è piccolo
      frazione_valida_min  il caso peggiore di NaN su quell'output
    """
    out = []
    for col, sub in df_sobol.groupby("output", sort=False):
        resto = sub.loc[sub["fattore"] == RESIDUAL_GROUP, "ST"]
        out.append({
            "output": col,
            "somma_S1": sub["S1"].sum(min_count=1),
            "somma_ST": sub["ST"].sum(min_count=1),
            "ST_resto": float(resto.iloc[0]) if len(resto) else np.nan,
            "S1_min": sub["S1"].min(),
            "frazione_valida_min": sub["frazione_valida"].min(),
        })
    return pd.DataFrame(out)


def screening_check(df_sobol: pd.DataFrame, tolleranza: float = 0.05) -> pd.DataFrame:
    """Lo screening di Morris ha buttato via qualcosa di importante?

    Guarda ST del gruppo "resto" output per output e lo confronta con
    una tolleranza (default 5% della varianza). L'esito è:

      ok         ST_resto <= tolleranza: i fattori scartati contano
                 meno della tolleranza, il sottoinsieme regge
      da rivedere  sopra la tolleranza: qualcosa di rilevante è
                 rimasto fuori. Si torna alle nubi di Morris e si
                 allarga il sottoinsieme, tipicamente col fattore che
                 nella nube di quel punto operativo stava appena sotto
                 la soglia

    È il controllo che rende difendibile la selezione manuale: senza,
    "ho scelto questi guardando il grafico" resta un'affermazione senza
    verifica
    """
    resto = df_sobol[df_sobol["fattore"] == RESIDUAL_GROUP]
    if resto.empty:
        raise ValueError(
            f"Nessun gruppo {RESIDUAL_GROUP!r} nei risultati: il disegno è stato "
            "costruito con residual_group=False, oppure erano stati selezionati "
            "tutti i fattori. Senza gruppo residuo la verifica non è possibile")

    tab = resto[["output", "ST", "ST_lo", "ST_hi", "frazione_valida"]].copy()
    tab = tab.rename(columns={"ST": "ST_resto", "ST_lo": "ST_resto_lo",
                              "ST_hi": "ST_resto_hi"})
    tab["tolleranza"] = tolleranza
    tab["esito"] = np.where(tab["ST_resto"] <= tolleranza, "ok", "da rivedere")
    return tab.sort_values("ST_resto", ascending=False).reset_index(drop=True)


def sobol_table(df_sobol: pd.DataFrame, index: str = "ST") -> pd.DataFrame:
    """Da tabella tidy a matrice fattori x output, ordinata per media.

    index: "S1", "ST" o "interazione".

    A differenza di ranking_table (Morris) qui non si normalizza: gli
    indici sono già frazioni di varianza, quindi confrontabili fra
    output diversi così come sono; è il vantaggio pratico della
    Sobol, e vale la pena dirlo in tesi accanto alle due tabelle
    """
    if index not in ("S1", "ST", "interazione"):
        raise ValueError("index deve essere 'S1', 'ST' o 'interazione'")
    wide = df_sobol.pivot(index="fattore", columns="output", values=index)
    wide[f"{index}_medio"] = wide.mean(axis=1)
    return wide.sort_values(f"{index}_medio", ascending=False)


def sobol_convergence(design: SobolDesign, Y: pd.DataFrame,
                      sizes: Optional[list] = None, seed: int = 0,
                      transform: Optional[str] = None) -> pd.DataFrame:
    """Gli indici ricalcolati su campioni via via più grandi.

    Non costa nessuna valutazione in più: riusa le prime n righe di
    ogni blocco di Y. Il grafico che ne esce (plot_sobol_convergence)
    è la prova di convergenza da mettere in tesi: se le curve di S_Ti
    sono ancora incrociate all'ultimo N, N va aumentato.

    sizes: le dimensioni da provare. Di default le potenze di 2 fino a
    N, il che tiene il campione bilanciato a ogni passo
    """
    if sizes is None:
        sizes, m = [], 64
        while m <= design.N:
            sizes.append(m)
            m *= 2
        if not sizes or sizes[-1] != design.N:
            sizes.append(design.N)

    parts = []
    for n in sizes:
        df = sobol_indices(design, Y, n_boot=0, seed=seed, n=int(n),
                           transform=transform)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def compare_with_morris(df_sobol: pd.DataFrame, df_morris: pd.DataFrame) -> pd.DataFrame:
    """Affianca il ranking di Morris (mu* normalizzato) a quello di
    Sobol (S_T), fattore per fattore, mediati sugli output.

    Serve a due frasi che in tesi vanno scritte:
      - lo screening ordinava bene? (i ranghi si corrispondono)
      - quanto costava in più Sobol per sapere la stessa cosa?

    Il gruppo "resto" non ha un corrispettivo in Morris e resta con
    rango_morris NaN. I fattori scartati compaiono solo dal lato
    Morris, e la colonna rango_sobol vuota è esattamente il punto:
    sono quelli che non è stato necessario quantificare
    """
    from .morris import ranking_table

    mor = ranking_table(df_morris)["mu_star_medio"].rename("mu_star_medio")
    sob = df_sobol.groupby("fattore")["ST"].mean().rename("ST_medio")

    tab = pd.concat([mor, sob], axis=1)
    tab["rango_morris"] = tab["mu_star_medio"].rank(ascending=False)
    tab["rango_sobol"] = tab["ST_medio"].rank(ascending=False)
    tab["delta_rango"] = tab["rango_morris"] - tab["rango_sobol"]
    return tab.sort_values("ST_medio", ascending=False)


# ---------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------

def _bars(ax, sub: pd.DataFrame, title: str, show_ci: bool = True,
          fontsize: float = 8.0):
    """Barre orizzontali S_i e S_Ti appaiate, ordinate per S_T.

    La lettura: la barra chiara (S_T) che sporge oltre quella scura
    (S_1) è la quota di interazione. Due barre uguali = fattore
    additivo
    """
    sub = sub.sort_values("ST", ascending=True)
    y = np.arange(len(sub))
    h = 0.38

    err1 = None
    errT = None
    if show_ci and sub[["S1_lo", "ST_lo"]].notna().all().all():
        err1 = np.abs(np.vstack([sub["S1"] - sub["S1_lo"], sub["S1_hi"] - sub["S1"]]))
        errT = np.abs(np.vstack([sub["ST"] - sub["ST_lo"], sub["ST_hi"] - sub["ST"]]))

    colori = ["tab:red" if nm == RESIDUAL_GROUP else "tab:blue" for nm in sub["fattore"]]
    ax.barh(y + h / 2, sub["ST"], height=h, color=colori, alpha=0.35,
            xerr=errT, error_kw=dict(lw=0.8, ecolor="0.4"), label=r"$S_{T_i}$")
    ax.barh(y - h / 2, sub["S1"], height=h, color=colori, alpha=0.95,
            xerr=err1, error_kw=dict(lw=0.8, ecolor="0.4"), label=r"$S_i$")

    ax.axvline(0.0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["fattore"], fontsize=fontsize)
    ax.set_xlabel("frazione di varianza")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_sobol_bars(df_sobol: pd.DataFrame, output: str, ax=None,
                    show_ci: bool = True, fontsize: float = 8.0):
    """Gli indici di un punto operativo.

    output è il nome della colonna di Y, per esempio
    "E[Hydrogen combustion]@1500nmi_450kt_fan"
    """
    sub = df_sobol[df_sobol["output"] == output]
    if sub.empty:
        raise ValueError(f"Nessun output {output!r}. Disponibili: "
                         f"{sorted(df_sobol['output'].unique())}")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.0, 0.45 * len(sub) + 2.0))

    _bars(ax, sub, output, show_ci=show_ci, fontsize=fontsize)
    if created:
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
    return ax


def plot_all_sobol_bars(df_sobol: pd.DataFrame, ncols: int = 3,
                        outputs: Optional[list] = None, show_ci: bool = True):
    """Una griglia con tutti i punti operativi, per vedere in un colpo
    solo se la ripartizione della varianza cambia da un punto
    all'altro (che è poi la domanda della tesi: i domini di eccellenza
    dipendono da parametri diversi in zone diverse del piano)"""
    names = outputs or list(pd.unique(df_sobol["output"]))
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(names) / ncols))
    n_fatt = df_sobol["fattore"].nunique()

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0 * ncols, (0.32 * n_fatt + 1.6) * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, nm in zip(axes, names):
        plot_sobol_bars(df_sobol, nm, ax=ax, show_ci=show_ci, fontsize=6.5)
    for ax in axes[len(names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Indici di Sobol per punto operativo "
                 r"(barra piena $S_i$, barra chiara $S_{T_i}$; "
                 "in rosso il gruppo dei fattori scartati)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_sobol_aggregate(df_sobol: pd.DataFrame, ax=None):
    """La media degli indici su tutti i punti operativi.

    Qui, a differenza della nube cumulativa di Morris, non serve
    normalizzare niente: gli indici sono già frazioni di varianza. Ma
    resta valido lo stesso avvertimento: la media pesa i sistemi in
    proporzione a quante missioni gli sono state assegnate, quindi va
    letta insieme ai singoli riquadri e non al posto loro
    """
    agg = (df_sobol.groupby("fattore")
           .agg(S1=("S1", "mean"), ST=("ST", "mean"),
                S1_lo=("S1", "min"), S1_hi=("S1", "max"),
                ST_lo=("ST", "min"), ST_hi=("ST", "max"))
           .reset_index())

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.5, 0.45 * len(agg) + 2.0))

    # qui le "barre di errore" non sono un bootstrap ma l'escursione
    # fra punti operativi: e' l'informazione piu' utile da vedere
    _bars(ax, agg, "Media sui punti operativi "
                   "(le barrette sono min-max fra punti, non intervalli di confidenza)")
    if created:
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
    return ax


def plot_sobol_convergence(df_conv: pd.DataFrame, output: Optional[str] = None,
                           index: str = "ST", ax=None):
    """S_T (o S_1) in funzione di N, una curva per fattore.

    output=None media su tutti i punti operativi. Le curve devono
    appiattirsi e smettere di incrociarsi: se all'ultimo N sono ancora
    in movimento, il ranking non è stabile e N va aumentato
    """
    if index not in ("S1", "ST"):
        raise ValueError("index deve essere 'S1' o 'ST'")

    df = df_conv if output is None else df_conv[df_conv["output"] == output]
    if df.empty:
        raise ValueError(f"Nessun output {output!r} in df_conv")

    piv = df.pivot_table(index="N", columns="fattore", values=index, aggfunc="mean")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.5, 5.0))

    for nm in piv.columns:
        stile = dict(color="tab:red", lw=2.0) if nm == RESIDUAL_GROUP else {}
        ax.plot(piv.index, piv[nm], marker="o", ms=3.5, label=nm, **stile)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (campione base)")
    ax.set_ylabel(r"$S_{T_i}$" if index == "ST" else r"$S_i$")
    ax.set_title("Convergenza degli indici" +
                 (f": {output}" if output else " (media sui punti operativi)"),
                 fontsize=9)
    ax.grid(alpha=0.25)
    if created:
        ax.legend(fontsize=7.5, ncol=2, loc="best")
        plt.tight_layout()
    return ax