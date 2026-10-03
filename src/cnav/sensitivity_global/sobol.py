"""
Indici di Sobol sui parametri tecnologici scelti a mano.

Cosa si calcola
---------------
Per ogni output (l'intensity di un'architettura in un punto della
griglia range-velocità) si stimano:

  S_i     indice del primo ordine: la frazione di varianza che
          sparirebbe se il fattore i fosse noto esattamente
  S_Ti    indice totale: la frazione di varianza in cui il fattore i è
          coinvolto, da solo o in interazione. S_Ti >= S_i

I grafici usano S_T (vedi plots.py per il perché); S_1 resta comunque
nella tabella tidy, così la quota di interazione S_T - S_1 è sempre
recuperabile.

IL GRUPPO "others"
-----------------
Non c'è più lo screening: i fattori analizzati singolarmente sono
quelli scelti dall'analista (i parametri tecnologici). Tutti gli altri
NON vengono congelati al nominale: restano campionati dalle loro PDF e
si muovono insieme come un unico fattore di gruppo, "others".

Congelarli risponderebbe a una domanda diversa ("come si ripartisce la
varianza di un modello a cui ho tolto dei pezzi"): la varianza totale
sarebbe più piccola di quella vera e tutti gli indici risulterebbero
gonfiati. Così invece il denominatore è la varianza vera, e S_T[others]
dice quanto pesa tutto ciò che tecnologico non è (regressioni OEW,
calibrazione, ...). Se in una zona del piano S_T[others] domina, il
risultato onesto è proprio quello: lì l'incertezza non è tecnologica.

I NaN
-----
Una configurazione infattibile dà NaN. Gli stimatori lavorano su
coppie (f_A, f_AB) e una coppia si perde se manca un termine; i NaN
vengono esclusi coppia per coppia e contati in n_valid. Sopra il
~10% di NaN l'indice descrive la varianza CONDIZIONATA alla
fattibilità: quei punti vengono marcati come non fattibili e restano
fuori da aggregati e mappe (soglia nan_tol).

La trasformazione logaritmica
-----------------------------
Vicino al confine di fattibilità l'intensity esplode prima di andare a
NaN (il punto fisso del sizing converge a MTOW enormi): code pesanti,
varianza decisa da pochi campioni, S_T > 1. Con transform="log" si
decompone la varianza di log(Y): la domanda diventa "chi governa le
variazioni RELATIVE dell'intensity". Ha un secondo vantaggio decisivo
per l'aggregazione: Var(log Y) è adimensionale (≈ CV^2), quindi le
varianze di sistemi con intensity di ordini di grandezza diversi sono
direttamente confrontabili e si possono usare come pesi (vedi
aggregation.py).

Bootstrap
---------
Gli intervalli si ottengono ricampionando le righe del campione base
con GLI STESSI indici per tutti gli output e tutti i gruppi. Questo
non serve solo alla coerenza dei confronti: è ciò che permette di
propagare l'incertezza di campionamento anche agli indici AGGREGATI
(ogni replica bootstrap dà un aggregato) e quindi di calcolare la
probabilità che un fattore sia davvero il più influente in una cella
delle mappe. Le repliche si conservano per questo in SobolField.

N: potenza di 2 (la sequenza di Sobol è bilanciata sui blocchi 2^m).
512 basta per le mappe, 1024-2048 per il grafico aggregato da tesi; la
verifica è empirica (sobol_convergence in aggregation.py).
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc

__all__ = [
    "RESIDUAL_GROUP",
    "SobolDesign",
    "SobolField",
    "sobol_design",
    "sobol_field",
    "tail_report",
    "residual_report",
    "sobol_table",
]

# nome del gruppo che raccoglie tutti i fattori non selezionati
RESIDUAL_GROUP = "others"


# ---------------------------------------------------------------------
# Disegno
# ---------------------------------------------------------------------

@dataclass
class SobolDesign:
    """Il disegno A / B / AB di Saltelli, con i fattori raggruppati.

    U              (n_blocks*N, k) le matrici impilate nell'ordine
                   A, B, AB_0, AB_1, ..., AB_{g-1}
    N              dimensione del campione base
    group_names    nomi dei gruppi, nell'ordine dei blocchi AB. L'ultimo
                   è RESIDUAL_GROUP se qualche fattore non è selezionato
    group_columns  per ogni gruppo, le colonne di U che gli appartengono
    factor_names   tutti i k fattori dello spazio
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
        """Da una colonna di risposte ai blocchi: (fA, fB, fAB) con fAB
        di forma (g, n). n < N tiene le prime n righe di ogni blocco
        (sequenza di Sobol estendibile: è come aver campionato con n)"""
        y = np.asarray(y, dtype=float).reshape(self.n_blocks, self.N)
        n = self.N if n is None else int(n)
        if not 1 <= n <= self.N:
            raise ValueError(f"n deve stare fra 1 e N={self.N}")
        return y[0, :n], y[1, :n], y[2:, :n]

    def cost_table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "N": self.N,
            "gruppi": self.n_groups,
            "blocchi": self.n_blocks,
            "righe_theta": self.n_evaluations,
            "fattori_totali": len(self.factor_names),
        }])


def sobol_design(space, selected: list, N: int = 512, seed: int = 1) -> SobolDesign:
    """Disegno di Saltelli: un gruppo per ogni fattore selezionato, più
    il gruppo RESIDUAL_GROUP con tutti gli altri (sempre presente se
    qualcosa resta fuori, perché è il controllo di quanto pesa ciò che
    non si è voluto analizzare).

    A e B vengono dalle prime e dalle seconde k colonne di una sequenza
    di Sobol in 2k dimensioni (campioni indipendenti a bassa discrepanza)
    """
    names = list(space.names)
    k = len(names)

    if not selected:
        raise ValueError("selected è vuoto: elencare i parametri da analizzare")
    doppioni = sorted({nm for nm in selected if selected.count(nm) > 1})
    if doppioni:
        raise ValueError(f"Fattori ripetuti in selected: {doppioni}")
    ignoti = [nm for nm in selected if nm not in names]
    if ignoti:
        raise ValueError(f"Fattori sconosciuti: {ignoti}. Disponibili: {names}")
    if N & (N - 1):
        warnings.warn(f"N = {N} non è una potenza di 2: la sequenza di Sobol "
                      "perde il bilanciamento", RuntimeWarning)

    group_names = list(selected)
    group_columns = [[names.index(nm)] for nm in selected]
    rest = [j for j, nm in enumerate(names) if nm not in selected]
    if rest:
        group_names.append(RESIDUAL_GROUP)
        group_columns.append(rest)

    sampler = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    with warnings.catch_warnings():
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
# Stimatori (vettorizzati)
# ---------------------------------------------------------------------

def _apply_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """None: varianza di Y. "log": varianza di log(Y). I valori non
    positivi diventano NaN (l'intensity è positiva per costruzione)"""
    if transform is None:
        return np.asarray(y, dtype=float)
    if transform != "log":
        raise ValueError("transform può essere None oppure 'log'")
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y > 0, np.log(y), np.nan)


def _estimate_batch(a, b, ab, min_valid: int = 8) -> tuple:
    """(S_i, S_Ti, n_valid) lungo l'ultimo asse, con broadcasting sugli
    altri, ignorando le coppie con NaN.

      S_i    Saltelli 2010:  mean(f_B (f_AB - f_A)) / V
      S_Ti   Jansen 1999:    mean((f_A - f_AB)^2) / (2V)

    V è la varianza del campione A+B (coppie valide), e le risposte
    vengono centrate prima di stimare S_i per non perdere cifre
    significative sulla covarianza
    """
    a, b, ab = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float),
                                   np.asarray(ab, float))
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(ab)
    n = ok.sum(axis=-1)
    nn = np.maximum(n, 1)[..., None]

    a = np.where(ok, a, 0.0)
    b = np.where(ok, b, 0.0)
    ab = np.where(ok, ab, 0.0)
    f0 = (a.sum(-1, keepdims=True) + b.sum(-1, keepdims=True)) / (2.0 * nn)
    ac = np.where(ok, a - f0, 0.0)
    bc = np.where(ok, b - f0, 0.0)
    abc = np.where(ok, ab - f0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = ac.sum(-1) + bc.sum(-1)
        s2 = (ac ** 2).sum(-1) + (bc ** 2).sum(-1)
        nf = nn[..., 0]
        V = (s2 - s1 ** 2 / (2.0 * nf)) / (2.0 * nf - 1.0)
        S = (bc * (abc - ac)).sum(-1) / nf / V
        ST = ((ac - abc) ** 2).sum(-1) / (2.0 * nf) / V

    bad = (n < min_valid) | ~np.isfinite(V) | ~(V > 0)
    S = np.where(bad, np.nan, S)
    ST = np.where(bad, np.nan, ST)
    return S, ST, n


# ---------------------------------------------------------------------
# Il campo di indici
# ---------------------------------------------------------------------

@dataclass
class SobolField:
    """Gli indici di tutti gli output, più le repliche bootstrap che
    servono ad aggregare (aggregation.py).

    outputs      DataFrame, una riga per output (stesso ordine degli
                 array): i metadati (sistema, propulsore, range, ...)
                 più V (varianza della risposta trasformata sul campione
                 A+B), frazione_valida e fattibile
    group_names  i fattori/gruppi, nell'ordine del secondo asse
    S1, ST       (n_out, g) stime puntuali
    ST_boot      (n_out, g, B) repliche bootstrap di S_T (B = 0 se
                 n_boot = 0)
    V_boot       (n_out, B) repliche della varianza, con gli stessi
                 ricampionamenti
    table        tabella tidy (output, fattore, S1, ST, intervalli, ...)
    """
    outputs: pd.DataFrame
    group_names: list
    S1: np.ndarray
    ST: np.ndarray
    ST_boot: np.ndarray
    V_boot: np.ndarray
    table: pd.DataFrame
    N: int
    transform: Optional[str]
    nan_tol: float

    @property
    def n_boot(self) -> int:
        return self.ST_boot.shape[-1]

    def mask(self, **criteria) -> np.ndarray:
        """Maschera booleana sugli output, per esempio
        field.mask(sistema="Hydrogen fuel cell", propulsore="fan")"""
        m = np.ones(len(self.outputs), dtype=bool)
        for col, val in criteria.items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            m &= self.outputs[col].isin(vals).to_numpy()
        return m


def sobol_field(design: SobolDesign, Y: pd.DataFrame,
                meta: Optional[pd.DataFrame] = None, n_boot: int = 200,
                seed: int = 0, n: Optional[int] = None,
                transform: Optional[str] = "log", nan_tol: float = 0.10,
                verbose: bool = False) -> SobolField:
    """Indici di Sobol di tutte le colonne di Y.

    Y     (design.n_evaluations, n_out), calcolata su design.U senza
          riordinare le righe (è quello che restituisce evaluate_outputs)
    meta  metadati degli output, indicizzati per nome di colonna
          (outputs_table(...).set_index("output")). Se None si usano
          solo i nomi
    nan_tol  frazione massima di NaN sul campione A+B perché un output
          sia considerato fattibile, e quindi entri in aggregati e mappe
    """
    if len(Y) != design.n_evaluations:
        raise ValueError(f"Y ha {len(Y)} righe, il disegno ne prevede "
                         f"{design.n_evaluations}: va calcolata su design.U")

    n_eff = design.N if n is None else int(n)
    g = design.n_groups
    cols = list(Y.columns)
    n_out = len(cols)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_eff, size=(n_boot, n_eff)) if n_boot > 0 else None

    S1 = np.full((n_out, g), np.nan)
    ST = np.full((n_out, g), np.nan)
    ST_lo = np.full((n_out, g), np.nan)
    ST_hi = np.full((n_out, g), np.nan)
    S1_lo = np.full((n_out, g), np.nan)
    S1_hi = np.full((n_out, g), np.nan)
    NV = np.zeros((n_out, g), dtype=int)
    ST_boot = np.full((n_out, g, max(n_boot, 0)), np.nan, dtype=np.float32)
    V_boot = np.full((n_out, max(n_boot, 0)), np.nan)
    V = np.full(n_out, np.nan)
    frac = np.zeros(n_out)

    for o, col in enumerate(cols):
        y = _apply_transform(Y[col].to_numpy(dtype=float), transform)
        fA, fB, fAB = design.split(y, n=n_eff)

        ab = np.concatenate([fA, fB])
        fin = np.isfinite(ab)
        frac[o] = fin.mean()
        if fin.sum() > 2:
            V[o] = np.var(ab[fin], ddof=1)

        S, T, nv = _estimate_batch(fA[None, :], fB[None, :], fAB)
        S1[o], ST[o], NV[o] = S, T, nv

        if n_boot > 0 and fin.any():
            a, b = fA[idx], fB[idx]                         # (B, n)
            Sb, Tb, _ = _estimate_batch(a[None], b[None], fAB[:, idx])   # (g, B)
            ST_boot[o] = Tb.astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q = np.nanpercentile(Tb, [2.5, 97.5], axis=1)
                ST_lo[o], ST_hi[o] = q[0], q[1]
                q = np.nanpercentile(Sb, [2.5, 97.5], axis=1)
                S1_lo[o], S1_hi[o] = q[0], q[1]
                V_boot[o] = np.nanvar(np.concatenate([a, b], axis=1), axis=1, ddof=1)

        if verbose and (o + 1) % 100 == 0:
            print(f"    indici: {o + 1}/{n_out} output")

    # metadati
    if meta is None:
        out = pd.DataFrame({"output": cols})
    else:
        out = meta.reindex(cols).reset_index()
        out = out.rename(columns={out.columns[0]: "output"})
    out["V"] = V
    out["frazione_valida"] = frac
    out["fattibile"] = (frac >= 1.0 - nan_tol) & np.isfinite(V) & (V > 0) \
        & np.isfinite(ST).all(axis=1)

    # tabella tidy
    rows = {
        "output": np.repeat(cols, g),
        "fattore": np.tile(design.group_names, n_out),
        "S1": S1.ravel(), "S1_lo": S1_lo.ravel(), "S1_hi": S1_hi.ravel(),
        "ST": ST.ravel(), "ST_lo": ST_lo.ravel(), "ST_hi": ST_hi.ravel(),
        "n_valid": NV.ravel(),
    }
    table = pd.DataFrame(rows)
    table["interazione"] = table["ST"] - table["S1"]
    table["frazione_valida"] = table["n_valid"] / n_eff
    table["N"] = n_eff
    table["trasformazione"] = transform or "nessuna"
    table = table.merge(out[["output", "fattibile"]], on="output", how="left")

    return SobolField(outputs=out, group_names=list(design.group_names), S1=S1, ST=ST,
                      ST_boot=ST_boot, V_boot=V_boot, table=table, N=n_eff,
                      transform=transform, nan_tol=nan_tol)


# ---------------------------------------------------------------------
# Diagnostiche
# ---------------------------------------------------------------------

def tail_report(design: SobolDesign, Y: pd.DataFrame, soglia: float = 5.0,
                nan_tol: float = 0.10) -> pd.DataFrame:
    """Lunghezza della coda di ciascun output, sul campione onesto A+B.

      coda_p99   99-esimo percentile / mediana: decide la varianza
      coda_max   massimo / mediana: a occhio, lo decide un campione

    Da leggere con frazione_nan: avvicinandosi al confine di
    fattibilità i campioni peggiori diventano NaN e la coda MIGLIORA
    mentre il problema peggiora (censura). Coda corta + molti NaN =
    troncato, non sano
    """
    rows = []
    for col in Y.columns:
        fA, fB, _ = design.split(Y[col].to_numpy())
        v = np.concatenate([fA, fB])
        frazione_nan = 1.0 - np.isfinite(v).mean()
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) == 0:
            rows.append({"output": col, "coda_p99": np.nan, "coda_max": np.nan,
                         "frazione_nan": 1.0, "verdetto": "nessun campione valido"})
            continue
        med = float(np.median(v))
        coda_p99 = float(np.percentile(v, 99)) / med
        coda_max = float(v.max()) / med
        if frazione_nan > nan_tol:
            verdetto = "censurato dai NaN"
        elif coda_p99 > soglia:
            verdetto = "coda lunga"
        else:
            verdetto = "ok"
        rows.append({"output": col, "coda_p99": coda_p99, "coda_max": coda_max,
                     "frazione_nan": frazione_nan, "verdetto": verdetto})
    return pd.DataFrame(rows).sort_values("coda_p99", ascending=False).reset_index(drop=True)


def residual_report(fld: SobolField, by: str = "architettura") -> pd.DataFrame:
    """Quanto pesa ciò che non è stato selezionato: S_T[others] sugli
    output fattibili, riassunto per architettura (o per altra colonna
    dei metadati). Non è più un controllo dello screening ma un
    risultato: la quota di incertezza non tecnologica"""
    if RESIDUAL_GROUP not in fld.group_names:
        raise ValueError("Nessun gruppo 'others': tutti i fattori sono selezionati")
    j = fld.group_names.index(RESIDUAL_GROUP)
    df = fld.outputs.loc[fld.outputs["fattibile"], ["output", by]].copy()
    df["ST_others"] = fld.ST[fld.outputs["fattibile"].to_numpy(), j]
    return (df.groupby(by)["ST_others"]
            .agg(mediana="median", p90=lambda s: s.quantile(0.9), massimo="max",
                 n_punti="size")
            .sort_values("mediana", ascending=False))


def sobol_table(fld: SobolField, index: str = "ST", only_feasible: bool = True) -> pd.DataFrame:
    """Matrice output x fattori di un indice ("S1", "ST", "interazione")"""
    if index not in ("S1", "ST", "interazione"):
        raise ValueError("index deve essere 'S1', 'ST' o 'interazione'")
    t = fld.table
    if only_feasible:
        t = t[t["fattibile"]]
    return t.pivot(index="output", columns="fattore", values=index)[fld.group_names]
