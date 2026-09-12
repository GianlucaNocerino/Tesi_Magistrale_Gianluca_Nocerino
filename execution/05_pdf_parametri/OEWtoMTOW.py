"""
Regressione bayesiana del rapporto OEW/MTOW in funzione del MTOW.

Modello:
    OEW/MTOW = a * MTOW**b + c + errore

Il coefficiente 'a' è FISSATO (valori diversi per fan ed elica, vedi A_FISSO);
i parametri liberi sono b, c e la dispersione residua sigma, con il vincolo
b > 0 e c > 0. Essendo a < 0, il rapporto decresce monotonamente con il MTOW:
c'è il valore limite per MTOW -> 0 e il termine a*MTOW**b la correzione di taglia.

Il MTOW entra in kg senza normalizzazione.
I dati di riferimento sono elencati direttamente in DATI: lo script è autonomo,
non serve nessun file esterno.

Lo script fa poi tre cose in sequenza:
  1. campiona le posterior di (b, c, sigma) con NUTS;
  2. ne ricava una specifica di PDF assegnate, pronta da mettere in tabella,
     in coordinate (b, r_pivot) dove i due parametri risultano indipendenti;
  3. verifica che le PDF riproducano la posteriori, e stampa a terminale il
     blocco di ParameterSpec pronto da incollare in pdf_specs.py.
 
L'unico file scritto su disco sono i grafici dei fit, in OUT_DIR.
 
Uso:
    python execution/05_pdf_parametri/OEWtoMTOW.py
 
Dipendenze:
    pip install pymc arviz pandas numpy scipy matplotlib
"""
 
from __future__ import annotations
 
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
 
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from scipy.optimize import brentq
 
# --------------------------------------------------------------------------
# DATI DI RIFERIMENTO  (Model, MTOW [kg], OEW [kg])
# --------------------------------------------------------------------------
 
DATI = {
    "turbofan": [
        ("A340_211",    260000, 127000),
        ("A350_941",    280000, 141700),
        ("A300_B2_1A",  137000,  85910),
        ("A310_203",    132000,  79207),
        ("A318_111",     68000,  39500),
        ("A320_211",     73500,  42600),
        ("A321_111",     89000,  48500),
        ("A330_201",    230000, 125206),
        ("B701B_21",    117000,  57600),
        ("B703_211",    141700,  64600),
        ("B712_3A10",    49895,  30617),
        ("B721_27",      72600,  39800),
        ("B722_27",      78100,  44330),
        ("B731_27",      44361,  26581),
        ("B737_17B",     60328,  37648),
        ("B73MAX8",      82644,  45000),
        ("B742B_27A",   352800, 172860),
        ("B752_353",     99800,  60800),
        ("B772_4903",   229500, 135550),
        ("CRJ_100",      21523,  13835),
        ("CL_600_2C10",  32999,  19731),
        ("ERJ_170_100",  37200,  20700),
        ("DC_10_10",    195045, 108940),
        ("MD_11",       276691, 128808),
        ("EMB_135ER",    18990,  11402),
    ],
    "turboprop": [
        ("ATR42_300",    16700,  10900),
        ("ATR42_400",    17900,  11400),
        ("ATR42_500",    18600,  11250),
        ("ATR42_600",    18600,  11550),
        ("ATR72_200",    21500,  13000),
        ("ATR72_500",    22000,  13600),
        ("ATR72_600",    22800,  13500),
        ("L100_30",      70370,  35260),
        ("LM_100J",      74389,  36446),
        ("EMB_120_FC",   11990,   7177),
        ("EMB_110_P1",    5670,   3564),
        ("Beech_1900D",   7815,   4932),
    ],
}
 
# --------------------------------------------------------------------------
# CONFIGURAZIONE
# --------------------------------------------------------------------------
 
OUT_DIR = Path(__file__).resolve().parent   # unica cartella scritta: solo i grafici dei fit
 
# Coefficiente 'a' imposto, per tipo di propulsore [1/kg**b]
A_FISSO = {
    "turbofan": -0.0004549,
    "turboprop": -0.00002678,
}
 
N_DRAWS = 2000
N_TUNE = 3000
N_CHAINS = 4
TARGET_ACCEPT = 0.995
CORES = 1             # metti 4 per campionare le catene in parallelo
SEED = 42
 
CI = 90.0
QLO, QHI = (100 - CI) / 2, 100 - (100 - CI) / 2
 
 
# --------------------------------------------------------------------------
# DATI
# --------------------------------------------------------------------------
 
def carica_dataset() -> pd.DataFrame:
    """Costruisce il DataFrame dai dati elencati in DATI e aggiunge OEW/MTOW."""
    righe = [
        {"Model": nome, "Type": tipo, "MTOW_kg": mtow, "OEW_kg": oew}
        for tipo, voci in DATI.items()
        for nome, mtow, oew in voci
    ]
    df = pd.DataFrame(righe)
    df["ratio"] = df["OEW_kg"] / df["MTOW_kg"]
    return df
 
 
# --------------------------------------------------------------------------
# USO DELLE PDF A VALLE
# Queste due funzioni sono tutto ciò che serve per propagare l'incertezza in
# un altro programma, senza rieseguire l'MCMC. Il dizionario k è quello
# prodotto da pdf_assegnata(), gli stessi numeri stampati nel blocco finale.
#
#   b, c = campiona_da_specifica(k, n=10_000)      # UNA volta per run
#   r = rapporto_oew(mtow, b, c, k["a"])           # a qualunque MTOW serva
#
# L'estrazione va fatta una volta per run Monte Carlo, non a ogni MTOW: una
# coppia (b, c) è un'intera curva, e ricampionarla a ogni valutazione
# distruggerebbe la coerenza del velivolo lungo il ciclo di dimensionamento.
# --------------------------------------------------------------------------
 
def rapporto_oew(mtow, b, c, a: float):
    """OEW/MTOW = a*MTOW**b + c. Accetta scalari o array (broadcasting)."""
    return a * np.asarray(mtow, float) ** b + c
 
 
def campiona_da_specifica(k: dict, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Estrae n coppie (b, c) dalle PDF assegnate descritte dal dizionario k
    (l'output di ModelloRapportoOEW.pdf_assegnata())
    """
    rng = np.random.default_rng() if rng is None else rng
    b = k["b_max"] - np.exp(rng.normal(k["b_log_mu"], k["b_log_sd"], size=n))
    b = np.clip(b, 1e-4, None)
    r = rng.normal(k["r_mu"], k["r_sd"], size=n)
    c = r - k["a"] * k["mtow_pivot"] ** b
    return b, c
 
 
def controlla_range(mtow, k: dict) -> None:
    """Avvisa se si sta estrapolando fuori dal dataset di calibrazione"""
    m = np.atleast_1d(np.asarray(mtow, float))
    if m.min() < k["mtow_min"] or m.max() > k["mtow_max"]:
        print(f"[avviso] MTOW fuori dal range di calibrazione {k['tipo']} "
              f"({k['mtow_min']:,.0f}-{k['mtow_max']:,.0f} kg): la curva decresce "
              f"senza asintoto e puo' dare valori privi di senso.")
 
 
# --------------------------------------------------------------------------
# MODELLO
# --------------------------------------------------------------------------
 
@dataclass
class ModelloRapportoOEW:
    """
    Regressione bayesiana di OEW/MTOW = a*MTOW**b + c, con 'a' assegnato.
 
    Parametri
    ---------
    mtow, ratio : array dei dati del gruppo (fan oppure elica)
    nome        : etichetta usata in stampe e nomi file
    a           : coefficiente fissato (non campionato)
    """
 
    mtow: np.ndarray
    ratio: np.ndarray
    nome: str
    a: float
    idata: az.InferenceData | None = field(default=None, repr=False)
 
    def __post_init__(self):
        self.mtow = np.asarray(self.mtow, dtype=float)
        self.ratio = np.asarray(self.ratio, dtype=float)
        self.a = float(self.a)
 
    # ---------------------------------------------------------------- modello
    def costruisci(self) -> pm.Model:
        """Definisce priors e verosimiglianza"""
        x = self.mtow
        y = self.ratio
 
        with pm.Model() as modello:
            # b > 0, con limite superiore fisico: b = 1 significa che l'OEW
            # smette del tutto di crescere con il MTOW, quindi oltre ~1.5 il
            # modello non ha piu' senso. Il limite superiore serve anche
            # numericamente: senza, il campionatore puo' esplorare b ~ 3-4,
            # dove MTOW**b vale 10^15 e la verosimiglianza diventa una parete.
            b = pm.TruncatedNormal("b", mu=0.5, sigma=0.4, lower=0.0, upper=1.5)
 
            # c > 0: valore del rapporto estrapolato a MTOW -> 0.
            c = pm.TruncatedNormal("c", mu=0.65, sigma=0.25, lower=0.0)
 
            # dispersione residua: raccoglie tutto cio' che il modello non
            # spiega (missione, materiali, epoca di progetto...)
            sigma = pm.HalfNormal("sigma", 0.05)
 
            mu = self.a * x**b + c
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
 
        self.modello = modello
        return modello
 
    def fit(self) -> az.InferenceData:
        """Campiona la posteriori con NUTS."""
        if not hasattr(self, "modello"):
            self.costruisci()
        with self.modello:
            self.idata = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                cores=CORES,
                target_accept=TARGET_ACCEPT,
                random_seed=SEED,
                progressbar=True,
            )
        return self.idata
 
    # ------------------------------------------------------------- estrazione
    def campioni(self) -> dict[str, np.ndarray]:
        """Campioni a posteriori appiattiti (catene concatenate)"""
        post = self.idata.posterior
        return {k: post[k].values.ravel() for k in ("b", "c", "sigma")}
 
    def tabella_sintesi(self) -> pd.DataFrame:
        """Media, mediana, sd, intervallo di credibilità e diagnostica MCMC"""
        righe = []
        for nome in ("b", "c", "sigma"):
            v = self.idata.posterior[nome].values.ravel()
            righe.append({
                "parametro": nome,
                "media": v.mean(),
                "mediana": np.median(v),
                "sd": v.std(ddof=1),
                f"q{QLO:g}": np.percentile(v, QLO),
                f"q{QHI:g}": np.percentile(v, QHI),
                "r_hat": float(az.rhat(self.idata, var_names=[nome])[nome].values),
                "ess_bulk": float(az.ess(self.idata, var_names=[nome])[nome].values),
            })
        return pd.DataFrame(righe).set_index("parametro")
 
    def residui(self) -> np.ndarray:
        """Residui rispetto alla curva con b e c mediani"""
        s = self.campioni()
        b, c = np.median(s["b"]), np.median(s["c"])
        return self.ratio - (self.a * self.mtow**b + c)
 
    # ------------------------------------------------------------- predizione
    def curve_posteriori(self, mtow_grid: np.ndarray) -> dict[str, np.ndarray]:
        """
        Valuta la curva su una griglia di MTOW per ogni campione a posteriori.
 
        Ritorna:
          mediana  : curva mediana
          mu_lo/hi : banda di credibilità sulla MEDIA (incertezza sui parametri)
          pi_lo/hi : banda predittiva su un NUOVO velivolo (parametri + sigma)
        """
        s = self.campioni()
        xg = np.asarray(mtow_grid, float)[None, :]
        mu = self.a * xg ** s["b"][:, None] + s["c"][:, None]
 
        rng = np.random.default_rng(SEED)
        n_rep = 5
        mu_rep = np.tile(mu, (n_rep, 1))
        sig_rep = np.tile(s["sigma"][:, None], (n_rep, 1))
        y_new = mu_rep + rng.normal(0.0, sig_rep, size=mu_rep.shape)
 
        return {
            "mediana": np.median(mu, axis=0),
            "mu_lo": np.percentile(mu, QLO, axis=0),
            "mu_hi": np.percentile(mu, QHI, axis=0),
            "pi_lo": np.percentile(y_new, QLO, axis=0),
            "pi_hi": np.percentile(y_new, QHI, axis=0),
        }
 
    def valuta(self, mtow: float) -> tuple[float, float, float]:
        """Rapporto OEW/MTOW previsto a un dato MTOW: (mediana, q_lo, q_hi)"""
        c = self.curve_posteriori(np.array([mtow]))
        return float(c["mediana"][0]), float(c["mu_lo"][0]), float(c["mu_hi"][0])
 
    # ------------------------------------------------ PDF assegnate
    def mtow_pivot(self) -> float:
        """
        MTOW al quale il rapporto previsto risulta SCORRELATO da b.
 
        b e c sono correlati (~ +0.82) e la loro dipendenza non è lineare, per
        cui una copula a un solo parametro non li descrive bene. Cambiando
        coordinate da (b, c) a (b, r_pivot), con
            r_pivot = a*MTOW_pivot**b + c
        la correlazione si annulla per costruzione, e le due quantità
        risultano indipendenti (vedi verifica_indipendenza)
        """
        s = self.campioni()
        b, c = s["b"], s["c"]
        f = lambda lx: np.corrcoef(b, self.a * np.exp(lx) ** b + c)[0, 1]
        return float(np.exp(brentq(f, np.log(1e3), np.log(1e7))))
 
    def pdf_assegnata(self) -> dict:
        """
        Ricava la specifica delle PDF da assegnare a b e c.
 
            b       : lognormale riflessa,  ln(b_max - b) ~ N(b_log_mu, b_log_sd)
            r_pivot : normale,              N(r_mu, r_sd)
            c       : ricavata da  c = r_pivot - a*MTOW_pivot**b
 
        La riflessione serve a riprodurre l'asimmetria sinistra di b, che le
        famiglie simmetriche non catturano.
        """
        s = self.campioni()
        b, c = s["b"], s["c"]
        xp = self.mtow_pivot()
        r = self.a * xp ** b + c
 
        b_max = float(b.max()) + 1e-3
        forma, _, scala = stats.lognorm.fit(b_max - b, floc=0.0)
 
        log_mu, log_sd = float(np.log(scala)), float(forma)
 
        return {
            "tipo": self.nome,
            "a": self.a,
            "mtow_pivot": xp,
            # --- parametri della PDF di b (lognormale riflessa) ---
            "b_max": b_max,
            "b_log_mu": log_mu,
            "b_log_sd": log_sd,
            # --- parametri della PDF di r_pivot (normale) ---
            "r_mu": float(r.mean()),
            "r_sd": float(r.std(ddof=1)),
            # --- valori notevoli, per i campi 'nominal' e 'mode' della tabella ---
            "b_moda": b_max - float(np.exp(log_mu - log_sd ** 2)),
            "b_mediana": b_max - float(np.exp(log_mu)),
            "r_moda": float(r.mean()),
            "c_mediana": float(np.median(c)),
            # --- range di validita' ---
            "mtow_min": float(self.mtow.min()),
            "mtow_max": float(self.mtow.max()),
        }
 
    def verifica_indipendenza(self) -> dict:
        """Controlla che b e r_pivot siano indipendenti, non solo scorrelati"""
        s = self.campioni()
        b, c = s["b"], s["c"]
        r = self.a * self.mtow_pivot() ** b + c
        quart = np.quantile(b, [0.0, 0.25, 0.5, 0.75, 1.0])
        medie = [float(r[(b >= quart[i]) & (b <= quart[i + 1])].mean()) for i in range(4)]
        return {
            "pearson": float(np.corrcoef(b, r)[0, 1]),
            "spearman": float(stats.spearmanr(b, r)[0]),
            "media_r_per_quartile_di_b": medie,
        }
 
    def campiona_da_pdf(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
        """Estrae (b, c) dalle PDF assegnate, senza usare i campioni MCMC"""
        return campiona_da_specifica(self.pdf_assegnata(), n, rng)
 
    def verifica_pdf(self, n: int = 200_000, rng=None) -> pd.DataFrame:
        """
        Confronta gli intervalli di credibilità del rapporto ottenuti dalla
        posteriori con quelli generati dalle PDF assegnate, su tutto il range
        """
        rng = np.random.default_rng(SEED) if rng is None else rng
        s = self.campioni()
        bp, cp = self.campiona_da_pdf(n, rng)
 
        righe = []
        for x in np.geomspace(self.mtow.min(), self.mtow.max(), 6):
            e = self.a * x ** s["b"] + s["c"]
            p = self.a * x ** bp + cp
            qe = np.percentile(e, [QLO, 50, QHI])
            qp = np.percentile(p, [QLO, 50, QHI])
            righe.append({
                "MTOW_kg": x,
                "post_med": qe[1], "post_lo": qe[0], "post_hi": qe[2],
                "pdf_med": qp[1], "pdf_lo": qp[0], "pdf_hi": qp[2],
                "err_ampiezza_%": 100 * ((qp[2] - qp[0]) / (qe[2] - qe[0]) - 1),
            })
        return pd.DataFrame(righe)
 
 
# --------------------------------------------------------------------------
# GRAFICI
# --------------------------------------------------------------------------
 
def grafico_fit(m: ModelloRapportoOEW, outdir: Path) -> None:
    """Dati, curva mediana, banda sulla media e banda predittiva"""
    xg = np.logspace(np.log10(m.mtow.min() * 0.8), np.log10(m.mtow.max() * 1.2), 300)
    cur = m.curve_posteriori(xg)
 
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(xg, cur["pi_lo"], cur["pi_hi"], alpha=0.15, color="C0",
                    label=f"banda predittiva {CI:g}% (nuovo velivolo)")
    ax.fill_between(xg, cur["mu_lo"], cur["mu_hi"], alpha=0.35, color="C0",
                    label=f"banda di credibilità {CI:g}% (curva media)")
    ax.plot(xg, cur["mediana"], color="C0", lw=2, label="mediana a posteriori")
    ax.plot(m.mtow, m.ratio, "o", color="k", ms=5, label="dati")
 
    s = m.campioni()
    testo = (f"a = {m.a:.6g} (fisso)\n"
             f"b = {np.median(s['b']):.4f}\n"
             f"c = {np.median(s['c']):.4f}")
    ax.text(0.03, 0.05, testo, transform=ax.transAxes, fontsize=9,
            va="bottom", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
 
    ax.set_xscale("log")
    ax.set_xlabel("MTOW [kg]")
    ax.set_ylabel("OEW / MTOW [-]")
    ax.set_title(f"Regressione bayesiana - {m.nome}  (n = {len(m.mtow)})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"fit_{m.nome}.png", dpi=150)
    plt.close(fig)
 
 
def grafico_confronto(modelli: list[ModelloRapportoOEW], outdir: Path) -> None:
    """Sovrappone le curve dei due gruppi"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, m in enumerate(modelli):
        xg = np.logspace(np.log10(m.mtow.min() * 0.8), np.log10(m.mtow.max() * 1.2), 300)
        cur = m.curve_posteriori(xg)
        col = f"C{i}"
        ax.fill_between(xg, cur["mu_lo"], cur["mu_hi"], alpha=0.25, color=col)
        ax.plot(xg, cur["mediana"], color=col, lw=2, label=m.nome)
        ax.plot(m.mtow, m.ratio, "o", color=col, ms=5, mec="k", mew=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("MTOW [kg]")
    ax.set_ylabel("OEW / MTOW [-]")
    ax.set_title(f"Confronto fan / elica - bande al {CI:g}% sulla curva media")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "confronto_fan_elica.png", dpi=150)
    plt.close(fig)
 
 
# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
 
RATIONALE_B = (
    "esponente di taglia. La posterior è asimmetrica a sinistra: la coda "
    "verso b piccolo è l'ipotesi 'nessuna dipendenza dal MTOW', che i dati "
    "non escludono del tutto. Le famiglie simmetriche non la catturano, da "
    "cui la lognormale riflessa. Indipendente da oew_r_pivot_{suf} "
    "(Spearman {sp:+.3f})"
)
 
RATIONALE_R = (
    "rapporto OEW/MTOW a MTOW_pivot = {pivot} kg. Il pivot è scelto "
    "risolvendo corr(b, r_pivot) = 0: parametrizzando invece con c (che e' il "
    "rapporto estrapolato a MTOW = 0) la correlazione con b sarebbe {rbc:+.2f} "
    "e non lineare. Validita': MTOW da {lo} a {hi} kg"
)
 
 
def _kg(v: float) -> str:
    """Formatta un peso con separatore delle migliaia leggibile"""
    return f"{v:,.0f}".replace(",", " ")
 
 
def _rationale(testo: str, indent: int = 16) -> str:
    """Manda a capo il rationale e lo quota riga per riga, stile pdf_specs.py"""
    righe = textwrap.wrap(testo, width=72 - indent + 16)
    sp = " " * indent
    return ("\n" + sp).join(f'"{r} "' if i < len(righe) - 1 else f'"{r}"'
                             for i, r in enumerate(righe))
 
 
def blocco_parameterspec(k: dict, ind: dict, rbc: float, suf: str) -> str:
    """Genera il testo Python dei due ParameterSpec, pronto da incollare"""
    return f'''
        # --- frazione OEW/MTOW, ramo {k["tipo"]} ---
        # OEW/MTOW = a*MTOW**b + c,  a = {k["a"]:.6g} (fisso)
        # MTOW_pivot = {_kg(k["mtow_pivot"])} kg
        # c NON è un parametro incerto: c = r_pivot - a*MTOW_pivot**b
        ParameterSpec(
            name="oew_b_{suf}",
            dist=reflected_lognormal({k["b_max"]:.4f}, {k["b_log_mu"]:.4f}, {k["b_log_sd"]:.4f}),
            nominal={k["b_mediana"]:.4f},
            pdf_label="{k["b_max"]:.4f} - exp(N({k["b_log_mu"]:.4f}, {k["b_log_sd"]:.4f}))",
            mode={k["b_moda"]:.4f},
            source="Regressione bayesiana su velivoli {k["tipo"]} storici",
            rationale=(
                {_rationale(RATIONALE_B.format(suf=suf, sp=ind["spearman"]))}
            ),
        ),
        ParameterSpec(
            name="oew_r_pivot_{suf}",
            dist=stats.norm(loc={k["r_mu"]:.5f}, scale={k["r_sd"]:.5f}),
            nominal={k["r_mu"]:.5f},
            pdf_label="N({k["r_mu"]:.5f}, {k["r_sd"]:.5f})",
            mode={k["r_moda"]:.5f},
            source="Regressione bayesiana su velivoli {k["tipo"]} storici",
            rationale=(
                {_rationale(RATIONALE_R.format(pivot=_kg(k["mtow_pivot"]), rbc=rbc, lo=_kg(k["mtow_min"]), hi=_kg(k["mtow_max"])))}
            ),
        ),'''
 
 
PREAMBOLO = '''
# ==========================================================================
#  Da incollare in pdf_specs.py
#  Serve questo helper accanto a triangular() e scaled_beta():
#
#   class ReflectedLognormal:
#       """b = b_max - exp(X), X ~ N(log_mu, log_sd)."""
#       def __init__(self, b_max, log_mu, log_sd):
#           self.b_max = b_max
#           self._ln = stats.lognorm(s=log_sd, scale=np.exp(log_mu))
#       def rvs(self, size=None, random_state=None):
#           return self.b_max - self._ln.rvs(size=size, random_state=random_state)
#       def ppf(self, q):  return self.b_max - self._ln.ppf(1.0 - np.asarray(q))
#       def cdf(self, x):  return 1.0 - self._ln.cdf(self.b_max - np.asarray(x))
#       def pdf(self, x):  return self._ln.pdf(self.b_max - np.asarray(x))
#       def mean(self):    return self.b_max - self._ln.mean()
#       def std(self):     return self._ln.std()
#
#   def reflected_lognormal(b_max, log_mu, log_sd):
#       return ReflectedLognormal(b_max, log_mu, log_sd)
#
#  I due 'a' e i due MTOW_pivot sono costanti, non parametri incerti:
#  vanno in constants.py. Nel CorrelationModel non va aggiunto nulla,
#  b e r_pivot sono indipendenti.
#
#  'nominal' è impostato al valore centrale della PDF: sostituiscilo con
#  quello del modello deterministico, se ne hai già uno.
# ==========================================================================
'''
 
 
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = carica_dataset()
 
    modelli: list[ModelloRapportoOEW] = []
    blocchi: list[str] = []
 
    for tipo, gruppo in df.groupby("Type"):
        if tipo not in A_FISSO:
            print(f"[avviso] nessun valore di 'a' per '{tipo}': gruppo saltato")
            continue
 
        m = ModelloRapportoOEW(
            mtow=gruppo["MTOW_kg"].to_numpy(),
            ratio=gruppo["ratio"].to_numpy(),
            nome=tipo,
            a=A_FISSO[tipo],
        )
        m.fit()
 
        # --- controllo minimo che la corsa sia valida ---------------------
        tab = m.tabella_sintesi()
        ver = m.verifica_pdf()
        div = int(m.idata.sample_stats.diverging.values.sum())
        s = m.campioni()
        rbc = float(np.corrcoef(s["b"], s["c"])[0, 1])
        print(f"[{tipo}] n={len(gruppo)}  r_hat max {tab['r_hat'].max():.3f}  "
              f"ESS min {tab['ess_bulk'].min():.0f}  divergenze {div}  "
              f"residuo rms {m.residui().std():.4f}  "
              f"err. max ampiezza PDF {ver['err_ampiezza_%'].abs().max():.1f}%")
 
        suf = "fan" if tipo == "turbofan" else "prop"
        blocchi.append(blocco_parameterspec(m.pdf_assegnata(), m.verifica_indipendenza(), rbc, suf))
 
        grafico_fit(m, OUT_DIR)
        modelli.append(m)
 
    if modelli:
        grafico_confronto(modelli, OUT_DIR)
 
    print(PREAMBOLO)
    for b in blocchi:
        print(b)
 
 
if __name__ == "__main__":
    main()
 



