"""
Le PDF dei parametri incerti, un grafico per parametro.

Affianca plot_parameter_distributions.py senza modificarlo: ne riusa il
caricamento di theta e i nominali. Cosa stampare e come (parametri,
titoli, etichette, colori, dimensioni, formati) si decide nella sezione
IMPOSTAZIONI qui sotto: è l'unica parte del file da toccare.

Uso:
    python execution/05_pdf_parametri/plot_pdf_singole.py
    python execution/05_pdf_parametri/plot_pdf_singole.py --parametri e_battery_Wh_per_kg eta_fuel_cell
    python execution/05_pdf_parametri/plot_pdf_singole.py --elenco

Produce:
    pdf_singole/<parametro>.<formato>   uno per parametro scelto

La curva teorica e il valore di riferimento non vengono ricalcolati: si
leggono dal pannello corrispondente della figura prodotta da
plot_parameter_distributions (che qui si costruisce in memoria e non si
salva). Così i grafici singoli restano identici alla figura d'insieme
anche se in cnav.uncertainty cambia il modo di valutare le PDF.
L'istogramma invece si ridisegna da theta, per poterne cambiare bin e
colori.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_parameter_distributions import (  # noqa: E402
    OUT_DIR,
    build_default_correlations,
    build_default_specs,
    carica_theta,
    nominali_di_calibrazione,
    parameter_sample_check,
    plot_parameter_distributions,
)


# ===========================================================================
# IMPOSTAZIONI
# Gerarchia (vince l'ultima): STILE -> colore di categoria -> PERSONALIZZAZIONI
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Quali parametri stampare
# ---------------------------------------------------------------------------
# None = tutti quelli presenti in theta. Altrimenti una lista di nomi, ad es.
#   PARAMETRI = ["e_battery_Wh_per_kg", "eta_fuel_cell", "oew_fan_b"]
# L'ordine della lista è l'ordine di stampa. Da riga di comando
# --parametri ha la precedenza su questa lista. Per vedere i nomi validi:
#   python .../plot_pdf_singole.py --elenco
PARAMETRI = None

# ---------------------------------------------------------------------------
# 2. Dove e in che formato salvare
# ---------------------------------------------------------------------------
CARTELLA = "pdf_singole"          # sottocartella di 05_pdf_parametri
FORMATI = ("png",)          # "pdf"/"svg" restano vettoriali per LaTeX
DPI = 300                         # usato solo dai formati raster

# ---------------------------------------------------------------------------
# 3. Stile di default, valido per tutti i grafici
# ---------------------------------------------------------------------------
STILE = dict(
    figsize=(6.0, 4.0),           # pollici; (3.3, 2.5) per mezza colonna
    bins=40,
    alpha_istogramma=0.35,
    colore=None,                  # None = colore della categoria (sotto)
    colore_curva=None,            # None = stesso colore dell'istogramma
    lw_curva=2.0,
    mostra_curva=True,            # PDF dichiarata (solo categoria "dichiarata")
    mostra_nominale=True,         # linea tratteggiata del valore di riferimento
    colore_nominale="black",
    titolo=None,                  # None = nome del parametro
    xlabel=None,                  # None = nessuna etichetta
    ylabel="Probability Density",
    mostra_asse_y=False,          # la scala della densità di rado dice qualcosa
    xlim=None,                    # None = gli stessi limiti della figura d'insieme
    griglia=False,
    legenda=True,
    fontsize_titolo=16,
    fontsize_assi=16,
    fontsize_tick=10,
    fontsize_legenda=15,
    posizione_legenda="best",    # "upper left", "upper right", ...

    etichetta_istogramma="Sample",
    etichetta_curva="Defined PDF",
    etichetta_nominale="Reference Value",
)

COLORI_CATEGORIA = {
    "dichiarata": "#4C72B0",
    "derivata":   "#DD8452",
    "empirica":   "#55A868",
}

# ---------------------------------------------------------------------------
# 4. Personalizzazioni per singolo parametro
# ---------------------------------------------------------------------------
# Qualsiasi chiave di STILE può essere ridefinita qui per un parametro solo.
# I testi accettano la sintassi LaTeX di matplotlib: r"$\eta_{FC}$ [-]".
PERSONALIZZAZIONI = {
    "e_battery_Wh_per_kg": dict(
        titolo="Battery Specific Energy",
        xlabel=r"$e_{battery}$ [Wh/kg]",
    ),
    "fuel_cell_specific_power_kW_per_kg": dict(
        titolo="Fuel Cell Specific Power",
        xlabel=r"$SP_{fc}$ [kW/kg]",
    ),
    "eta_fuel_cell": dict(
        titolo="Fuel Cell Efficiency",
        xlabel=r"$\eta_{fc}$",
    ),
    "gamma_tank": dict(
        titolo="Tank Gravimetric Efficiency",
        xlabel=r"$\gamma_{tank}$",
    ),
    "hydrogen_empty_weight_multiplier": dict(
        titolo="H2 Empty Weight Multiplier",
        xlabel=r"$M_{OEW, \, H2}$",
    ),    
    "electricity": dict(
        titolo="Electricity Chain Efficiency",
        xlabel=r"$\eta_{elec}$",
    ),   
    "liquid_hydrogen": dict(
        titolo="LH2 Chain Efficiency",
        xlabel=r"$\eta_{LH_2}$",
    ),    
    "e_saf": dict(
        titolo="e-SAF Chain Efficiency",
        xlabel=r"$\eta_{e-SAF}$",
    ),   
    "eta_motor": dict(
        titolo="Electric Motor Efficiency",
        xlabel=r"$\eta_{motor}$",
    ),    
    "eta_p_propeller": dict(
        titolo="Baseline Propeller Propulsive Efficiency",
        xlabel=r"$\eta_{p_{prop}}$",
    ), 
    "eta_p_fan": dict(
        titolo="Baseline Fan Propulsive Efficiency",
        xlabel=r"$\eta_{p_{fan}}$",
    ),
    "oew_fan_b": dict(
        titolo="Fan OEW Fraction coefficient b",
        xlabel=r"$b_{fan}$",
    ), 
    "oew_fan_c": dict(
        titolo="Fan OEW Fraction coefficient c",
        xlabel=r"$c_{fan}$",
    ), 
    "oew_prop_b": dict(
        titolo="Propeller OEW Fraction coefficient b",
        xlabel=r"$b_{prop}$",
    ),
    "oew_prop_c": dict(
        titolo="Propeller OEW Fraction coefficient c",
        xlabel=r"$c_{prop}$",
    ), 
    "oew_fan_r_pivot": dict(
        titolo="Fan OEW Fraction (INTRODUCED) parameter",
        xlabel=r"$r_{fan}$",
    ), 
    "oew_prop_r_pivot": dict(
        titolo="Propeller OEW Fraction (INTRODUCED) parameter",
        xlabel=r"$r_{prop}$",
    ),
    "pax_weight_kg": dict(
        titolo="Passenger weight",
        xlabel=r"$W_{pax}$",
    ),
    "fan_pressure_ratio": dict(
        titolo="Fan Pressure ratio",
        xlabel=r"$\pi_{fan}$",
    ),
    "fan_scaling_mach_ref": dict(
        titolo="Reference Cruise Mach",
        xlabel=r"$M_{ref}$",
    ),
    "propeller_curve_peak_mach": dict(
        titolo="Propeller Peak Mach",
        xlabel=r"$M_{peak}$",
    ),
    "propeller_curve_rise_rate": dict(
        titolo="Propeller Rise Rate factor ",
        xlabel=r"$k_{r}$",
    ),
    "propeller_curve_decay_width": dict(
        titolo="Propeller Decay Width parameter",
        xlabel=r"$\sigma_{d}$",
    ), 

    # esempio di grafico più personalizzato:
    # "oew_fan_b": dict(
    #     titolo="Esponente OEW/MTOW (turbofan)",
    #     xlabel=r"$b_{fan}$ [-]",
    #     bins=60,
    #     colore="#8172B3",
    #     xlim=(0.30, 0.52),
    #     legenda=False,
    #     figsize=(5, 3.5),
    # ),
}

# ===========================================================================
# MOTORE: da qui in giù non serve modificare nulla
# ===========================================================================


def elementi_del_pannello(ax):
    """Curva teorica, nominale e limiti x di un pannello della figura d'insieme"""
    curva, nominale = None, None
    for linea in ax.get_lines():
        x = np.asarray(linea.get_xdata(), float)
        y = np.asarray(linea.get_ydata(), float)
        if linea.get_linestyle() in ("--", "dashed"):
            nominale = float(x[0])
        elif x.size > 2:
            curva = (x, y)

    # se la libreria non disegna l'istogramma in densità la curva è scalata
    # sui conteggi: la riporto in densità dividendo per l'area delle barre
    area = sum(p.get_width() * p.get_height() for p in ax.patches
               if hasattr(p, "get_width"))
    if curva is not None and area > 0 and abs(area - 1.0) > 0.05:
        curva = (curva[0], curva[1] / area)
    return curva, nominale, ax.get_xlim()


def stile_di(nome, categoria):
    st = dict(STILE)
    if st.get("colore") is None:
        st["colore"] = COLORI_CATEGORIA.get(categoria, "gray")
    extra = PERSONALIZZAZIONI.get(nome, {})
    sconosciute = set(extra) - set(STILE)
    if sconosciute:
        print(f"  [{nome}] chiavi ignorate in PERSONALIZZAZIONI: {sorted(sconosciute)}")
    st.update({k: v for k, v in extra.items() if k in STILE})
    return st


def grafico_singolo(nome, valori, categoria, pannello):
    st = stile_di(nome, categoria)
    curva, nominale, xlim = pannello

    fig, ax = plt.subplots(figsize=st["figsize"], constrained_layout=True)
    valori = np.asarray(valori, float)
    valori = valori[np.isfinite(valori)]
    ax.hist(valori, bins=st["bins"], density=True, color=st["colore"],
            alpha=st["alpha_istogramma"], label=st["etichetta_istogramma"])

    if st["mostra_curva"] and curva is not None:
        ax.plot(*curva, color=st["colore_curva"] or st["colore"],
                lw=st["lw_curva"], label=st["etichetta_curva"])
    if st["mostra_nominale"] and nominale is not None and np.isfinite(nominale):
        ax.axvline(nominale, color=st["colore_nominale"], ls="--", lw=1.2,
                   label=st["etichetta_nominale"])

    ax.set_title(st["titolo"] or nome, fontsize=st["fontsize_titolo"], fontweight='bold')
    if st["xlabel"]:
        ax.set_xlabel(st["xlabel"], fontsize=st["fontsize_assi"])
    if st["mostra_asse_y"]:
        ax.set_ylabel(st["ylabel"], fontsize=st["fontsize_assi"])
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=st["fontsize_tick"])
    if st["xlim"] or xlim:
        ax.set_xlim(st["xlim"] or xlim)
    if st["griglia"]:
        ax.grid(alpha=0.3)
    if st["legenda"]:
        ax.legend(fontsize=st["fontsize_legenda"], loc=st["posizione_legenda"],
                  frameon=False)
    return fig


def parametri_scelti(theta, da_cli):
    richiesti = da_cli or PARAMETRI or list(theta.columns)
    ignoti = [p for p in richiesti if p not in theta.columns]
    if ignoti:
        raise SystemExit(f"Parametri inesistenti: {ignoti}\n"
                         f"Quelli validi si vedono con --elenco")
    return richiesti


def leggi_argomenti():
    ap = argparse.ArgumentParser(description="PDF dei parametri, un grafico per parametro")
    ap.add_argument("--parametri", nargs="+", metavar="NOME",
                    help="quali parametri (sostituisce PARAMETRI nelle impostazioni)")
    ap.add_argument("--elenco", action="store_true",
                    help="stampa i nomi dei parametri ed esce")
    return ap.parse_args()


def main():
    args = leggi_argomenti()
    theta, origine = carica_theta()
    if args.elenco:
        print("\n".join(theta.columns))
        return

    scelti = parametri_scelti(theta, args.parametri)
    specs = build_default_specs()
    corr = build_default_correlations(specs)
    print(f"theta: {theta.shape[0]} campioni x {theta.shape[1]} parametri (da {origine})")

    # figura d'insieme solo in memoria: serve a leggere curve e nominali
    fig_insieme, _ = plot_parameter_distributions(
        theta, specs, corr, nominals=nominali_di_calibrazione(theta, specs))
    pannelli = {ax.get_title(): ax for ax in fig_insieme.axes}
    df = parameter_sample_check(theta, specs, corr)
    categorie = dict(zip(df["parametro"], df["categoria"]))

    cartella = OUT_DIR / CARTELLA
    cartella.mkdir(exist_ok=True)
    print(f"  {CARTELLA}/")
    for nome in scelti:
        if nome in pannelli:
            pannello = elementi_del_pannello(pannelli[nome])
        else:
            print(f"    [{nome}] pannello non trovato nella figura d'insieme: "
                  f"lo disegno senza curva teorica né nominale")
            pannello = (None, None, None)
        fig = grafico_singolo(nome, theta[nome], categorie.get(nome), pannello)
        for fmt in FORMATI:
            fig.savefig(cartella / f"{nome}.{fmt}", dpi=DPI)
        plt.close(fig)
        print(f"    {nome}  ({', '.join(FORMATI)})")
    plt.close(fig_insieme)


if __name__ == "__main__":
    main()
