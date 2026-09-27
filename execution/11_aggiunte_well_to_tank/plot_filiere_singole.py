"""
Le PDF dei parametri di filiera (well-to-tank), un grafico per parametro
e per scenario.

Affianca supply_chain_maps.py senza modificarlo: ne riusa le terne degli
scenari, le specs, le copule e il campionamento, quindi i campioni
disegnati qui sono gli stessi che hanno prodotto le mappe (stesso SEED,
stesso numero di campioni della propagazione). Cosa stampare e come
(scenari, parametri, titoli, etichette, colori, dimensioni, formati) si
decide nella sezione IMPOSTAZIONI qui sotto: è l'unica parte del file da
toccare.

Uso:
    python execution/11_aggiunte_well_to_tank/plot_filiere_singole.py
    python execution/11_aggiunte_well_to_tank/plot_filiere_singole.py --scenari 2035 2050
    python execution/11_aggiunte_well_to_tank/plot_filiere_singole.py --parametri eta_grid gco2_grid
    python execution/11_aggiunte_well_to_tank/plot_filiere_singole.py --elenco

Produce:
    pdf_singole_filiere/<scenario>/<parametro>.<formato>

Come in 05_pdf_parametri/plot_pdf_singole.py, curva teorica e valore
nominale non vengono ricalcolati: si leggono dal pannello corrispondente
della figura di plot_parameter_distributions (costruita solo in memoria,
non salvata). L'istogramma invece si ridisegna dal campione, per poterne
cambiare bin e colori.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from supply_chain_maps import (  # noqa: E402
    OUT_DIR,
    RESULT_PATH,
    SCENARI,
    PropagationResult,
    build_supply_chain_correlations,
    build_supply_chain_specs,
    plot_parameter_distributions,
    sample_supply_chain,
    scenario_compilato,
)


# ===========================================================================
# IMPOSTAZIONI
# Gerarchia (vince l'ultima):
#   STILE -> COLORI_FILIERA -> PERSONALIZZAZIONI[parametro]
#         -> PERSONALIZZAZIONI_SCENARIO[scenario][parametro]
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Cosa stampare
# ---------------------------------------------------------------------------
# None = tutto. Altrimenti liste di nomi, ad es.
#   SCENARI_DA_STAMPARE = ["2035", "2050"]
#   PARAMETRI = ["eta_grid", "gco2_grid"]
# Da riga di comando --scenari e --parametri hanno la precedenza.
# Per vedere i nomi validi:
#   python .../plot_filiere_singole.py --elenco
SCENARI_DA_STAMPARE = None
PARAMETRI = None

# ---------------------------------------------------------------------------
# 2. Campione
# ---------------------------------------------------------------------------
# None = quanti campioni ha la propagazione della Fase 7, cioè esattamente
# il campione usato per le mappe. Il numero serve solo se il file della
# propagazione non c'è.
N_CAMPIONI_SE_MANCA_PROPAGAZIONE = 1500

# ---------------------------------------------------------------------------
# 3. Dove e in che formato salvare
# ---------------------------------------------------------------------------
CARTELLA = "pdf_singole_filiere"  # sottocartella di 11_aggiunte_well_to_tank
FORMATI = ("png",)          # "pdf"/"svg" restano vettoriali per LaTeX
DPI = 300                         # usato solo dai formati raster

# Stessi limiti sull'asse x per lo stesso parametro in tutti gli scenari
# stampati: utile per confrontare le figure affiancate (la terna del 2050
# è molto più larga di quella attuale). Un xlim esplicito in
# PERSONALIZZAZIONI vince comunque.
XLIM_COMUNE_FRA_SCENARI = False

# ---------------------------------------------------------------------------
# 4. Stile di default, valido per tutti i grafici
# ---------------------------------------------------------------------------
# Nei testi (titolo, xlabel, ylabel, etichette) si può scrivere {anno}:
# viene sostituito con il titolo dello scenario (2024, 2035, 2050).
STILE = dict(
    figsize=(6.0, 4.0),           # pollici; (3.3, 2.5) per mezza colonna
    bins=40,
    alpha_istogramma=0.35,
    colore="#4C72B0",
    colore_curva=None,            # None = stesso colore dell'istogramma
    lw_curva=2.0,
    mostra_curva=True,            # PDF dichiarata
    mostra_nominale=True,         # linea tratteggiata della moda
    colore_nominale="black",
    titolo=None,                  # None = "<parametro> - {anno}"
    xlabel=None,                  # None = nessuna etichetta
    ylabel="Probability Density",
    mostra_asse_y=False,
    xlim=None,                    # None = gli stessi limiti della figura d'insieme
    griglia=False,
    legenda=True,
    posizione_legenda="best",     # "upper left", "upper right", ...
    fontsize_titolo=16,
    fontsize_assi=16,
    fontsize_tick=10,
    fontsize_legenda=12,
    etichetta_istogramma="Sample",
    etichetta_curva="Defined PDF",
    etichetta_nominale="Mode",
)

# Colore per filiera (il suffisso del parametro: grid, lh2, saf).
# Lasciare il dizionario vuoto per usare ovunque STILE["colore"].
COLORI_FILIERA = {
    # "grid": "#4C72B0",
    # "lh2":  "#55A868",
    # "saf":  "#DD8452",
}

# ---------------------------------------------------------------------------
# 5. Personalizzazioni per parametro (valgono in tutti gli scenari)
# ---------------------------------------------------------------------------
# Qualsiasi chiave di STILE può essere ridefinita. Sintassi LaTeX ammessa.
PERSONALIZZAZIONI = {
    "eta_grid": dict(
        titolo="Chain Efficiency, Electric Grid - {anno}",
        xlabel=r"$\eta_{grid}$",
    ),
    "gco2_grid": dict(
        titolo="Carbon Intensity, Electric Grid - {anno}",
        xlabel=r"$g_{CO_2,\,grid}$ [g/MJ$_{el}$]",
    ),
    "eta_lh2": dict(
        titolo="Chain Efficiency, Liquid Hydrogen - {anno}",
        xlabel=r"$\eta_{LH_2}$",
    ),
    "gco2_lh2": dict(
        titolo="Carbon Intensity, Liquid Hydrogen - {anno}",
        xlabel=r"$g_{CO_2,\,LH_2}$ [g/MJ]",
    ),
    "eta_saf": dict(
        titolo="Chain Efficiency, SAF - {anno}",
        xlabel=r"$\eta_{SAF}$",
    ),
    "gco2_saf": dict(
        titolo="Carbon Intensity, SAF - {anno}",
        xlabel=r"$g_{CO_2,\,SAF}$ [g/MJ]",
    ),
}

# ---------------------------------------------------------------------------
# 6. Personalizzazioni per un solo scenario
# ---------------------------------------------------------------------------
# Per ritoccare un grafico specifico senza toccare gli altri anni.
PERSONALIZZAZIONI_SCENARIO = {
    # "2050": {
    #     "eta_grid": dict(xlim=(0.65, 1.25), posizione_legenda="upper right"),
    # },
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


def stile_di(nome, scenario):
    st = dict(STILE)
    filiera = nome.rsplit("_", 1)[-1]
    if filiera in COLORI_FILIERA:
        st["colore"] = COLORI_FILIERA[filiera]
    for livello, extra in (("PERSONALIZZAZIONI", PERSONALIZZAZIONI.get(nome, {})),
                           (f"PERSONALIZZAZIONI_SCENARIO['{scenario}']",
                            PERSONALIZZAZIONI_SCENARIO.get(scenario, {}).get(nome, {}))):
        sconosciute = set(extra) - set(STILE)
        if sconosciute:
            print(f"    [{nome}] chiavi ignorate in {livello}: {sorted(sconosciute)}")
        st.update({k: v for k, v in extra.items() if k in STILE})
    return st


def _testo(t, anno):
    return t.replace("{anno}", str(anno)) if isinstance(t, str) else t


def grafico_singolo(nome, valori, st, anno, pannello, xlim_comune):
    curva, nominale, xlim = pannello

    fig, ax = plt.subplots(figsize=st["figsize"], constrained_layout=True)
    valori = np.asarray(valori, float)
    valori = valori[np.isfinite(valori)]
    ax.hist(valori, bins=st["bins"], density=True, color=st["colore"],
            alpha=st["alpha_istogramma"],
            label=_testo(st["etichetta_istogramma"], anno))

    if st["mostra_curva"] and curva is not None:
        ax.plot(*curva, color=st["colore_curva"] or st["colore"],
                lw=st["lw_curva"], label=_testo(st["etichetta_curva"], anno))
    if st["mostra_nominale"] and nominale is not None and np.isfinite(nominale):
        ax.axvline(nominale, color=st["colore_nominale"], ls="--", lw=1.2,
                   label=_testo(st["etichetta_nominale"], anno))

    ax.set_title(_testo(st["titolo"], anno) or f"{nome} - {anno}",
                 fontsize=st["fontsize_titolo"], fontweight='bold')
    if st["xlabel"]:
        ax.set_xlabel(_testo(st["xlabel"], anno), fontsize=st["fontsize_assi"])
    if st["mostra_asse_y"]:
        ax.set_ylabel(_testo(st["ylabel"], anno), fontsize=st["fontsize_assi"])
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=st["fontsize_tick"])
    limiti = st["xlim"] or xlim_comune or xlim
    if limiti:
        ax.set_xlim(limiti)
    if st["griglia"]:
        ax.grid(alpha=0.3)
    if st["legenda"]:
        ax.legend(fontsize=st["fontsize_legenda"], loc=st["posizione_legenda"],
                  frameon=False)
    return fig


def numero_campioni():
    if RESULT_PATH.exists():
        return PropagationResult.load(RESULT_PATH).n_samples, "propagazione"
    n = int(N_CAMPIONI_SE_MANCA_PROPAGAZIONE)
    print(f"[!] {RESULT_PATH} non trovato: uso {n} campioni, che NON coincidono "
          f"con quelli delle mappe se la propagazione ne ha un numero diverso")
    return n, "N_CAMPIONI_SE_MANCA_PROPAGAZIONE"


def campiona_scenario(chiave, n):
    """Stesse specs, copule e seed di supply_chain_maps.esegui_scenario"""
    specs = build_supply_chain_specs(SCENARI[chiave]["filiere"], chiave)
    correlazioni = build_supply_chain_correlations(chiave)
    return sample_supply_chain(n, specs, correlazioni), specs


def scegli(richiesti, validi, cosa):
    ignoti = [x for x in richiesti if x not in validi]
    if ignoti:
        raise SystemExit(f"{cosa} inesistenti: {ignoti}. Validi: {list(validi)}\n"
                         f"(si vedono anche con --elenco)")
    return list(richiesti)


def leggi_argomenti():
    ap = argparse.ArgumentParser(
        description="PDF dei parametri di filiera, un grafico per parametro e scenario")
    ap.add_argument("--scenari", nargs="+", metavar="NOME",
                    help="quali scenari (sostituisce SCENARI_DA_STAMPARE)")
    ap.add_argument("--parametri", nargs="+", metavar="NOME",
                    help="quali parametri (sostituisce PARAMETRI)")
    ap.add_argument("--elenco", action="store_true",
                    help="stampa scenari e parametri disponibili ed esce")
    return ap.parse_args()


def main():
    args = leggi_argomenti()

    compilati = [s for s in SCENARI if scenario_compilato(SCENARI[s])]
    if args.elenco:
        print("scenari:  ", ", ".join(compilati))
        if compilati:
            specs = build_supply_chain_specs(SCENARI[compilati[0]]["filiere"], compilati[0])
            print("parametri:", ", ".join(specs))
        return

    scenari = scegli(args.scenari or SCENARI_DA_STAMPARE or compilati, SCENARI, "Scenari")
    saltati = [s for s in scenari if s not in compilati]
    for s in saltati:
        print(f"[!] Scenario '{s}' saltato: ci sono terne ancora da compilare")
    scenari = [s for s in scenari if s in compilati]
    if not scenari:
        raise SystemExit("Nessuno scenario da stampare.")

    n, origine = numero_campioni()
    print(f"{n} campioni per scenario (da {origine})")

    # primo passaggio: campioni e pannelli di ogni scenario
    dati = {}
    for chiave in scenari:
        theta_sc, specs = campiona_scenario(chiave, n)
        fig, _ = plot_parameter_distributions(theta_sc, specs, n_cols=3,
                                              title=SCENARI[chiave]["titolo"])
        pannelli = {ax.get_title(): elementi_del_pannello(ax) for ax in fig.axes}
        plt.close(fig)
        dati[chiave] = (theta_sc, pannelli)

    parametri = scegli(args.parametri or PARAMETRI or list(dati[scenari[0]][0].columns),
                       dati[scenari[0]][0].columns, "Parametri")

    xlim_comuni = {}
    if XLIM_COMUNE_FRA_SCENARI:
        for nome in parametri:
            limiti = [dati[s][1][nome][2] for s in scenari if nome in dati[s][1]]
            if limiti:
                xlim_comuni[nome] = (min(l[0] for l in limiti), max(l[1] for l in limiti))

    # secondo passaggio: i grafici
    for chiave in scenari:
        theta_sc, pannelli = dati[chiave]
        anno = SCENARI[chiave]["titolo"]
        cartella = OUT_DIR / CARTELLA / chiave
        cartella.mkdir(parents=True, exist_ok=True)
        print(f"  {CARTELLA}/{chiave}/")
        for nome in parametri:
            if nome in pannelli:
                pannello = pannelli[nome]
            else:
                print(f"    [{nome}] pannello non trovato nella figura d'insieme: "
                      f"lo disegno senza curva teorica né nominale")
                pannello = (None, None, None)
            st = stile_di(nome, chiave)
            fig = grafico_singolo(nome, theta_sc[nome], st, anno, pannello,
                                  xlim_comuni.get(nome))
            for fmt in FORMATI:
                fig.savefig(cartella / f"{nome}.{fmt}", dpi=DPI)
            plt.close(fig)
            print(f"    {nome}  ({', '.join(FORMATI)})")


if __name__ == "__main__":
    main()
