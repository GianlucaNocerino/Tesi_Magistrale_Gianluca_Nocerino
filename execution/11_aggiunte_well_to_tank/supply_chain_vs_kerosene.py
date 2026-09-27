"""
Filiere reali contro cherosene: il confronto di energy_mix_effects.py,
ma con le filiere di produzione di supply_chain_maps.py (di default lo
scenario attuale, 2024) al posto del mix elettrico unico.

In energy_mix_effects.py tutti i sistemi sostenibili passano per la
stessa elettricità di rete, quindi hanno lo stesso rendimento di catena
e la stessa intensità di carbonio. Qui ognuno ha la sua filiera:

    batteria            -> rete elettrica      (eta_grid, gco2_grid)
    idrogeno (FC e comb.)-> LH2 da gas naturale (eta_lh2,  gco2_lh2)
    "e-SAF"             -> SAF da HEFA         (eta_saf,  gco2_saf)
    cherosene           -> filiera fossile     (eta_kerosene,
                                                gco2_kerosene_upstream
                                                + combustione)

e le due metriche restano le stesse:

    primaria = energia_al_serbatoio / eta
    co2      = energia_al_serbatoio * g

NON SI RICALCOLA NIENTE
--------------------------------------------
Questo script non modifica gli altri due e non ne duplica la logica: ne
importa le funzioni, esattamente come plot_filiere_singole.py.

  - l'energia al serbatoio dei sistemi sostenibili si ricostruisce dal
    cubo della Fase 7 come in supply_chain_maps.py;
  - quella del cherosene si legge da kerosene_cube.npz, la cache di
    energy_mix_effects.py (stesse righe di theta, stessa griglia: la
    firma coincide e la propagazione non riparte).

STESSI CAMPIONI DELLE MAPPE GIA' FATTE
--------------------------------------------
I sei parametri di filiera sono campionati con la stessa matrice LHS di
supply_chain_maps.py (stesso SEED, stesso numero di righe), e i due del
cherosene con una matrice LHS a parte, affiancata alla prima tramite
l'argomento u_matrix di sample_literature_parameters. Le copule del
cherosene toccano solo le sue due colonne, quindi i sei parametri di
filiera restano identici a quelli che hanno prodotto map_co2_attuale e
map_primary_attuale (lo script lo verifica). Di conseguenza, togliendo
il cherosene, le mappe nuove devono coincidere con quelle vecchie: è il
test di non regressione.

Il legame fra rete e raffinazione (QUOTA_ELETTRICA_RAFFINAZIONE) e
quello fra rendimento ed emissione del cherosene sono gli stessi di
energy_mix_effects.py.

ATTENZIONE SUI CONFINI DEL CONTEGGIO DELLA CO2
--------------------------------------------
Per il cherosene si conta anche la combustione a bordo (73.5 g/MJ), che
è CO2 fossile. Per idrogeno e SAF si usa l'indice di filiera di
SCENARI: il confronto è coerente solo se quell'indice è già un valore
well-to-wake, cioè comprende la combustione e, per il SAF, il credito
della CO2 biogenica. È una questione di come sono costruiti i numeri,
non di codice: va dichiarata in tesi.

Uso:
    python execution/11_aggiunte_well_to_tank/supply_chain_vs_kerosene.py
        -> scenario attuale (2024)
    python execution/11_aggiunte_well_to_tank/supply_chain_vs_kerosene.py 2035
        -> un altro scenario di supply_chain_maps.py (il cherosene resta
           quello di energy_mix_effects.py)

Presuppone execution/07_propagazione/propagation_result.npz e, per non
ripropagare il cherosene, kerosene_cube.npz prodotto da
energy_mix_effects.py con lo stesso N_CAMPIONI.

Output (nella cartella di questo script), con <s> lo scenario:
    supply_chain_vs_kerosene_specs_<s>.csv   le PDF di tutti gli otto parametri
    map_co2_<s>_kerosene.png/.csv            migliore per CO2, cherosene compreso
    map_primary_<s>_kerosene.png/.csv        migliore per energia primaria, idem
    advantage_co2_<s>.png/.csv               P(il migliore sostenibile batte il cherosene)
    advantage_primary_<s>.png/.csv
    supply_chain_vs_kerosene_<s>.csv         tabella per missione
    supply_chain_co2_<s>.png                 barre per missione, CO2
    supply_chain_primary_<s>.png             barre per missione, energia primaria
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))

import energy_mix_effects as eme          # noqa: E402
import supply_chain_maps as scm           # noqa: E402
from energy_mix_effects import (          # noqa: E402
    KEROSENE_LABEL, PropagationResult, CopulaPair, CorrelationModel,
    probability_map, sample_literature_parameters)


# =====================================================================
# 1. CONFIGURAZIONE
# =====================================================================

SCENARIO_DEFAULT = "attuale"   # chiave di scm.SCENARI

N_CAMPIONI = None      # deve essere lo stesso di energy_mix_effects.py,
                       # altrimenti la cache del cherosene non combacia
                       # e la propagazione riparte da zero
N_WORKERS = 1          # usati solo se il cherosene va ripropagato
SEED_KEROSENE = 13     # seed della matrice LHS dei due parametri del
                       # cherosene: lo stesso di energy_mix_effects.py

# Le missioni della tabella e delle barre. None = le stesse di
# energy_mix_effects.py, così i due grafici a barre si possono
# affiancare. Attenzione: quelle missioni erano state scelte per
# diversificare il vincitore rispetto all'electricity intensity; con
# le filiere reali il vincitore può cambiare. Per sceglierne altre,
# stesso formato di eme.MISSIONI:
#     MISSIONI = [("300 nmi, 250 kt (propeller)",
#                  Mission(range_nmi=300.0, cruise_speed_kt=250.0,
#                          propulsor="propeller")), ...]
MISSIONI = None

# Come chiamare il SAF sopra le barre, scenario per scenario: la chiave
# interna resta "e-SAF combustion" (vedi TECNOLOGIA_FILIERA in
# supply_chain_maps.py), ma nel 2024 il combustibile è HEFA
NOME_SAF = {
    "attuale": "SAF (HEFA)",
    "2035": "SAF (HEFA + PtL)",
    "2050": "SAF (PtL + HEFA)",
}


# =====================================================================
# 2. I PARAMETRI: SEI DI FILIERA + DUE DEL CHEROSENE
# =====================================================================

NOMI_KEROSENE = ("eta_kerosene", "gco2_kerosene_upstream")


def build_specs(chiave: str) -> dict:
    """Le otto PDF: prima le sei di filiera, nell'ordine di
    supply_chain_maps.py, poi le due del cherosene, come in
    energy_mix_effects.py. L'ordine conta: è quello delle colonne di
    u_matrix."""
    specs = scm.build_supply_chain_specs(scm.SCENARI[chiave]["filiere"], chiave)
    kero = eme.build_upstream_specs()
    for nome in NOMI_KEROSENE:
        specs[nome] = kero[nome]
    return specs


def build_correlations(chiave: str) -> CorrelationModel:
    """Le copule di supply_chain_maps.py più le due del cherosene.

    Il driver del cherosene è gco2_grid, che nella filiera della rete fa
    già da driver: gli stessi due legami di energy_mix_effects.py, dove
    il ruolo di gco2_grid lo aveva gco2_el (stessi numeri nello scenario
    attuale).
    """
    filiere = scm.build_supply_chain_correlations(chiave)
    kerosene = (
        CopulaPair(a="gco2_grid", b="gco2_kerosene_upstream",
                   rho=eme.QUOTA_ELETTRICA_RAFFINAZIONE,
                   rationale="La raffinazione consuma elettricità di rete"),
        CopulaPair(a="gco2_kerosene_upstream", b="eta_kerosene",
                   rho=eme.RHO_ETA_G,
                   rationale="Cherosene: rendimento ed emissione in verso opposto"),
    )
    return CorrelationModel(copulas=tuple(filiere.copulas) + kerosene)


def campiona(chiave: str, specs: dict, correlazioni: CorrelationModel,
             n_totale: int, n: int) -> pd.DataFrame:
    """Gli otto parametri, con le prime sei colonne LHS identiche a
    quelle di supply_chain_maps.py.

    Si campiona su n_totale righe (tutte quelle della Fase 7) e poi si
    taglia a n: è quello che fa supply_chain_maps.py, e con N_CAMPIONI
    piccolo le righe restano comunque le stesse.
    """
    n_filiera = len(specs) - len(NOMI_KEROSENE)
    u_filiera = qmc.LatinHypercube(d=n_filiera, seed=scm.SEED).random(n=n_totale)
    u_kero = qmc.LatinHypercube(d=len(NOMI_KEROSENE), seed=SEED_KEROSENE).random(n=n_totale)
    u = np.hstack([u_filiera, u_kero])

    theta = sample_literature_parameters(n_totale, specs=specs,
                                         correlations=correlazioni,
                                         u_matrix=u)

    # verifica: le sei colonne di filiera devono essere quelle delle mappe
    # gia' fatte. Se non lo sono, il test di non regressione non ha senso
    specs_filiera = {k: v for k, v in specs.items() if k not in NOMI_KEROSENE}
    riferimento = scm.sample_supply_chain(
        n_totale, specs_filiera, scm.build_supply_chain_correlations(chiave))
    uguali = np.allclose(theta[list(specs_filiera)].to_numpy(),
                         riferimento.to_numpy(), rtol=0, atol=0)
    print(f"  Parametri di filiera identici a supply_chain_maps.py: "
          f"{'OK' if uguali else 'NO'}")
    if not uguali:
        raise RuntimeError("Il campionamento delle filiere non coincide con "
                           "quello di supply_chain_maps.py")

    return theta.iloc[:n].reset_index(drop=True)


# =====================================================================
# 3. DALLA FASE 7 E DALLA CACHE DEL CHEROSENE AI CUBI DELLE METRICHE
# =====================================================================

def energia_serbatoio_sostenibili(result: PropagationResult, n: int) -> np.ndarray:
    """Come scm.energia_al_serbatoio, ma sulle prime n righe e in float32:
    il cubo intero in float64 occuperebbe il doppio senza servire."""
    cubo = np.array(result.intensities[:n], dtype=np.float32)
    theta = result.theta.iloc[:n]
    for j, label in enumerate(result.labels):
        colonna = scm.TECNOLOGIA_WTT[label.rsplit(" (", 1)[0]]
        if colonna is None:          # batteria: è già l'elettricità di rete
            continue
        eta_wtt = scm._colonna(theta, colonna).astype(np.float32)
        cubo[:, j, :, :] *= eta_wtt[:, None, None]
    return cubo


def cubo_metrica(base: np.ndarray, labels: list, theta: pd.DataFrame,
                 metrica: str) -> np.ndarray:
    """Applica a ogni etichetta la sua filiera.

    base contiene, per i sostenibili, l'energia al serbatoio (per la
    batteria l'elettricità di rete), e per il cherosene l'energia di
    combustibile al serbatoio: in tutti i casi la grandezza a cui si
    riferiscono eta e g.
    """
    def colonna(nome):
        return theta[nome].to_numpy(dtype=np.float32)[:, None, None]

    out = np.empty_like(base)
    for j, label in enumerate(labels):
        if label.startswith(KEROSENE_LABEL):
            eta = colonna("eta_kerosene")
            g = colonna("gco2_kerosene_upstream") + np.float32(eme.KEROSENE_GCO2_PER_MJ_BURN)
        else:
            filiera = scm.TECNOLOGIA_FILIERA[label.rsplit(" (", 1)[0]]
            eta = colonna(f"eta_{filiera}")
            g = colonna(f"gco2_{filiera}")

        if metrica == "co2":
            out[:, j, :, :] = base[:, j, :, :] * g
        elif metrica == "primaria":
            out[:, j, :, :] = base[:, j, :, :] / eta
        else:
            raise ValueError(f"metrica sconosciuta: {metrica!r}")
    return out


def check_regressione(res_m: PropagationResult, labels: list, chiave: str,
                      slug: str, n_totale: int) -> None:
    """Senza cherosene, la mappa deve essere quella di supply_chain_maps.py.

    Si ricalcola la mappa di probabilità sui soli sostenibili e la si
    confronta con il CSV già salvato. Ha senso solo usando tutte le
    righe: con N_CAMPIONI ridotto le probabilità sono stimate su un
    campione diverso e il confronto viene saltato.
    """
    csv = scm.OUT_DIR / f"map_{slug}_{chiave}.csv"
    if len(res_m.theta) != n_totale or not csv.exists():
        print(f"  Test di non regressione [{slug}]: saltato "
              f"({'N_CAMPIONI ridotto' if csv.exists() else csv.name + ' assente'})")
        return

    fossile = np.array([l.startswith(KEROSENE_LABEL) for l in labels])
    solo = res_m.intensities[:, ~fossile]
    res_s = PropagationResult(intensities=solo, best_index=eme._best_index(solo),
                              theta=res_m.theta, grid=res_m.grid,
                              labels=[l for l, f in zip(labels, fossile) if not f])
    nuova = probability_map(res_s).to_dataframe()
    vecchia = pd.read_csv(csv)
    scarto = float(np.max(np.abs(nuova["P"].to_numpy() - vecchia["P"].to_numpy())))
    # un campione su n_totale che cambia vincitore vale 1/n_totale: può
    # succedere per arrotondamento float32 contro float64 in un pareggio
    esito = "OK" if scarto <= 1.0 / n_totale + 1e-9 else "FALLITO"
    print(f"  Test di non regressione [{slug}], senza cherosene contro "
          f"{csv.name}: {esito} (scarto massimo su P = {scarto:.4f})")


# =====================================================================
# 4. MAIN
# =====================================================================

def main(argv: list = None) -> None:
    chiave = argv[0] if argv else SCENARIO_DEFAULT
    if chiave not in scm.SCENARI:
        raise SystemExit(f"Scenario sconosciuto: {chiave!r}. "
                         f"Disponibili: {list(scm.SCENARI)}")
    if not scm.scenario_compilato(scm.SCENARI[chiave]):
        raise SystemExit(f"Scenario '{chiave}' non compilato in supply_chain_maps.py")
    anno = scm.SCENARI[chiave]["titolo"]

    # le impostazioni di questo script prevalgono su quelle dei moduli
    # importati, senza modificarne i file
    eme.N_WORKERS = N_WORKERS
    if MISSIONI is not None:
        eme.MISSIONI = MISSIONI
    eme.NOME_BREVE = {**eme.NOME_BREVE,
                      "e-SAF combustion": NOME_SAF.get(chiave, "SAF")}

    result = PropagationResult.load(scm.RESULT_PATH)
    n_totale = result.n_samples
    n = n_totale if N_CAMPIONI is None else min(int(N_CAMPIONI), n_totale)
    print(f"Cubo della Fase 7: {n_totale} campioni, usati qui: {n}")
    print(f"Scenario di filiera: {chiave} ({anno})")

    # --- il cherosene: dalla cache di energy_mix_effects.py ----------
    print("\n" + "=" * 78)
    print("CHEROSENE")
    print("=" * 78)
    kero = eme.cubo_cherosene(result.theta.iloc[:n], result.grid)

    # --- gli otto parametri ------------------------------------------
    specs = build_specs(chiave)
    correlazioni = build_correlations(chiave)
    print("\n" + "=" * 78)
    print(f"PARAMETRI A MONTE - {anno}")
    print("=" * 78)
    theta_up = campiona(chiave, specs, correlazioni, n_totale, n)
    tabella = pd.DataFrame([
        {"scenario": chiave, "parametro": s.name, "nominale": s.nominal,
         "PDF": s.pdf_label, "fonte": s.source, "motivazione": s.rationale}
        for s in specs.values()
    ])
    print(tabella[["parametro", "nominale", "PDF"]].to_string(index=False))
    print(f"(la combustione del cherosene, {eme.KEROSENE_GCO2_PER_MJ_BURN:.1f} g/MJ, "
          f"è deterministica)")
    scm.stampa_correlazioni(theta_up, correlazioni)
    tabella.to_csv(scm.OUT_DIR / f"supply_chain_vs_kerosene_specs_{chiave}.csv",
                   index=False)

    # --- base comune alle due metriche -------------------------------
    base = np.concatenate([energia_serbatoio_sostenibili(result, n),
                           np.asarray(kero, dtype=np.float32)], axis=1)
    labels = list(result.labels) + eme.etichette_cherosene()
    del kero

    metriche = {
        "co2": {
            "slug": "co2",
            "titolo_mappa": f"System Most Likely to minimize CO2 - {anno}",
            "titolo_vantaggio": f"Probability that the best Sustainable system\nemits less CO2 than Kerosene - {anno} Supply Chains",
            "ylabel": "CO2 [g/(pax*nmi)]",
            "titolo_barre": f"CO2 Emissions, {anno} Supply Chains:\nBest Sustainable vs Conventional Kerosene",
        },
        "primaria": {
            "slug": "primary",
            "titolo_mappa": f"System Most Likely to minimize Primary Energy - {anno}",
            "titolo_vantaggio": f"Probability that the best Sustainable system demands\nless Primary Energy than Kerosene - {anno} Supply Chains",
            "ylabel": "Primary Energy [MJ/(pax*nmi)]",
            "titolo_barre": f"Primary Energy Demand, {anno} Supply Chains:\nBest Sustainable vs Conventional Kerosene",
        },
    }

    righe = []
    for metrica, info in metriche.items():
        cubo = cubo_metrica(base, labels, theta_up, metrica)
        res_m = eme.risultato_con_metrica(result, cubo, labels, n)

        check_regressione(res_m, labels, chiave, info["slug"], n_totale)
        eme.disegna_mappa(res_m, info["titolo_mappa"], f"{info['slug']}_{chiave}_kerosene")
        eme.disegna_vantaggio(eme.probabilita_vantaggio(cubo, labels), result.grid,
                              info["titolo_vantaggio"], f"{info['slug']}_{chiave}")
        righe += eme.righe_missioni(cubo, labels, result.grid, metrica)

        del cubo, res_m   # un cubo per volta

    # --- tabella e barre ---------------------------------------------
    df = (pd.DataFrame(righe)
          .sort_values(["ordine", "metrica"], kind="stable")
          .drop(columns="ordine")
          .reset_index(drop=True))
    eme.stampa_missioni(df)
    df.to_csv(scm.OUT_DIR / f"supply_chain_vs_kerosene_{chiave}.csv", index=False)

    for metrica, info in metriche.items():
        eme.plot_missioni(df, metrica, info["ylabel"], info["titolo_barre"],
                          f"supply_chain_{info['slug']}_{chiave}.png")

    print(f"\nTutto salvato in: {scm.OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
