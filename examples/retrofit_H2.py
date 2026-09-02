"""
retrofit_h2.py

Modello semplificato di retrofit a idrogeno liquido (LH2) per un velivolo
tubo-e-ala esistente, basato sulla procedura iterativa descritta nel
paragrafo "6.5.2 Example: Retrofitting an existing aircraft" delle dispense del professor Malpica.

Idea generale
-------------
Si parte da un velivolo di riferimento (qui l'Airbus A350-1000) e si
aggiunge un serbatoio criogenico cilindrico in fusoliera, allungando la
fusoliera stessa. Questo comporta:
  - un aumento della wetted area di fusoliera (piu' drag)
  - un aumento di massa (serbatoio + penalita' strutturale + fusoliera piu'
    pesante), che va ad aggiornare l'OEW
  - un nuovo GTOW = OEW_cryo + payload + massa di idrogeno
  - un nuovo L/D e quindi un nuovo range via equazione di Breguet

Il peso di idrogeno viene trovato per bisezione, imponendo che il range
calcolato sia uguale al range di progetto (target_range_km).

Infine, si cerca di caratterizzare la correlazione tra la tank gravimetric efficiency e
il liquid hydrogen empty weight multiplier
"""

from dataclasses import dataclass
import math


@dataclass
class Aircraft:
    """Dati di un velivolo di riferimento (baseline), es. Tabella 6.2."""
    name: str
    target_range_km: float
    mtow_kg: float
    oew_kg: float
    payload_kg: float
    cruise_speed_ms: float
    rho_cruise: float          # densita' aria a quota di crociera [kg/m3]
    wing_area_m2: float
    cf: float                  # coefficiente di attrito superficiale
    oswald_e: float
    aspect_ratio: float
    fuselage_diameter_m: float
    fuselage_length_m: float
    # opzionale: wetted area totale nota del velivolo baseline [m2]
    # (se disponibile, evita di dover stimare Swet con la regressione log-log)
    aircraft_wetted_area_m2: float = None


class HydrogenRetrofitter:
    """Modello di retrofit LH2, iterativo, basato su Breguet."""

    # --- costanti fisiche / di modello (Tabella 6.2) ---
    LH2_DENSITY = 71.0          # kg/m3
    LH2_LHV = 120e6             # J/kg (potere calorifico inferiore)
    G = 9.80665                 # m/s2
    TANK_GRAV_EFF = 0.5        # efficienza gravimetrica del serbatoio
    PROP_EFFICIENCY = 0.4       # efficienza propulsiva complessiva eta_0
    STRUCTURAL_PENALTY_FRAC = 0.06   # penalita' strutturale = 6% del peso fusoliera base
    FUEL_LOST_FRAC = 0.014      # frazione di GTOW persa in rullaggio/salita/discesa
    FUEL_RESERVE_FRAC = 0.10    # riserva di carburante non bruciata in crociera

    # --- parametri calibrabili (non del tutto certi dal testo scansionato) ---
    tank_margin_factor = 1.10       # margine extra per isolamento/integrazione serbatoio
    wfus_regression_k = 5.0         # Wfus[lb] = k * Swet_fus[ft2]  (eq. 6.8)
    swet_D = 0.8390                 # log10(Swet[ft2]) = D*log10(GTOW[lb]) + C
    swet_C = -0.0876                # usate solo se aircraft_wetted_area_m2 non e' fornito

    M2_TO_FT2 = 10.7639
    KG_TO_LB = 2.20462

    def __init__(self, baseline: Aircraft):
        self.ac = baseline

        # geometria/peso di fusoliera del velivolo baseline
        self.fuselage_wetted_base = self._fuselage_wetted_area(baseline.fuselage_length_m)
        self.fuselage_weight_base = self._fuselage_weight(self.fuselage_wetted_base)

        # wetted area totale baseline: usa il valore noto se fornito,
        # altrimenti stima dalla regressione log10(Swet) vs log10(GTOW)
        if baseline.aircraft_wetted_area_m2 is not None:
            self.aircraft_wetted_base = baseline.aircraft_wetted_area_m2
        else:
            self.aircraft_wetted_base = self._aircraft_wetted_area_from_gtow(baseline.mtow_kg)

    # ------------------------------------------------------------------
    # Geometria del serbatoio e della fusoliera
    # ------------------------------------------------------------------
    def _tank_length(self, m_lh2_kg: float) -> float:
        """Lunghezza del serbatoio (con margine) dato il peso di LH2."""
        r_tank = self.ac.fuselage_diameter_m / 2.0   # d_tank = d_fusoliera (caso limite)
        v_lh2 = m_lh2_kg / self.LH2_DENSITY
        l_tank = v_lh2 / (math.pi * r_tank**2)
        return l_tank * self.tank_margin_factor

    def _fuselage_wetted_area(self, length_m: float) -> float:
        """Wetted area di fusoliera (eq. 6.7), da rapporto di snellezza."""
        d = self.ac.fuselage_diameter_m
        lam = length_m / d  # fineness ratio
        return math.pi * d * length_m * (1 - 2.0/lam)**(2.0/3.0) * (1 + 1.0/lam**2)

    def _fuselage_weight(self, swet_fus_m2: float) -> float:
        """Peso di fusoliera (eq. 6.8), regressione su wetted area."""
        swet_ft2 = swet_fus_m2 * self.M2_TO_FT2
        w_lb = self.wfus_regression_k * swet_ft2
        return w_lb / self.KG_TO_LB

    def _aircraft_wetted_area_from_gtow(self, gtow_kg: float) -> float:
        """Wetted area totale stimata dalla regressione log-log (eq. 6.12)."""
        gtow_lb = gtow_kg * self.KG_TO_LB
        log_swet_ft2 = self.swet_D * math.log10(gtow_lb) + self.swet_C
        swet_ft2 = 10**log_swet_ft2
        return swet_ft2 / self.M2_TO_FT2

    # ------------------------------------------------------------------
    # Aerodinamica
    # ------------------------------------------------------------------
    def _aero(self, swet_total_m2: float, w_cruise_kg: float):
        cd0 = swet_total_m2 / self.ac.wing_area_m2 * self.ac.cf
        cl = (2 * w_cruise_kg * self.G) / (
            self.ac.rho_cruise * self.ac.wing_area_m2 * self.ac.cruise_speed_ms**2
        )
        cd = cd0 + cl**2 / (math.pi * self.ac.oswald_e * self.ac.aspect_ratio)
        return cl / cd  # L/D

    # ------------------------------------------------------------------
    # Un'iterazione completa: dato un peso di idrogeno, calcola il range
    # ------------------------------------------------------------------
    def evaluate(self, m_lh2_kg: float) -> dict:
        # 1-2. geometria serbatoio e nuova fusoliera
        l_tank = self._tank_length(m_lh2_kg)
        fuselage_length_new = self.ac.fuselage_length_m + l_tank
        swet_fus_new = self._fuselage_wetted_area(fuselage_length_new)
        wfus_new = self._fuselage_weight(swet_fus_new)

        # 3. aggiornamento OEW
        w_tank = (1 - self.TANK_GRAV_EFF) / self.TANK_GRAV_EFF * m_lh2_kg
        w_structural = self.STRUCTURAL_PENALTY_FRAC * self.fuselage_weight_base
        oew_cryo = (
            self.ac.oew_kg + w_tank + w_structural
            + (wfus_new - self.fuselage_weight_base)
        )

        # Empty Weight Multiplier: moltiplicatore da applicare all'OEW del
        # velivolo convenzionale per ottenere quello dell'aereo a idrogeno,
        # escludendo il peso del serbatoio (in questo modello semplificato
        # motore e sistemi di thermal management non sono trattati come voci
        # separate, quindi l'unica esclusione applicabile qui e' il serbatoio;
        # se in futuro aggiungi un termine esplicito per motore/TMS, sottraili
        # allo stesso modo da oew_cryo prima di dividere).
        ewm = (oew_cryo - w_tank) / self.ac.oew_kg

        # 4. nuovo GTOW
        gtow = oew_cryo + self.ac.payload_kg + m_lh2_kg

        # aerodinamica: wetted area totale = baseline + delta fusoliera
        swet_total = self.aircraft_wetted_base + (swet_fus_new - self.fuselage_wetted_base)
        w_cruise = gtow - 0.5 * m_lh2_kg  # peso medio in crociera (approssimazione)
        ld = self._aero(swet_total, w_cruise)

        # range di Breguet
        w_fuel_lost = self.FUEL_LOST_FRAC * gtow
        w_initial = gtow - w_fuel_lost
        w_final = gtow - (1 - self.FUEL_RESERVE_FRAC) * m_lh2_kg
        range_m = (self.LH2_LHV / self.G) * self.PROP_EFFICIENCY * ld * math.log(w_initial / w_final)

        return {
            "m_lh2_kg": m_lh2_kg,
            "range_km": range_m / 1000.0,
            "gtow_kg": gtow,
            "oew_kg": oew_cryo,
            "l_over_d": ld,
            "fuselage_length_m": fuselage_length_new,
            "fuselage_weight_kg": wfus_new,
            "aircraft_wetted_area_m2": swet_total,
            "empty_weight_multiplier": ewm,
            "mtow_exceeded": gtow > self.ac.mtow_kg,
        }

    # ------------------------------------------------------------------
    # 5. Iterazione sul peso di idrogeno per centrare il range target
    # ------------------------------------------------------------------
    def solve(self, m_lo=1000.0, m_hi=200000.0, tol_km=0.5, max_iter=200) -> dict:
        """Bisezione su m_lh2 finche' range(m_lh2) == target_range_km."""

        def residual(m):
            return self.evaluate(m)["range_km"] - self.ac.target_range_km

        f_lo, f_hi = residual(m_lo), residual(m_hi)
        if f_lo * f_hi > 0:
            raise ValueError(
                "Il range target non e' compreso nell'intervallo di ricerca "
                f"[{m_lo}, {m_hi}] kg di LH2: allarga i bound di solve()."
            )

        for _ in range(max_iter):
            m_mid = 0.5 * (m_lo + m_hi)
            f_mid = residual(m_mid)
            if abs(f_mid) < tol_km:
                return self.evaluate(m_mid)
            if f_lo * f_mid < 0:
                m_hi, f_hi = m_mid, f_mid
            else:
                m_lo, f_lo = m_mid, f_mid

        return self.evaluate(0.5 * (m_lo + m_hi))

    # ------------------------------------------------------------------
    # Sensibilita' locale (indice di elasticita' S_loc), coerente con la
    # definizione usata nel resto della tesi:
    #     S_loc = (theta / y) * (dy / dtheta)
    # calcolato con differenze finite centrate. Serve a quantificare quanto
    # un output (es. empty_weight_multiplier) risponde a una variazione
    # relativa di un parametro (es. TANK_GRAV_EFF), ed e' il punto di
    # partenza naturale per l'analisi di propagazione dell'incertezza.
    # ------------------------------------------------------------------
    def local_sensitivity(self, param_name: str, output_key: str,
                           rel_perturbations=(0.01, 0.02, 0.05)) -> dict:
        """
        param_name: nome dell'attributo da perturbare (es. 'TANK_GRAV_EFF').
        output_key: chiave del dizionario restituito da solve() da osservare
                    (es. 'empty_weight_multiplier', 'gtow_kg', 'l_over_d').
        rel_perturbations: ampiezze di perturbazione relativa da testare, per
                    verificare che S_loc non dipenda in modo significativo
                    dalla dimensione del passo (richiesto per la Fase 3).
        """
        theta0 = getattr(self, param_name)
        y0 = self.solve()[output_key]

        sensitivities = {}
        for rel in rel_perturbations:
            d_theta = rel * theta0

            setattr(self, param_name, theta0 + d_theta)
            y_plus = self.solve()[output_key]

            setattr(self, param_name, theta0 - d_theta)
            y_minus = self.solve()[output_key]

            dy_dtheta = (y_plus - y_minus) / (2 * d_theta)
            sensitivities[rel] = (theta0 / y0) * dy_dtheta

        setattr(self, param_name, theta0)  # ripristina il valore baseline

        return {
            "param": param_name,
            "output": output_key,
            "baseline_value": theta0,
            "baseline_output": y0,
            "S_loc_by_perturbation": sensitivities,
        }

    # ------------------------------------------------------------------
    # Correlazione e modello lineare surrogato tra un parametro e un output,
    # valutati su un insieme di punti (es. un intervallo di incertezza noto
    # dalla letteratura). Utile per:
    #   - quantificare la correlazione di Pearson (quanto la relazione e'
    #     lineare) sull'intervallo di interesse
    #   - ricavare una formula lineare compatta output = a*param + b, da
    #     usare nella propagazione dell'incertezza senza dover rilanciare
    #     solve() ad ogni campione
    #   - confrontare la pendenza della regressione con quella prevista
    #     dalla sensibilita' locale nel punto centrale dell'intervallo, come
    #     controllo di coerenza tra le due stime
    # ------------------------------------------------------------------
    def linear_surrogate(self, param_name: str, output_key: str,
                          param_values) -> dict:
        theta0 = getattr(self, param_name)  # per ripristino a fine calcolo

        points = []
        for theta in param_values:
            setattr(self, param_name, theta)
            y = self.solve()[output_key]
            points.append((theta, y))
        setattr(self, param_name, theta0)

        n = len(points)
        mx = sum(x for x, _ in points) / n
        my = sum(y for _, y in points) / n
        sxy = sum((x - mx) * (y - my) for x, y in points)
        sxx = sum((x - mx) ** 2 for x, _ in points)
        syy = sum((y - my) ** 2 for _, y in points)

        slope = sxy / sxx
        intercept = my - slope * mx
        pearson = sxy / math.sqrt(sxx * syy) if syy > 0 else float("nan")

        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
        r2 = 1 - ss_res / syy if syy > 0 else float("nan")

        # confronto con la pendenza prevista dalla sensibilita' locale nel
        # punto centrale dell'intervallo campionato
        eta_center = 0.5 * (min(param_values) + max(param_values))
        setattr(self, param_name, eta_center)
        y_center = self.solve()[output_key]
        sens = self.local_sensitivity(param_name, output_key, rel_perturbations=(0.02,))
        s_loc_center = sens["S_loc_by_perturbation"][0.02]
        slope_from_sloc = s_loc_center * y_center / eta_center
        setattr(self, param_name, theta0)

        return {
            "param": param_name,
            "output": output_key,
            "points": points,
            "slope": slope,
            "intercept": intercept,
            "pearson": pearson,
            "r2": r2,
            "slope_from_local_sensitivity": slope_from_sloc,
        }


# ----------------------------------------------------------------------
# Esempio: retrofit dell'A350-1000 (Tabella 6.2)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    a350 = Aircraft(
        name="A350-1000",
        target_range_km=13870.0,
        mtow_kg=316000.0,
        oew_kg=155129.0,
        payload_kg=34770.0,
        cruise_speed_ms=252.1,
        rho_cruise=0.38,
        wing_area_m2=465.0,
        cf=0.003,
        oswald_e=0.85,
        aspect_ratio=9.12,
        fuselage_diameter_m=5.96,
        fuselage_length_m=72.25,
        aircraft_wetted_area_m2=2445.0,  # valore noto da Tabella 6.3 (baseline)
    )

    retrofitter = HydrogenRetrofitter(a350)

    # Se hai i valori esatti di margine/regressione dal tuo testo, sovrascrivili qui, es.:
    # retrofitter.tank_margin_factor = 1.15
    # retrofitter.wfus_regression_k = 5.0

    result = retrofitter.solve()

    print(f"Retrofit LH2 di {a350.name}")
    print("-" * 40)
    print(f"Massa LH2:            {result['m_lh2_kg']:.0f} kg")
    print(f"GTOW:                 {result['gtow_kg']:.0f} kg")
    print(f"OEW:                  {result['oew_kg']:.0f} kg")
    print(f"Range ottenuto:       {result['range_km']:.0f} km")
    print(f"L/D:                  {result['l_over_d']:.2f}")
    print(f"Lunghezza fusoliera:  {result['fuselage_length_m']:.2f} m")
    print(f"Peso fusoliera:       {result['fuselage_weight_kg']:.0f} kg")
    print(f"Wetted area totale:   {result['aircraft_wetted_area_m2']:.0f} m2")
    print(f"Empty Weight Mult.:   {result['empty_weight_multiplier']:.3f}")
    print(f"MTOW superato?        {result['mtow_exceeded']}")

    print("\nConfronto con Tabella 6.3 (valori attesi tra parentesi):")
    print(f"  GTOW  ~270435 kg -> ottenuto {result['gtow_kg']:.0f} kg")
    print(f"  Fuel  ~51920 kg  -> ottenuto {result['m_lh2_kg']:.0f} kg")
    print(f"  L/D   ~16.14     -> ottenuto {result['l_over_d']:.2f}")

    # --- sensibilita' locale di EWM rispetto a TANK_GRAV_EFF ---
    sens = retrofitter.local_sensitivity(
        param_name="TANK_GRAV_EFF",
        output_key="empty_weight_multiplier",
        rel_perturbations=(0.01, 0.02, 0.05),
    )
    print(f"\nSensibilita' locale: S_loc(EWM, eta_grav) attorno a eta={sens['baseline_value']:.2f}")
    for rel, s in sens["S_loc_by_perturbation"].items():
        print(f"  perturbazione {rel*100:.0f}% -> S_loc = {s:.4f}")

    # --- correlazione e modello lineare surrogato su un intervallo mirato ---
    # esempio: intervallo di incertezza realistico per eta_grav [0.47, 0.58]
    fit = retrofitter.linear_surrogate(
        param_name="TANK_GRAV_EFF",
        output_key="empty_weight_multiplier",
        param_values=[0.47, 0.49, 0.51, 0.53, 0.55, 0.58],
    )
    print(f"\nModello lineare surrogato EWM(eta_grav) su [0.47, 0.58]:")
    print(f"  EWM = {fit['slope']:.5f} * eta_grav + {fit['intercept']:.5f}")
    print(f"  Pearson r = {fit['pearson']:.4f}   R^2 = {fit['r2']:.4f}")
    print(f"  Pendenza da regressione:        {fit['slope']:.5f}")
    print(f"  Pendenza da sensibilita' locale: {fit['slope_from_local_sensitivity']:.5f}")