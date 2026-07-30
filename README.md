# Modello deterministico per comparare sistemi propulsivi carbon-neutral

Modello Python che riproduce la metodologia di:

> Adler, E.J., Martins, J.R.R.A. (2025). *Energy demand comparison for
  carbon-neutral flight*. Progress in Aerospace Sciences, 152, 101051.
> https://doi.org/10.1016/j.paerosci.2024.101051

Dati range, velocità di crociera e un insieme di parametri, il
modello dimensiona un aeromobile (analisi tank-to-wake, in stile Breguet)
per quattro sistemi propulsivi carbon-neutral, batteria elettrica, fuel
cell a idrogeno, combustione a idrogeno, combustione e-SAF, e calcola
l'elettricità, da fonti rinnovabili, necessaria a produrre l'energia richiesta
(analisi well-to-tank), normalizzata per passeggero-miglio nautico (Electricity Intensity)
[MJ/(pax·nmi)].

Questo è il primo passo di una tesi magistrale sulla quantificazione
dell'incertezza applicata a tali sistemi propulsivi: il modello
deterministico è il punto di partenza su cui, in un secondo momento, verrà
costruito uno strato di propagazione dell'incertezza (es. Monte Carlo o
metodi polinomiali del caos sui parametri di `TechAssumptions` e
`WellToTankEfficiencies`) per stimare come cambieranno nel tempo i domini
di "eccellenza" di ciascun sistema.

## Struttura del Modello

```
carbon_neutral_aviation/
├── src/cnav/                      # il pacchetto Python
│   ├── units.py                   # conversioni unità di misura
│   ├── constants.py               # Tutti gli input: parametri tecnologici + coefficienti per fit empirici + WTT
│   ├── atmosphere.py              # velocità del suono
│   ├── propulsive_efficiency.py   # fattore di scala (fan e propeller)
│   ├── mission.py                 # classe Mission (range, velocità, e propeller o fan)
│   ├── T_to_W_definitions.py      # tutte le definizioni per la catena tank-to-wake: fisica condivisa + stime empiriche di peso
│   ├── well_to_tank.py            # catena well-to-tank
│   ├── propulsion_systems.py      # classe generale e una sottoclasse per ogni tipo di sistema propulsivo carbon neutral
│   ├── aircraft_sizer.py          # ciclo iterativo di dimensionamento dell'aeroplano
│   └── energy_intensity.py        # well-to-tank + tank-to-wake -> MJ/(pax*nmi)
├── examples/
│   ├── plot_intensity_vs_range.py   # riproduce la Fig. 4 dell'articolo
│   └── plot_best_system_map.py      # riproduce la Fig. 3 dell'articolo
└── tests/
    └── test_basic.py               # test per il controllo del codice
```

Il pacchetto è organizzato a "strati": `constants`/`units`/`atmosphere` sono
i mattoncini di base; `T_to_W_definitions.py` contiene tutte le formule condivise fra i
4 sistemi propulsivi; `propulsion_systems.py` definisce solo le differenze
fra un sistema e l'altro; `aircraft_sizer.py` e' il ciclo iterativo generico; 
`energy_intensity.py` e' il livello piu' alto. Se in futuro si vuole aggiungere un
quinto sistema propulsivo (es. una configurazione ibrida), basterà scrivere
una nuova sottoclasse di `PropulsionSystem` in `propulsion_systems.py`.

## Uso rapido

```python
from cnav import (Mission, TechAssumptions, WellToTankEfficiencies,
                   build_energy_carriers, build_default_systems, compute_intensity)

tech = TechAssumptions()
wtt = WellToTankEfficiencies()
systems = build_default_systems(build_energy_carriers(wtt))

mission = Mission(range_nmi=500, cruise_speed_kt=450, propulsor="fan")
for system in systems:
    result = compute_intensity(mission, system, tech)
    print(system.name, result.intensity_MJ_per_pax_nmi)
```

## Tutti gli input in un unico posto

`TechAssumptions` (in `constants.py`) raccoglie ogni numero usato dal
modello (nessuna funzione ha valori scritti al suo interno, tutte
li ricevono come argomento). I campi sono organizzati in due gruppi
concettualmente diversi (comodo per la UQ):

1. **Parametri tecnologici** (`eta_motor`, `e_battery_Wh_per_kg`,
   `eta_fuel_cell`, `fan_pressure_ratio`, `pax_weight_kg`, ...): valori che
   rappresentano lo stato dell'arte di una tecnologia e che possono
   migliorare nel tempo.
2. **Coefficienti di correlazioni empiriche** (`oew_fan_a/b/c`,
   `oew_prop_a/b/c`, `payload_a/b/c/d/e`, `battery_oew_fraction`,
   `hydrogen_ld_multiplier`, `propeller_curve_*`): regressioni su dati
   limitati o assunzioni di modellazione. La loro incertezza è
   statistica/di fit, non tecnologica, dunque non migliorano con la ricerca,
   semplicemente non sono note con precisione oggi.

I poteri calorifici (idrogeno, e-SAF) restano invece fuori da
`TechAssumptions`, come costanti fisse in `constants.py`: sono proprietà
chimiche, non hanno incertezza.

**Nota su `fan_pressure_ratio`**: l'articolo di Adler & Martins non
specifica quale rapporto di compressione del fan abbiano usato per
costruire la curva Fig. 9, si dice solo "adottata da Michel [55]", che nel
suo articolo usa 1.5 come valore di esempio ricorrente. Il default qui
riprende quel valore, ma e' una scelta di questo modello, non
un'informazione dell'articolo di riferimento della tesi.

Per gli studi di sensitività/incertezza futuri, `TechAssumptions` e
`WellToTankEfficiencies` sono `dataclass` immutabili con un metodo
`.with_changes(...)` per generare copie con parametri diversi, comodo per
il campionamento (es. Monte Carlo):

```python
tech_perturbato = tech.with_changes(e_battery_Wh_per_kg=450.0)
```

### Installazione

```bash
pip install -r requirements.txt
```

### Eseguire i test

```bash
pytest tests/
```

### Generare i grafici di esempio

```bash
python examples/plot_intensity_vs_range.py
python examples/plot_best_system_map.py
```

## Stato di validazione del modello

Questo è un primo modello funzionante, non ancora una riproduzione
quantitativa validata dell'articolo. In particolare:

1. **Fig. 9 (fattore di scala dell'efficienza propulsiva): risolta per il fan, ancora aperta per l'elica**
   Recuperati e controllati entrambi i riferimenti citati dall'articolo:
   - **Fan**: Michel, U. (2011), "The benefits of variable area fan nozzles
     on turbofan engines", AIAA 2011-226 [55], fornisce un'equazione chiusa
     per l'efficienza propulsiva di un fan senza perdite in funzione del 
     Mach di volo e del rapporto di compressione del fan.
   - **Elica**: Alves, Silvestre, Gamboa (2020), "Aircraft Propellers - Is
     There a Future?", Energies 13(16), 4157 [56], non fornisce
     un'equazione chiusa per l'elica: usa la stessa formula generale 
     del disco attuatore di Michel ma senza i parametri (diametro, carico 
     del disco, RPM) per calcolarla esplicitamente. Nel modello si usa una
     approssimazione ai minimi quadrati della curva corrispondente mostrato
     nell'articolo di riferimento.
2. **Il peso medio per passeggero** 
   (di default 100 kg)  non è specificato nell'articolo fornito ed è un'assunzione da
   verificare.
3. **L'energia di salita** è descritta a parole nell'articolo
   ma senza un'equazione numerata esplicita; l'implementazione è un'interpretazione 
   ragionevole (energia cinetica + potenziale, resistenza aerodinamica trascurata).
4. **Quota di crociera**
   Come quota di crociera, fondamentale per calcolare, passando per il modello atmosferico, il
   Mach, in questo modello si considera non quella teorica (25000 ft per propeller e 35000 ft per fan), ma quella effettiva (che potrebbe essere minore, nel caso la missione sia breve e la salita occupa più di metà della missione totale).


## Riferimenti

Adler, E.J., Martins, J.R.R.A. (2025). Energy demand comparison for
carbon-neutral flight. *Progress in Aerospace Sciences*, 152, 101051.

Michel, U. (2011). The benefits of variable area fan nozzles on turbofan
engines. AIAA 2011-226. (Rif. [55]: fonte della curva Fig. 9 per il fan.)

Alves, P., Silvestre, M., Gamboa, P. (2020). Aircraft Propellers - Is
There a Future? *Energies*, 13(16), 4157. (Rif. [56]: citato per la curva
Fig. 9 dell'elica, ma senza equazione chiusa utilizzabile — vedi sopra.)
