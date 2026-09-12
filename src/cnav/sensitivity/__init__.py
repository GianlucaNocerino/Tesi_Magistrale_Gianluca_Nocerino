"""
cnav.sensitivity - strumenti di analisi di sensibilità per il modello
deterministico in cnav.model.

- sensitivity_analysis_tools.py: motore di calcolo generico (indici di
  elasticità via differenze finite) + strumenti esplorativi/di plotting
  (tornado plot, sweep multi-punto, superfici 3D, grafico interattivo).
- local_sensitivity.py: procedura di "Fase 3" (consegna del professore) —
  condizioni operative standard, verifica di robustezza rispetto alla
  dimensione della perturbazione, individuazione dei confini tra
  tecnologie. Costruita sopra sensitivity_analysis_tools.py, non lo
  duplica.
"""
