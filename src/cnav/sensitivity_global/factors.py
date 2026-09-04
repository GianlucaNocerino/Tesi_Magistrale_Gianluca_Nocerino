"""
Lo spazio dei fattori della sensibilità globale.

Perchè non si campionano direttamente i 23 parametri (incerti e di calibrazione)
----------------------------------------
Sia Morris sia Sobol richiedono fattori indipendenti. La matrice theta
prodotta da assemble_theta non lo è, contiene parametri derivati
(funzioni deterministiche di altri), una coppia legata da copula e sei
parametri di calibrazione che arrivano come righe intere di Theta_acc,
congiuntamente dipendenti fra loro.

Gli "antenati" indipendenti però ci sono, ed è su quelli che si lavora:

  - i quantili uniformi u_i dei parametri di letteratura non derivati
    (14 fattori). Sono indipendenti per costruzione: copule e relazioni
    deterministiche agiscono dopo, dentro la mappa u -> theta, e quindi
    fanno parte del modello f, non degli input
  - un unico fattore di gruppo per il blocco di calibrazione: il
    quantile con cui si sceglie quale riga di Theta_acc usare. I sei
    parametri di calibrazione non hanno marginali separate, hanno una
    nuvola empirica di righe accettate, separarli richiederebbe di
    inventare una struttura di dipendenza che la calibrazione non ha
    prodotto. Il loro indice di gruppo risponde a una domanda comunque
    interessante per la tesi: quanto pesa l'incertezza residua di
    calibrazione rispetto a tutta quella di letteratura

In totale 15 fattori, tutti in [0, 1]. Lavorare sulla scala dei
quantili ha anche un effetto collaterale utile: gli effetti elementari
di Morris risultano adimensionali e confrontabili fra parametri con
unità di misura diverse, senza doverli normalizzare a mano.

NOTA SULL'INTERPRETAZIONE DELLA COPPIA CON COPULA
-------------------------------------------------
liquid_hydrogen ed e_saf condividono il fattore "efficienza di
elettrolisi". Nella parametrizzazione attuale la copula gaussiana
scrive z_b = rho*z_a + sqrt(1-rho^2)*z_b, quindi il fattore
u_liquid_hydrogen porta con se anche la parte comune, mentre
u_e_saf rappresenta solo il residuo idiosincratico dell'e-SAF. Gli
indici vanno letti (e scritti in tesi) con questi nomi, non come se
fossero due parametri simmetrici
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..uncertainty.distributions import (
    build_default_correlations,
    build_default_specs,
    independent_parameter_names,
    sample_literature_parameters,
)

__all__ = ["CALIBRATION_FACTOR", "FactorSpace"]

# nome del fattore di gruppo che rappresenta l'intero blocco di calibrazione
CALIBRATION_FACTOR = "calibration_block"


@dataclass
class FactorSpace:
    """La mappa (ipercubo unitario) -> (matrice theta).

    names: i nomi dei fattori, nell'ordine delle colonne di U. Gli
    ultimi sono CALIBRATION_FACTOR se theta_acc è stata fornita.

    Uso tipico:

        space = FactorSpace.default(theta_acc)
        U = <matrice (n, space.k) di quantili in [0,1]>
        theta = space.theta_from_unit(U)

    theta esce con le stesse colonne e la stessa semantica di
    assemble_theta: è direttamente utilizzabile da evaluate_sample o
    da theta_row_to_tech_wtt
    """
    specs: dict
    correlations: object
    theta_acc: Optional[pd.DataFrame]
    literature_names: list

    @classmethod
    def default(cls, theta_acc: Optional[pd.DataFrame] = None,
                specs: Optional[dict] = None,
                correlations=None) -> "FactorSpace":
        specs = specs or build_default_specs()
        correlations = correlations if correlations is not None else build_default_correlations(specs)
        return cls(specs=specs, correlations=correlations, theta_acc=theta_acc,
                   literature_names=independent_parameter_names(specs, correlations))

    @property
    def names(self) -> list:
        names = list(self.literature_names)
        if self.theta_acc is not None:
            names.append(CALIBRATION_FACTOR)
        return names

    @property
    def k(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(
                f"Fattore sconosciuto: {name!r}. I fattori disponibili sono "
                f"{self.names}") from None

    def theta_from_unit(self, U: np.ndarray) -> pd.DataFrame:
        """Da una matrice (n, k) di quantili alla matrice theta (n, 23).

        I quantili vengono ritagliati dentro (0, 1) prima della
        trasformata inversa: le PDF troncate hanno ppf infinita agli
        estremi esatti, e un disegno di Morris finisce esattamente sugli
        estremi molto più spesso di un LHS
        """
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if U.shape[1] != self.k:
            raise ValueError(f"U ha {U.shape[1]} colonne, attese {self.k} "
                             f"(fattori: {self.names})")
        if U.min() < 0.0 or U.max() > 1.0:
            raise ValueError("U deve stare in [0, 1]: è una matrice di quantili")
        U = np.clip(U, 1e-9, 1.0 - 1e-9)

        n = U.shape[0]
        n_lit = len(self.literature_names)
        lit = sample_literature_parameters(n, specs=self.specs,
                                           correlations=self.correlations,
                                           u_matrix=U[:, :n_lit])
        if self.theta_acc is None:
            return lit

        n_rows = len(self.theta_acc)
        idx = np.minimum((U[:, n_lit] * n_rows).astype(int), n_rows - 1)
        cal = self.theta_acc.iloc[idx].reset_index(drop=True)

        theta = pd.concat([lit, cal], axis=1)
        theta.index.name = "sample_idx"
        return theta
