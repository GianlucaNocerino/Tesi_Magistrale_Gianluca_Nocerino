"""
Model-form uncertainty: le varianti concettuali del modello deterministico.

A cosa serve
------------
Fino a qui l'incertezza propagata è quella dei PARAMETRI: il modello è
uno solo e si fanno variare i numeri che ci entrano. Ma una parte
dell'incertezza non sta nei numeri, sta nelle ipotesi: stimare il peso
a vuoto con la regressione del paper invece che con quella di Raymer
non è un parametro diverso, è un modello diverso, e nessuna PDF sui
parametri del primo può rappresentare il secondo.

Questo modulo introduce i rami alternativi. Sono tre, indipendenti fra
loro, e ognuno ha un'opzione di default che riproduce esattamente il
modello base già calibrato: ModelForm() senza argomenti non cambia
nemmeno una cifra decimale rispetto alla versione precedente del
codice, e c'è un test di regressione che lo verifica. Le varianti si
attivano una alla volta o insieme.

I tre rami
----------
oew_model
    "paper"   (default) OEW/MTOW = a*MTOW^b + c, la forma del paper di
              riferimento, con b e c che vengono dalla regressione
              bayesiana sui velivoli storici
    "raymer"  la regressione statistica di Raymer per classi di
              velivoli, We/W0 = A * W0^C con W0 in libbre. Cambia la
              forma funzionale (potenza pura invece di potenza più
              offset) e cambia la fonte dei coefficienti (tabella di
              manuale invece di regressione propria)

aero_model
    "fixed"        (default) L/D è un valore nominale costante, con
                   l'eventuale moltiplicatore per lo stivaggio
                   dell'idrogeno. Vale per ogni missione, anche quelle
                   troppo corte per salire in quota
    "polar_decay"  L/D degrada quando la missione è troppo corta per
                   raggiungere la quota di progetto: si vola in aria
                   più densa, il C_L di crociera si allontana da quello
                   di massima efficienza, e la polare parabolica dà
                   E/E_max = 2*xi/(1 + xi^2) con xi = rho_opt/rho_real

reserve_model
    "paper"  (default) alternato + loiter, con il loiter volato alla
             velocità di crociera. È l'ipotesi accademica: nessuno
             attende alla velocità di crociera, che è la più dispendiosa
    "easa"   i vincoli operativi reali (EASA CAT.OP.MPA.150): 5% di
             rotta in più per le contingenze, alternato e riserva finale di 
             30 minuti (fan) o 45 (elica) volata alla velocità di massima
             autonomia oraria, presa come frazione di quella di crociera

Come viaggia la scelta nel modello
----------------------------------
ModelForm è un campo di TechAssumptions. Sembra un'anomalia (è una
scelta strutturale in mezzo a dei numeri), ma è la soluzione che
richiede zero cambiamenti di firma: ogni funzione del modello riceve
già `tech`, quindi riceve già la forma del modello, e non c'è nessun
argomento in più da far passare attraverso il sizer e i sistemi
propulsivi. Il prezzo è ricordarsi che `tech` non è più solo un
contenitore di parametri.

Attenzione: la matrice theta contiene solo colonne numeriche, quindi
model_form non può essere impostata da theta_row_to_tech_wtt. Si passa
al costruttore, come si fa in model_form_uncertainty.py.

Cosa non cambia
---------------
La calibrazione. Theta_acc è stato ottenuto calibrando il modello base
contro il paper, e i parametri che contiene (peso passeggero, forma
delle curve di efficienza) vengono riusati tali e quali sotto ogni
variante. È un'approssimazione, e va dichiarata: a rigore ogni forma
del modello avrebbe una propria regione accettabile, perché cambiando
il modello dei pesi cambia anche quale peso passeggero riproduce
meglio le curve del paper. Riusare Theta_acc significa assumere che
quello spostamento sia piccolo rispetto all'effetto della variante,
il che è un'ipotesi ragionevole ma non verificata.
"""
from dataclasses import dataclass

__all__ = ["ModelForm", "OEW_MODELS", "AERO_MODELS", "RESERVE_MODELS",
           "MODEL_FORM_PRESETS", "preset"]

OEW_MODELS = ("paper", "raymer")
AERO_MODELS = ("fixed", "polar_decay")
RESERVE_MODELS = ("paper", "easa")


@dataclass(frozen=True)
class ModelForm:
    """Quale ramo concettuale usare per ciascuna delle tre ipotesi.

    Frozen e senza campi numerici: i numeri delle varianti (i
    coefficienti di Raymer, la frazione di contingenza, la frazione di
    velocità per il loiter) stanno in TechAssumptions come tutti gli
    altri, così restano perturbabili dalla sensibilità e campionabili
    dalla propagazione. Qui ci sono solo gli interruttori.

    ModelForm() è il modello base.
    """
    oew_model: str = "paper"
    aero_model: str = "fixed"
    reserve_model: str = "paper"

    def __post_init__(self):
        for campo, valore, ammessi in (
            ("oew_model", self.oew_model, OEW_MODELS),
            ("aero_model", self.aero_model, AERO_MODELS),
            ("reserve_model", self.reserve_model, RESERVE_MODELS),
        ):
            if valore not in ammessi:
                raise ValueError(f"{campo}={valore!r} non ammesso, attesi {ammessi}")

    @property
    def is_baseline(self) -> bool:
        return self == ModelForm()

    @property
    def label(self) -> str:
        """Etichetta breve per grafici e nomi di file"""
        if self.is_baseline:
            return "base"
        parti = []
        if self.oew_model != "paper":
            parti.append(f"pesi:{self.oew_model}")
        if self.aero_model != "fixed":
            parti.append(f"aero:{self.aero_model}")
        if self.reserve_model != "paper":
            parti.append(f"riserve:{self.reserve_model}")
        return " + ".join(parti)

    @property
    def slug(self) -> str:
        """Come label ma utilizzabile come nome di file"""
        return (self.label.replace(" + ", "_").replace(":", "-")
                .replace(" ", ""))

    def describe(self) -> str:
        """Le tre scelte per esteso, da stampare in testa a un run"""
        righe = [
            f"  pesi     : {self.oew_model:12s} "
            f"({'a*MTOW^b + c, regressione bayesiana' if self.oew_model == 'paper' else 'A*W0^C di Raymer, W0 in lb'})",
            f"  aero     : {self.aero_model:12s} "
            f"({'L/D nominale costante' if self.aero_model == 'fixed' else 'decadimento da polare parabolica'})",
            f"  riserve  : {self.reserve_model:12s} "
            f"({'alternato + loiter a velocita di crociera' if self.reserve_model == 'paper' else 'EASA CAT.OP.MPA.150'})",
        ]
        return "\n".join(righe)


# I confronti che ha senso fare in tesi: ogni variante da sola, per
# isolarne l'effetto, e poi tutte insieme, che è lo scenario "modello
# alternativo completo". Confrontare solo base contro tutte-insieme
# direbbe che qualcosa cambia ma non che cosa
MODEL_FORM_PRESETS = {
    "base": ModelForm(),
    "raymer": ModelForm(oew_model="raymer"),
    "polare": ModelForm(aero_model="polar_decay"),
    "easa": ModelForm(reserve_model="easa"),
    "tutte": ModelForm(oew_model="raymer", aero_model="polar_decay",
                       reserve_model="easa"),
}


def preset(nome: str) -> ModelForm:
    """Una delle forme predefinite, per nome"""
    if nome not in MODEL_FORM_PRESETS:
        raise ValueError(f"preset {nome!r} sconosciuto, disponibili: "
                         f"{sorted(MODEL_FORM_PRESETS)}")
    return MODEL_FORM_PRESETS[nome]
