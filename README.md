# Hs-Ansbach-CCS — Fehlerdiagnose auf dem Tennessee Eastman Process

Vergleich linearer Projektionsverfahren (PCA, DyCA, DPCA, CVA, ICA, LDA, DyCVDA)
auf dem TEP-Benchmark: einmal zur **Charakterisierung** der Fehlerklassen über
Eigenwertspektren, einmal zur **Klassifikation** über TSFresh-Features.

Der gesamte Code liegt im Paket [`tep/`](tep); die Notebooks enthalten nur noch
Konfiguration, Aufrufe und die inhaltliche Dokumentation des jeweiligen Versuchs.

## Setup

```bash
pip install -r requirements.txt      # Python 3.12
jupyter lab
```

Die vier TEP-CSVs (`TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`,
`TEP_FaultFree_Testing.csv`, `TEP_Faulty_Testing.csv`, zusammen ~5 GB) sind
**nicht** im Repo. Sie gehören nach `data_csv/` — darauf steht `data_dir` in jeder
Konfigurationszelle. Dorthin schreiben die Eigenwert-Notebooks auch ihre
exportierten Spektren; nur die TSFresh-Chunk-Caches bleiben im Wurzelverzeichnis.

Es muss der Datensatz von **Rieth et al. (2017)** sein
([doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1)): 500 Simulations-
läufe je Fehlerklasse, Spalten `faultNumber`, `simulationRun`, `sample`,
`xmeas_1..41`, `xmv_1..11`. Der ältere Braatz-/Downs-Vogel-Satz (`d00.dat` …
`d21.dat`) passt **nicht** — er hat nur einen Lauf je Fehler, womit die
Statistik über die Läufe und damit die gesamte Auswertung entfällt.

## Notebooks

**Eigenwertspektren** — pro Lauf ein Wertevektor, gemittelt je Fehlerklasse:

| Notebook | Verfahren |
|---|---|
| `PCA_eigenwerte.ipynb`, `DyCA_Eigenwerte.ipynb` | PCA, DyCA |
| `DPCA_Eigenwerte.ipynb`, `CVA_Eigenwerte.ipynb`, `ICA_Eigenwerte.ipynb` | DPCA, CVA, ICA |
| `LDA_Eigenwerte.ipynb` | LDA (paarweiser Laufvergleich, immer `scaler`) |
| `LazyClassifier_PCA_DyCA.ipynb` | Klassifikation **auf** den exportierten Spektren |

`LazyClassifier_PCA_DyCA.ipynb` liest die Train-Spektren aus den CSVs der
Eigenwert-Notebooks — die müssen also vorher gelaufen sein. Die Test-Spektren
rechnet es selbst und cached sie.

**TSFresh-Klassifikation** — Features direkt aus dem projizierten Zeitsignal,
dreiphasig (Train-Extraktion + Selektion, Test-Extraktion, LazyClassifier):

| Notebook | Projektionen |
|---|---|
| `TSFresh_PCA_DyCA.ipynb` | raw, PCA, DyCA |
| `TSFresh_DPCA_CVA_ICA.ipynb` | raw, DPCA, CVA, ICA |
| `TSFresh_DyCVDA.ipynb` | raw, DyCVDA |

`raw` läuft überall als gemeinsamer Anker mit und ist bitidentisch — die drei
Notebooks teilen sich dessen Cache und sind direkt vergleichbar.

## Paketaufbau

```
tep/core.py      Spaltennamen, Splits, Cutoffs, Vorverarbeitung, lineare Algebra
tep/plotting.py  Confusion-Matrizen
tep/eigen/       SpectrumConfig, Spektren-Registry, Aggregation, Plots, Klassifikation
tep/tsfresh/     PipelineConfig, Projektions-Registry, Feature-Cache, Phase A/B/C
```

Beide Familien teilen `tep.core` — `PROC_COLS`, Cutoffs und Skalierung dürfen
nicht in zwei Fassungen driften.

## Zwei Stellschrauben, die man kennen sollte

- **`scaling_mode`** — `"global_mean"` zieht einen skalaren Gesamtmittelwert ab
  (Rohvarianzen bleiben, große Variablen dominieren), `"scaler"` standardisiert
  spaltenweise in Normalbetriebs-Einheiten. Der Modus steckt im Namen der
  Ergebnis-CSV, ein Umschalten überschreibt also nichts.
- **Probeläufe** — `runs_per_fault=3` (Eigenwerte) bzw. `smoke_test=True`
  (TSFresh) machen aus Stunden Minuten, in einem eigenen Cache-Ordner.

`chunk_runs=250` muss bleiben, solange der TSFresh-Cache geteilt wird: die
Chunk-Dateien hängen über ihren Index an dieser Aufteilung.
