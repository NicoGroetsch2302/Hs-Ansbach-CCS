# Hs-Ansbach-CCS — Fehlerdiagnose auf dem Tennessee Eastman Process

Vergleich linearer Projektionsverfahren (PCA, DyCA, DPCA, CVA, ICA, LDA, DyCVDA)
auf dem TEP-Benchmark: einmal zur **Charakterisierung** der Fehlerklassen über
Eigenwertspektren, einmal zur **Klassifikation** über TSFresh-Features.

Der gesamte Code liegt im Paket [`tep/`](tep). Einstiegspunkt ist
[`main.py`](main.py), gesteuert über [`params.yaml`](params.yaml); die
[`notebooks/`](notebooks) sind die ältere, interaktive Fassung derselben Läufe.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Python 3.12
```

## Läufe starten

```bash
python main.py                    # alle Stufen aus params.yaml
python main.py eigen tsfresh      # nur diese Stufen
python main.py -p probe.yaml      # andere Parameterdatei
```

| Stufe | was sie tut |
|---|---|
| `eigen` | Spektrum je (Fault, Run), Aggregation, vier Standardplots je Verfahren |
| `amplitudes` | niedrigdimensionale Amplituden y(t) je Lauf als NPZ |
| `classify` | Klassifikation **auf** den Spektren + Confusion-Matrizen |
| `tsfresh` | die drei Schritte, Konfigurationsvergleich, Confusion-Matrizen |

Bilder landen in `plots/`. Zwischenergebnisse werden nicht neu gerechnet, wenn
sie schon auf Platte liegen (Spektren-CSVs, NPZ, TSFresh-Chunks, `summary`- und
Vorhersage-CSV) — neu rechnen heißt: die betreffende Datei löschen.

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

## notebooks/ — die interaktive Fassung

Dieselben Läufe zum Durchklicken. `main.py` deckt sie ab; sie bleiben für explorative
Arbeit und die inhaltliche Dokumentation der Versuche.

**Eigenwertspektren** — pro Lauf ein Wertevektor, gemittelt je Fehlerklasse:

| Notebook | Verfahren |
|---|---|
| `notebooks/PCA_eigenwerte.ipynb`, `notebooks/DyCA_Eigenwerte.ipynb` | PCA, DyCA |
| `notebooks/DPCA_Eigenwerte.ipynb`, `notebooks/CVA_Eigenwerte.ipynb`, `notebooks/ICA_Eigenwerte.ipynb` | DPCA, CVA, ICA |
| `notebooks/LDA_Eigenwerte.ipynb` | LDA (paarweiser Laufvergleich, immer `scaler`) |
| `notebooks/LazyClassifier_PCA_DyCA.ipynb` | Klassifikation **auf** den exportierten Spektren |

`notebooks/LazyClassifier_PCA_DyCA.ipynb` liest die Train-Spektren aus den CSVs der
Eigenwert-Notebooks — die müssen also vorher gelaufen sein. Die Test-Spektren
rechnet es selbst und cached sie.

**TSFresh-Klassifikation** — Features direkt aus dem projizierten Zeitsignal,
dreiphasig (Train-Extraktion + Selektion, Test-Extraktion, LazyClassifier):

| Notebook | Projektionen |
|---|---|
| `notebooks/TSFresh_PCA_DyCA.ipynb` | raw, PCA, DyCA |
| `notebooks/TSFresh_DPCA_CVA_ICA.ipynb` | raw, DPCA, CVA, ICA |
| `notebooks/TSFresh_DyCVDA.ipynb` | raw, DyCVDA |

`raw` läuft überall als gemeinsamer Anker mit und ist bitidentisch — die drei
Notebooks teilen sich dessen Cache und sind direkt vergleichbar.

## Paketaufbau

```
tep/core.py      Spaltennamen, Splits, Cutoffs, Vorverarbeitung, lineare Algebra
tep/plotting.py  Confusion-Matrizen
tep/eigen/       Spektren-Registry, Aggregation, Plots, Klassifikation
tep/tsfresh/     Projektions-Registry, Feature-Cache, die drei Schritte
```

Nur Funktionen — keine Konfigurationsobjekte, keine Zustandsklassen. Jede
Funktion nimmt als Argumente entgegen, was sie braucht, und gibt zurück, was
die nächste braucht; die Einstellungen stehen als Konstanten oben im Notebook.
Die beiden Registries (`SPECTRA`, `PROJECTORS`) sind gewöhnliche dicts, ein
eigenes Verfahren ist ein weiterer Eintrag darin.

Beide Familien teilen `tep.core` — `PROC_COLS`, Cutoffs und Skalierung dürfen
nicht in zwei Fassungen driften. `python test_tep.py` prüft die geteilten
Bausteine und beide Registries.

## Zwei Stellschrauben, die man kennen sollte

- **`SCALING_MODE`** — `"global_mean"` zieht einen skalaren Gesamtmittelwert ab
  (Rohvarianzen bleiben, große Variablen dominieren), `"scaler"` standardisiert
  spaltenweise in Normalbetriebs-Einheiten. Der Modus steckt im Namen der
  Ergebnis-CSV und im Cache-Ordner, ein Umschalten überschreibt also nichts.
- **Probeläufe** — `RUNS_PER_FAULT=3` (Eigenwerte) bzw. `SMOKE_TEST=True`
  (TSFresh) machen aus Stunden Minuten, in einem eigenen Cache-Ordner.

`CHUNK_RUNS=250` muss bleiben, solange der TSFresh-Cache geteilt wird: die
Chunk-Dateien hängen über ihren Index an dieser Aufteilung.
