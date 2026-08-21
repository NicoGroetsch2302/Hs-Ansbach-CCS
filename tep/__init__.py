"""Gemeinsamer Unterbau der TEP-Notebooks (Tennessee Eastman Process).

Zwei Notebook-Familien, ein Fundament:

    tep.core     Spaltennamen, Splits, Cutoffs, Vorverarbeitung und die
                 linearalgebraischen Bausteine. Alles, was beide Familien
                 brauchen und was nicht in zwei Fassungen driften darf.

    tep.tsfresh  Fehler-KLASSIFIKATION: TEP-Runs projizieren, TSFresh-
                 Features extrahieren, auswaehlen, Modelle vergleichen,
                 Confusion-Matrizen.

    tep.eigen    Fehler-CHARAKTERISIERUNG: Eigenwert- bzw.
                 Nicht-Gaussianitaets-Spektren pro Run, aggregiert je
                 Fehlerklasse, als Spektren- und Balkenplots. Dazu die
                 Klassifikation auf den exportierten Spektren.

Es gibt keine Konfigurationsobjekte und keine Zustandsklassen: jede
Funktion nimmt entgegen, was sie braucht, und gibt zurueck, was die
naechste braucht. Die Notebooks importieren aus den Unterpaketen:

    from tep.tsfresh import select_features, benchmark_models
    from tep.eigen import load_train, run_spectra, aggregate
"""
