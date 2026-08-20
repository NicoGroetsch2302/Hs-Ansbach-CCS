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

Die Notebooks importieren aus den Unterpaketen:

    from tep.tsfresh import Pipeline, PipelineConfig
    from tep.eigen import SpectrumConfig, run_spectra
"""
