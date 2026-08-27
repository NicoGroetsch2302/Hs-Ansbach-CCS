#!/usr/bin/env bash
# Alle Notebooks der Reihe nach ausfuehren (voller Lauf, alle 500 Runs).
# Reihenfolge ist Pflicht: LazyClassifier liest die Train-Spektren, die die
# Eigenwert-Notebooks exportieren.
#
#   ./run_all.sh            alle
#   ./run_all.sh notebooks/PCA_eigenwerte.ipynb   einzelne
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.

LOGDIR=logs
mkdir -p "$LOGDIR"

NOTEBOOKS=(
    notebooks/PCA_eigenwerte.ipynb            # zuerst: LazyClassifier braucht sie
    notebooks/DyCA_Eigenwerte.ipynb
    notebooks/DPCA_Eigenwerte.ipynb
    notebooks/CVA_Eigenwerte.ipynb
    notebooks/ICA_Eigenwerte.ipynb
    notebooks/LDA_Eigenwerte.ipynb
    notebooks/LazyClassifier_PCA_DyCA.ipynb
    notebooks/TSFresh_PCA_DyCA.ipynb
    notebooks/TSFresh_DPCA_CVA_ICA.ipynb
    notebooks/TSFresh_DyCVDA.ipynb
)
[ $# -gt 0 ] && NOTEBOOKS=("$@")

for nb in "${NOTEBOOKS[@]}"; do
    log="$LOGDIR/$(basename "$nb" .ipynb).log"
    echo "=== $(date +%H:%M:%S)  $nb  -> $log"
    start=$SECONDS
    # timeout=-1: einzelne Zellen laufen Stunden, der Default wuerde sie killen.
    if .venv/bin/jupyter nbconvert --execute --inplace \
            --ExecutePreprocessor.timeout=-1 \
            --to notebook "$nb" > "$log" 2>&1; then
        echo "    ok   nach $(( (SECONDS - start) / 60 )) min"
    else
        echo "    FEHLER nach $(( (SECONDS - start) / 60 )) min - siehe $log"
        tail -20 "$log"
        exit 1
    fi
done
echo "=== $(date +%H:%M:%S)  alle Notebooks durch"
