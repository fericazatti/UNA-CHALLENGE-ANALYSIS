import os
import sys
import argparse
from tqdm import tqdm
from bids import BIDSLayout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.stft import generate_stft_maps_for_subject

def main():
    # --- Argumentos por línea de comandos ---
    parser = argparse.ArgumentParser(description="Procesamiento STFT por lotes")
    parser.add_argument(
        "--n", type=int, default=None,
        help="Cantidad de sujetos a procesar (por defecto: todos)"
    )
    args = parser.parse_args()

    # --- Paths ---
    project_root = "/home/ferna96/Documents/personal/MaestriaFM/GPU/una_challenge_analysis/una-challenge-analysis"
    bids_root = os.path.join(project_root, "data/raw/data/ds005873")

    # --- Detectar sujetos disponibles ---
    layout = BIDSLayout(bids_root, validate=False)
    all_subjects = layout.get_subjects()

    # --- Seleccionar cuántos sujetos procesar ---
    if args.n is not None:
        subjects_to_process = all_subjects[:args.n]
    else:
        subjects_to_process = all_subjects

    print(f"Se procesarán {len(subjects_to_process)} sujetos: {subjects_to_process}")

    # --- Procesamiento con barra de progreso ---
    for subj in tqdm(subjects_to_process, desc="Procesando sujetos"):
        generate_stft_maps_for_subject(
            subject_id=subj,
            session_id='01',
            project_root=project_root
        )

if __name__ == "__main__":
    main()

