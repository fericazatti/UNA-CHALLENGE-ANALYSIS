import os
import sys
import argparse
from tqdm import tqdm
from bids import BIDSLayout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.stft_gpu import generate_stft_maps_for_subject

def main():
    parser = argparse.ArgumentParser(description="Procesamiento STFT por lotes con GPU")
    parser.add_argument("--n", type=int, default=None, help="Cantidad de sujetos a procesar (por defecto: todos)")
    args = parser.parse_args()

    project_root = "/home/ferna96/Documents/personal/MaestriaFM/GPU/una_challenge_analysis/una-challenge-analysis"
    bids_root = os.path.join(project_root, "data/raw/data/ds005873")

    layout = BIDSLayout(bids_root, validate=False)
    all_subjects = layout.get_subjects()

    subjects_to_process = all_subjects[:args.n] if args.n is not None else all_subjects
    print(f"Se procesarán {len(subjects_to_process)} sujetos con GPU: {subjects_to_process}")

    for subj in tqdm(subjects_to_process, desc="Procesando sujetos (GPU)"):
        generate_stft_maps_for_subject(subject_id=subj, session_id='01', project_root=project_root)

if __name__ == "__main__":
    main()
