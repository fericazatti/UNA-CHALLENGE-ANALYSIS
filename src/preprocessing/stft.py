import os
import numpy as np
import mne
import pandas as pd
import time
from tqdm import tqdm
from scipy.signal import stft
from bids import BIDSLayout


# Silenciar salidas verbosas de MNE
mne.set_log_level('WARNING')

def generate_stft_maps_for_subject(subject_id: str, session_id: str, project_root: str,
                                    task: str = "szMonitoring",
                                    window_sec: int = 5,
                                    overlap: float = 0.5,
                                    fs: int = 256,
                                    nperseg: int = 128,
                                    noverlap: int = 64):
    """
    Procesa todos los runs EEG de un sujeto/sesión (BIDS),
    calcula mapas STFT por ventana, y guarda tiempos en CSV (una fila por run).
    """
    # --- Rutas base ---
    bids_root = os.path.join(project_root, "data/raw/data/ds005873")
    output_root = os.path.join(project_root, f"data/processed/stft_maps/sub-{subject_id}")
    os.makedirs(output_root, exist_ok=True)

    # --- Layout BIDS ---
    layout = BIDSLayout(bids_root, validate=False)
    eeg_files = layout.get(
        subject=subject_id,
        session=session_id,
        task=task,
        suffix='eeg',
        extension='edf',
        return_type='filename'
    )

    if not eeg_files:
        print(f"[{subject_id}] No se encontraron archivos EEG para ses-{session_id}")
        return

    print(f"\n[{subject_id}] Procesando {len(eeg_files)} run(s):")

    for edf_path in tqdm(sorted(eeg_files), desc=f"Procesamiento del sujeto {subject_id}"):
        basename = os.path.basename(edf_path)
        run_id = basename.split("_")[3]  # ej: run-01

        run_start = time.perf_counter()

        run_output_dir = os.path.join(output_root, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        # Preprocesamiento
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.filter(l_freq=0.5, h_freq=120.0)
        raw.notch_filter(freqs=50)

        duration = raw.n_times / raw.info['sfreq']
        duration_minutes = round(duration / 60, 2)

        # Segmentación
        step = window_sec * (1 - overlap)
        onset_times = np.arange(0, duration - window_sec, step)
        onset_samples = (onset_times * fs).astype(int)
        events = np.column_stack((onset_samples, np.zeros_like(onset_samples), np.ones_like(onset_samples, dtype=int)))
        event_id = {'window': 1}
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=window_sec, baseline=None, preload=True)

        data = epochs.get_data()
        for idx, epoch in enumerate(data):
            stft_maps = [np.abs(stft(ch, fs=fs, nperseg=nperseg, noverlap=noverlap)[2]) for ch in epoch]
            stft_maps = np.stack(stft_maps)
            np.save(os.path.join(run_output_dir, f"epoch_{idx:03}.npy"), stft_maps)

        run_end = time.perf_counter()
        elapsed_run = round(run_end - run_start, 2)

        # Guardar en CSV
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_file = os.path.join(reports_dir, "processing_times.csv")

        entry = pd.DataFrame([{
            "subject": subject_id,
            "session": session_id,
            "run": run_id,
            "task": task,
            "runtime_seconds": elapsed_run,
            "duration_minutes": duration_minutes,
            "gpu_runtime_seconds": ""
        }])

        if os.path.exists(report_file):
            prev = pd.read_csv(report_file)

            for col in ["subject", "session", "run", "task"]:
                prev[col] = prev[col].astype(str)
                entry[col] = entry[col].astype(str)

            mask = (
                (prev["subject"] == subject_id) &
                (prev["session"] == session_id) &
                (prev["run"] == run_id) &
                (prev["task"] == task)
            )
            prev = prev[~mask]
            updated = pd.concat([prev, entry], ignore_index=True)
        else:
            updated = entry

        updated.to_csv(report_file, index=False)
        tqdm.write(f"[{subject_id}] {run_id}: {elapsed_run}s | duración: {duration_minutes} min")
