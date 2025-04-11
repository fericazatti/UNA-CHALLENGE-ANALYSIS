import os
import numpy as np
import mne
import pandas as pd
import time
from tqdm import tqdm
from cuda_stft import compute_stft_gpu
from bids import BIDSLayout

mne.set_log_level('WARNING')

def generate_stft_maps_for_subject(subject_id: str, session_id: str, project_root: str,
                                    task: str = "szMonitoring",
                                    window_sec: int = 5,
                                    overlap: float = 0.5,
                                    fs: int = 256,
                                    nperseg: int = 128,
                                    noverlap: int = 64):

    bids_root = os.path.join(project_root, "data/raw/data/ds005873")
    output_root = os.path.join(project_root, f"data/processed/stft_maps_gpu/sub-{subject_id}")
    os.makedirs(output_root, exist_ok=True)

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

    print(f"\n[{subject_id}] Procesando {len(eeg_files)} run(s) con GPU:")

    for edf_path in tqdm(sorted(eeg_files), desc=f"GPU {subject_id}"):
        basename = os.path.basename(edf_path)
        run_id = basename.split("_")[3]

        run_start = time.perf_counter()

        run_output_dir = os.path.join(output_root, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.filter(l_freq=0.5, h_freq=120.0)
        raw.notch_filter(freqs=50)

        duration = raw.n_times / raw.info['sfreq']
        duration_minutes = round(duration / 60, 2)

        step = window_sec * (1 - overlap)
        onset_times = np.arange(0, duration - window_sec, step)
        onset_samples = (onset_times * fs).astype(int)
        events = np.column_stack((onset_samples, np.zeros_like(onset_samples), np.ones_like(onset_samples, dtype=int)))
        event_id = {'window': 1}
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=window_sec, baseline=None, preload=True)

        data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
        n_epochs, n_channels, n_samples = data.shape

        for ch_idx in range(n_channels):
            flat_epochs = data[:, ch_idx, :].reshape(-1)
            spectrogram = compute_stft_gpu(flat_epochs.astype(np.float32), n_samples, nperseg, 1e-10)

            for ep_idx in range(n_epochs):
                ep_map = spectrogram[ep_idx]  # (frecuencias,)
                np.save(os.path.join(run_output_dir, f"ch{ch_idx:02}_epoch_{ep_idx:03}.npy"), ep_map)

        run_end = time.perf_counter()
        elapsed_run = round(run_end - run_start, 2)

        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_file = os.path.join(reports_dir, "processing_times.csv")

        entry = pd.DataFrame([{
            "subject": subject_id,
            "session": session_id,
            "run": run_id,
            "task": task,
            "runtime_seconds": "",
            "duration_minutes": duration_minutes,
            "gpu_runtime_seconds": elapsed_run
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
        tqdm.write(f"[GPU {subject_id}] {run_id}: {elapsed_run}s | duración: {duration_minutes} min")
