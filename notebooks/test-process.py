from bids import BIDSLayout
import mne
import os

# Ruta base al dataset BIDS
bids_root = "/home/ferna96/Documents/personal/MaestriaFM/GPU/una_challenge_analysis/una-challenge-analysis/data/raw/data/ds005873/"

# Inicializar layout BIDS
layout = BIDSLayout(bids_root, validate=False)

# Buscar archivos EEG en formato .edf del sujeto sub-001
edf_files = layout.get(subject='001', suffix='eeg', extension='edf', return_type='filename')

# Validar que haya archivos
if not edf_files:
    raise FileNotFoundError("No se encontró ningún archivo .edf para sub-001.")

# Usar el primer archivo como ejemplo
edf_path = edf_files[0]
print(f"Archivo EEG encontrado: {edf_path}")

# Cargar archivo con MNE
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Filtrado: pasa banda (0.5–120 Hz)
raw.filter(l_freq=0.5, h_freq=120.0)

# Filtro notch a 50 Hz
raw.notch_filter(freqs=50)

# Visualizar info básica
print(raw.info)
