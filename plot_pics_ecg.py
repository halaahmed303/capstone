from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import wfdb

BASE_DIR = Path(__file__).resolve().parent
record_name = "infant1_ecg"
record_path = BASE_DIR / record_name

record = wfdb.rdrecord(str(record_path))
signal = (
    record.p_signal[:, 0]
    if record.p_signal is not None
    else record.d_signal[:, 0]
)
fs = record.fs

valid_mask = ~np.isnan(signal)
signal = signal[valid_mask]

# نرسم أول 4 ثواني عشان شكل الموجة يكون واضح
n_samples = min(int(4 * fs), len(signal))
y_wave = signal[:n_samples]
t_axis = np.arange(n_samples) / fs

plt.figure(figsize=(11, 4))
plt.plot(t_axis, y_wave, linewidth=1.2)
plt.title(f"Preterm Infant ECG Waveform ({record_name})")
plt.xlabel("Time (seconds)")
plt.ylabel(record.units[0] if record.units else "Amplitude")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(str(BASE_DIR / "ecg_waveform.png"), dpi=200)
plt.show()
print("تم حفظ الرسم في ecg_waveform.png")
