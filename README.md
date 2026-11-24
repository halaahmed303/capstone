# Infant Incubator Monitoring Prototype (Raspberry Pi 4)

A prototype monitoring pipeline for infant incubators running on **Raspberry Pi 4**, focused on environmental monitoring, vital/clinical status classification (`Healthy` / `At Risk`), and ECG waveform plotting (without cameras or medical diagnosis).

## Project Files
- `baseline_seq.py`: Neonatal status classification (`Healthy` / `At Risk`).
- `plot_pics_ecg.py`: ECG waveform visualization.
- `temperature.py` (or `temperature. py`): Incubator temperature and humidity check and alerts.
- `كله.py`: Combined summary output for classification, environment status, and ECG plot availability.
- `libraries`: Required Python packages (such as `pandas` and `openpyxl`).

## Local Data Setup (Not Uploaded to GitHub)
Large dataset files (such as `4 Incubator temperature and incubator humidity infant.xlsx` and ECG signal/image files) are kept locally on the machine / Raspberry Pi 4 and are not pushed to GitHub. Place them alongside the scripts or in the expected relative path before running.

## Environment Thresholds (Prototype)
- **Temperature:** around `37°C` (configurable in script)
- **Humidity:** `40% - 60%`
  # capstone
