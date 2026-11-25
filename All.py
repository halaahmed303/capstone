from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# مسارات ملفاتك المحلية (حطيها جنب السكربت ده وقت التشغيل)
EXCEL_FILE = BASE_DIR / "4 Incubator temperature and incubator humidity infant.xlsx"
ECG_IMAGE = BASE_DIR / "ecg_waveform.png"

TEMP_MIN, TEMP_MAX = 36.5, 37.5  # حوالين 37°C
HUM_MIN, HUM_MAX = 40.0, 60.0  # من 40% لـ 60%


def get_env_status(row_index: int = 0, hour: int = 4) -> str:
  df_temp = pd.read_excel(
      EXCEL_FILE, sheet_name="Incubator temperature infant "
  ).iloc[:96]
  df_hum = pd.read_excel(
      EXCEL_FILE, sheet_name="Incubator humidity infant"
  ).iloc[:96]

  temp_val = float(df_temp.iloc[row_index][f"Incubator temperature {hour}h"])
  hum_val = float(df_hum.iloc[row_index][f"Incubator humidity {hour}h"])
  study_id = df_temp.iloc[row_index]["Study ID"]

  notes = []
  if not (TEMP_MIN <= temp_val <= TEMP_MAX):
    notes.append(f"تنبيه حرارة ({temp_val:.1f}°C)")
  if not (HUM_MIN <= hum_val <= HUM_MAX):
    notes.append(f"تنبيه رطوبة ({hum_val:.1f}%)")
  if not notes:
    notes.append("البيئة مضبوطة")

  return (
      f"حالة #{study_id} ({hour}h) -> حرارة: {temp_val:.1f}°C | رطوبة:"
      f" {hum_val:.1f}% | {' - '.join(notes)}"
  )


def show_summary(risk_status: str = "Healthy / At Risk (راجع baseline_seq.py)"):
  print("=== ملخص مراقبة الحضانة ===")
  print("1. التصنيف:", risk_status)
  print("2. الحرارة والرطوبة:", get_env_status(row_index=0, hour=4))
  print(
      "3. رسمة ECG:",
      "موجودة" if ECG_IMAGE.exists() else "غير موجودة في نفس الفولدر",
  )


if __name__ == "__main__":
  show_summary()
