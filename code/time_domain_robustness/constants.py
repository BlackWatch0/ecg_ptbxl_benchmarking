"""Column names used by the time-domain robustness pipeline."""

KEY_COLUMNS = ("RecordNumber", "LeadIndex", "BeatIndex")

FEATURE_COLUMNS = (
    "Mean_RR", "CV_RR", "pNN50", "Kurt_RR", "Skew_P", "Skew_QRS",
    "Skew_ST_T", "Skew_global", "RMS_global", "SD_R_amp", "SE", "NTEO",
    "ZCR",
)
