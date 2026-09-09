import os
import csv
from datetime import datetime

FIELDS = [
    "timestamp", "employee_id", "name", "similarity",
    "confidence_percent", "status", "verification_method",
]


def append_record(log_path, employee_id, name, similarity, confidence_percent, status, verification_method):
    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "employee_id": employee_id,
            "name": name,
            "similarity": round(similarity, 4) if similarity is not None else "",
            "confidence_percent": confidence_percent,
            "status": status,
            "verification_method": verification_method,
        })


def read_log(log_path):
    import pandas as pd

    if not os.path.exists(log_path):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(log_path)
