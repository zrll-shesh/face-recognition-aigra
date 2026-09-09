import json
import os
import uuid
from datetime import datetime
import numpy as np


class EmployeeDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.records = {}
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.records = json.load(f)
        else:
            self.records = {}

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def enroll(self, name, embeddings, employee_id=None, metadata=None):
        employee_id = employee_id or str(uuid.uuid4())[:8]
        vectors = [np.asarray(e, dtype=np.float32).tolist() for e in embeddings]
        self.records[employee_id] = {
            "name": name,
            "embeddings": vectors,
            "num_samples": len(vectors),
            "enrolled_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.save()
        return employee_id

    def remove(self, employee_id):
        if employee_id in self.records:
            del self.records[employee_id]
            self.save()
            return True
        return False

    def get(self, employee_id):
        return self.records.get(employee_id)

    def list_employees(self):
        return [
            {
                "id": eid,
                "name": r["name"],
                "num_samples": r["num_samples"],
                "enrolled_at": r["enrolled_at"],
            }
            for eid, r in self.records.items()
        ]

    def all_embeddings(self):
        pairs = []
        for eid, r in self.records.items():
            for vec in r["embeddings"]:
                pairs.append((eid, r["name"], np.asarray(vec, dtype=np.float32)))
        return pairs

    def is_empty(self):
        return len(self.records) == 0

    def count(self):
        return len(self.records)
