"""Out-of-fold prediction accumulator: each patient predicted exactly once per modality."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class OOFAccumulator:
    def __init__(self):
        self._rows = []
        self._seen = set()  # (modality, patient_id)

    def add(self, modality, patient_ids, y_true, y_prob):
        for pid, yt, yp in zip(patient_ids, y_true, y_prob):
            key = (modality, str(pid))
            if key in self._seen:
                raise ValueError(f"Patient {pid} already recorded for modality {modality}")
            self._seen.add(key)
            self._rows.append(
                {"modality": modality, "patient_id": str(pid), "true_label": int(yt), "prob": float(yp)}
            )

    def to_frame(self):
        return pd.DataFrame(self._rows, columns=["modality", "patient_id", "true_label", "prob"])

    def save(self, path):
        df = self.to_frame()
        df.to_csv(path, index=False)
        logger.info("Saved %d OOF rows to %s", len(df), path)
        return df
