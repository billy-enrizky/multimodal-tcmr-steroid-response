import numpy as np
from src.oof import OOFAccumulator


def test_each_patient_recorded_once_per_modality():
    acc = OOFAccumulator()
    acc.add(modality="clinical", patient_ids=["A", "B"], y_true=[0, 1], y_prob=[0.2, 0.8])
    acc.add(modality="clinical", patient_ids=["C"], y_true=[1], y_prob=[0.6])
    df = acc.to_frame()
    clin = df[df.modality == "clinical"]
    assert sorted(clin.patient_id.tolist()) == ["A", "B", "C"]
    assert len(clin) == 3
    assert clin.patient_id.is_unique


def test_duplicate_patient_in_modality_raises():
    acc = OOFAccumulator()
    acc.add(modality="clinical", patient_ids=["A"], y_true=[0], y_prob=[0.2])
    try:
        acc.add(modality="clinical", patient_ids=["A"], y_true=[0], y_prob=[0.3])
        assert False, "expected ValueError on duplicate patient"
    except ValueError:
        pass
