import numpy as np

from predictx.data import LOADERS


def test_real_labels_and_outcome_feature_separation():
    for name in ["ilpd", "ckd", "nhanes"]:
        c = LOADERS[name]()
        assert len(c.ids) == len(set(c.ids))
        assert not set(c.X) & set(c.y)
        assert set(np.unique(c.y.to_numpy()[np.isfinite(c.y.to_numpy())])) <= {0.0, 1.0}
        assert np.isfinite(c.X.to_numpy()[c.X.notna().to_numpy()]).all()


def test_nhanes_actual_alignment_and_absent_breast_label():
    c = LOADERS["nhanes"]()
    assert len(c.y.columns) == 4
    assert c.y.notna().sum(axis=1).eq(4).any()
    assert "breast_cancer" not in c.y
    assert not any(k.startswith(("DIQ", "KIQ", "MCQ")) for k in c.X)
