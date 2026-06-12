from __future__ import annotations


def enn_all_train_rows(model):
    """Return (x, y, yvar?) for all rows via index-based gather."""
    n = len(model)
    return model.train_rows_at(list(range(n)))
