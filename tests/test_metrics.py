"""`fv.metrics` — pure arrays, no fv imports, so these tests need no world.

`spearman` is checked against HAND-COMPUTED golden numbers (metrica-de-tarea.md
§5.2) and not against scipy: scipy is not in this venv, and a test that skips
when its oracle is missing is a test that never runs.
"""

from __future__ import annotations

import pytest

# (x, y, expected) — the table of metrica-de-tarea.md §5.2, recomputed by hand:
# ranks with ties sharing the mean rank, then Pearson over the ranks.
GOLDEN = [
    ("monotona perfecta", [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 1.0),
    ("inversa perfecta", [1, 2, 3, 4, 5], [50, 40, 30, 20, 10], -1.0),
    ("con empates en y", [1, 2, 3, 4, 5], [5, 6, 7, 8, 7], 0.8207826816681233),
    ("ruidosa", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
     [0.11, 0.30, 0.25, 0.55, 0.44, 0.70], 0.8857142857142857),
]


@pytest.mark.parametrize("name,x,y,expected", GOLDEN,
                         ids=[g[0].replace(" ", "-") for g in GOLDEN])
def test_spearman_matches_hand_computed_cases(name, x, y, expected):
    from fv.metrics import spearman
    assert spearman(x, y) == pytest.approx(expected, abs=1e-12)


def test_spearman_is_none_for_a_constant_series():
    """None, never 0: with a constant series the correlation is UNDEFINED, and 0
    would be read as «no correlacionan», which is a different claim."""
    from fv.metrics import spearman
    assert spearman([1, 2, 3], [7, 7, 7]) is None
    assert spearman([7, 7, 7], [1, 2, 3]) is None
    assert spearman([1], [2]) is None            # one point has no ranking either


def test_spearman_is_symmetric_and_rank_only():
    """It ranks: any monotone rescaling of a series leaves it unchanged."""
    from fv.metrics import spearman
    x = [0.1, 0.4, 0.2, 0.9, 0.5]
    y = [3.0, 1.0, 2.0, 5.0, 4.0]
    r = spearman(x, y)
    assert spearman(y, x) == pytest.approx(r)
    assert spearman([v ** 3 for v in x], y) == pytest.approx(r)


def test_spearman_refuses_mismatched_series():
    """Two series of different length are a bug at the caller, not a number."""
    from fv.metrics import spearman
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])
