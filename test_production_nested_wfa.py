import pandas as pd

from core.production_nested_wfa import (
    GATE_GRID, GateConfig, RotationParams, _folds, parameter_grid,
)


def test_nested_folds_are_expanding_and_oos_non_overlapping():
    idx = pd.bdate_range("2020-01-01", periods=30)
    folds = _folds(idx, train_days=12, test_days=7)
    assert [len(test) for _, test in folds] == [7, 7, 4]
    assert len(folds[1][0]) == 19
    assert folds[0][0][-1] < folds[0][1][0]
    assert folds[0][1][-1] < folds[1][1][0]


def test_expanded_grid_contains_production_neighborhood():
    grid = parameter_grid()
    assert RotationParams(60, 10, 8, 20, 200) in grid
    assert len(grid) == 162


def test_nested_gate_grid_contains_all_combinations():
    assert len(GATE_GRID) == 8
    assert GateConfig(True, True, True) in GATE_GRID
    assert GateConfig(False, False, False) in GATE_GRID
