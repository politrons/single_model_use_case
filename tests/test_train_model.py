from __future__ import annotations

import pandas as pd


def test_prediction_proba_to_map_from_list(load_entry_module):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_a")
    result = mod._prediction_proba_to_map([0.2, 0.8])
    assert result == {"0": 0.2, "1": 0.8}


def test_prediction_proba_to_map_from_json_string(load_entry_module):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_b")
    result = mod._prediction_proba_to_map('{"0": 0.1, "1": 0.9}')
    assert result == {"0": 0.1, "1": 0.9}


def test_split_predict_output_for_dataframe(load_entry_module):
    mod = load_entry_module("src/training/train_model.py", "train_model_test_c")
    raw = pd.DataFrame({"prediction": [1, 0], "prediction_proba": [0.9, 0.1]})
    preds, probs = mod._split_predict_output(raw)
    assert list(preds) == [1, 0]
    assert list(probs) == [0.9, 0.1]

