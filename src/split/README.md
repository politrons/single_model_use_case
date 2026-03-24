# Split

Data split strategies and contracts used by training/validation workflows.

## Modules
- `split.py`
  - Main split orchestration entrypoint.
- `split_contract.py`
  - Abstract contract for custom split implementations.
- `split_config.py`
  - Configuration-based split implementation.
- `general.py`
  - General split strategy helpers.
- `time_series.py`
  - Time-series-aware split helpers.

## Related package
- `split/strategies/` contains concrete strategy implementations.

