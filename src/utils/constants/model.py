from typing import Final

# Config params sections
CONFIG_SECTION_MODEL: Final[str] = "model"
CONFIG_SECTION_TARGET_TRANSFORMER: Final[str] = "target_transformer"
CONFIG_SECTION_SAMPLING: Final[str] = "sampling"
CONFIG_SECTION_FEATURES_TRANSFORMERS: Final[str] = "features_transformers"
CONFIG_SECTION_HYPERPARAM_SEARCH: Final[str] = "hyperparam_search"
CONFIG_SECTION_EARLY_STOPPING: Final[str] = "early_stopping"
CONFIG_SECTION_DISCARDED_FEATURES: Final[str] = "discarded_features"

# Config params constants
CONFIG_TASK: Final[str] = "task"
CONFIG_MODEL_CLASS: Final[str] = "type"
CONFIG_PREDICTION_METHOD: Final[str] = "prediction_method"
PREDICTION_METHOD_PREDICT_PROBA: Final[str] = "predict_proba"
CONFIG_SAMPLING_CLASS: Final[str] = "type"
CONFIG_TRANSFORMER_CLASS: Final[str] = "type"
CONFIG_TRANSFORMER_FEATURES: Final[str] = "features"
CONFIG_METRIC_CLASS: Final[str] = "type"
CONFIG_SCORING_CLASS: Final[str] = "type"
CONFIG_LOSS_CLASS: Final[str] = "type"
CONFIG_OPTIMIZER_CLASS: Final[str] = "type"

# Shared constants
CONFIG_PARALLELISM: Final[str] = "parallelism"
CONFIG_N_JOBS: Final[str] = "n_jobs"
CONFIG_EVAL_SET: Final[str] = "eval_set"

# Pipeline - core
STEP_MODEL: Final[str] = "model"
STEP_PREPROCESS: Final[str] = "preprocessor"
STEP_SAMPLER: Final[str] = "sampler"
STEP_TARGET_TRANSFORMER: Final[str] = "target_transformer"

# Hyper parameters search - base
CONFIG_SEARCH_STRATEGY: Final[str] = "search_strategy"
CONFIG_SEARCH_MODEL_SPACE: Final[str] = "model_params_space"
CONFIG_NUM_FOLDS: Final[str] = "num_folds"
CONFIG_FACTOR: Final[str] = "factor"
CONFIG_MAX_EVALS: Final[str] = "max_evals"
CONFIG_EVAL_METRIC: Final[str] = "eval_metric"
CONFIG_SCORING: Final[str] = "scoring"

# Hyper parameters search - split
CONFIG_SPLIT_STRATEGY: Final[str] = "split_strategy"
SPLIT_GENERAL: Final[str] = "general"
SPLIT_TIME: Final[str] = "time_series"
SPLIT_STRATIFY: Final[str] = "stratified"

# Hyper parameters search - scikit
SCIKIT_LIBRARY: Final[str] = "scikit-learn"
SCIKIT_GRID: Final[str] = "grid"
SCIKIT_HALVING_RANDOM: Final[str] = "halving_random"

# Hyper parameters search - ray
RAY_LIBRARY: Final[str] = "ray_library"
RAY_SEARCH: Final[str] = "ray"
CONFIG_NUM_SAMPLES: Final[str] = "num_samples"
CONFIG_SCHEDULER: Final[str] = "scheduler"
CONFIG_SEARCH_ALGO: Final[str] = "search_algo"

# Algos - core
ALGO_XGBOOST: Final[str] = "xgboost"
ALGO_LIGHTGBM: Final[str] = "lightgbm"
ALGO_SCIKIT_RANDOMFOREST: Final[str] = "scikit.random_forest"
ALGO_SCIKIT_RIDGE: Final[str] = "scikit.ridge"
ALGO_SCIKIT_LINEAR: Final[str] = "scikit.linear"
ALGO_TENSORFLOW: Final[str] = "tensorflow"
ALGO_NEURALPROPHET: Final[str] = "neuralprophet"

# Algos - tensorflow - core
CONFIG_BUILDING_BLOCKS: Final[str] = "building_blocks"
CONFIG_ARCHITECTURE: Final[str] = "architecture"
CONFIG_LOSS: Final[str] = "loss"
CONFIG_OPTIMIZER: Final[str] = "optimizer"
CONFIG_EPOCHS: Final[str] = "epochs" # also used in neuralprophet
CONFIG_BATCH_SIZE: Final[str] = "batch_size" # also used in neuralprophet
CONFIG_BLOCK: Final[str] = "block"
CONFIG_BLOCK_NAME: Final[str] = "name"
CONFIG_BLOCK_COMPOSITE_LAYERS: Final[str] = "composite_layers"
CONFIG_BLOCK_OVERRIDE: Final[str] = "overrides"
CONFIG_LAYER: Final[str] = "layer"
CONFIG_LAYER_NAME: Final[str] = "name"
CONFIG_LAYER_TYPE: Final[str] = "type"
CONFIG_OPTIMISE_THRESHOLD: Final[str] = "optimise_threshold"

# Algos - tensorflow - optimizer
OPTIMIZER_ADAM: Final[str] = "adam"
OPTIMIZER_SGD: Final[str] = "sgd"
OPTIMIZER_RMSPROP: Final[str] = "rmsprop"

# Algos - tensorflow - lossess
LOSS_BINARYCROSSENTROPY: Final[str] = "binary_crossentropy"
LOSS_MEANSQUAREDERROR: Final[str] = "mean_squared_error" # also used in neuralprophet
LOSS_HUBER: Final[str] = "huber"

# Algos - tensorflow - layers
LAYER_DENSE: Final[str] = "dense"
LAYER_DROPOUT: Final[str] = "dropout"
LAYER_BATCHNORM: Final[str] = "batchnormalization"
LAYER_FLATTEN: Final[str] = "flatten"

# Algos - tensorflow - early stopping
TF_PATIENCE: Final[str] = "patience"
TF_MIN_DELTA: Final[str] = "min_delta"

# Algos - neuralprophet - core
FREQUENCY: Final[str] = "frequency"
CONFIG_ID_COLUMN: Final[str] = "id_column"

# Algos - neuralprophet - adders
LAGGED_REGRESSORS: Final[str] = "lagged_regressors"
FUTURE_REGRESSORS: Final[str] = "future_regressors"
EVENTS: Final[str] = "events"
COUNTRY_HOLIDAYS: Final[str] = "country_holidays"
SEASONALITY: Final[str] = "seasonality"

# Metrics
METRIC_AUC: Final[str] = "auc"
METRIC_R2_SCORE: Final[str] = "r2_score"
METRIC_R2: Final[str] = "r2"
METRIC_FALSE_POSITIVES: Final[str] = "false_positives"
METRIC_FALSE_NEGATIVES: Final[str] = "false_negatives"

# Tasks
TASK_CLASSIFICATION: Final[str] = "classification"
TASK_REGRESSION: Final[str] = "regression"

# Sampling
SAMPLING_SMOTE: Final[str] = "smote"

# Transformers
TRANSFORMER_MINMAX: Final[str] = "minmaxscaler"
TRANSFORMER_QUANTILE: Final[str] = "quantiletransformer"
TRANSFORMER_ONEHOT: Final[str] = "onehotencoder"
TRANSFORMER_FUNCTION: Final[str] = "functiontransformer"
TRANSFORMER_SIMPLEIMP: Final[str] = "simpleimputer"
TRANSFORMER_LOG: Final[str] = 'logtransformer'
TRANSFORMER_CLIP: Final[str] = 'clippertransformer'
TRANSFORMER_ROBUST: Final[str] = 'robustscaler'
TRANSFORMER_STANDARD: Final[str] = 'standardscaler'
TRANSFORMER_ORDINAL: Final[str] = 'ordinalencoder'

__all__ = [
    "CONFIG_SECTION_MODEL",
    "CONFIG_SECTION_TARGET_TRANSFORMER",
    "CONFIG_SECTION_SAMPLING",
    "CONFIG_SECTION_FEATURES_TRANSFORMERS",
    "CONFIG_SECTION_HYPERPARAM_SEARCH",
    "CONFIG_SECTION_EARLY_STOPPING",
    "CONFIG_SECTION_DISCARDED_FEATURES",
    "CONFIG_TASK",
    "CONFIG_MODEL_CLASS",
    "CONFIG_PREDICTION_METHOD",
    "PREDICTION_METHOD_PREDICT_PROBA",
    "CONFIG_SAMPLING_CLASS",
    "CONFIG_TRANSFORMER_CLASS",
    "CONFIG_TRANSFORMER_FEATURES",
    "CONFIG_METRIC_CLASS",
    "CONFIG_SCORING_CLASS",
    "CONFIG_LOSS_CLASS",
    "CONFIG_OPTIMIZER_CLASS",
    "CONFIG_PARALLELISM",
    "CONFIG_N_JOBS",
    "CONFIG_EVAL_SET",
    "STEP_MODEL",
    "STEP_PREPROCESS",
    "STEP_SAMPLER",
    "STEP_TARGET_TRANSFORMER",
    "CONFIG_SEARCH_STRATEGY",
    "CONFIG_SEARCH_MODEL_SPACE",
    "CONFIG_NUM_FOLDS",
    "CONFIG_FACTOR",
    "CONFIG_MAX_EVALS",
    "CONFIG_EVAL_METRIC",
    "CONFIG_SCORING",
    "CONFIG_SPLIT_STRATEGY",
    "SPLIT_GENERAL",
    "SPLIT_TIME",
    "SPLIT_STRATIFY",
    "SCIKIT_LIBRARY",
    "SCIKIT_GRID",
    "SCIKIT_HALVING_RANDOM",
    "RAY_LIBRARY",
    "RAY_SEARCH",
    "CONFIG_NUM_SAMPLES",
    "CONFIG_SCHEDULER",
    "CONFIG_SEARCH_ALGO",
    "ALGO_XGBOOST",
    "ALGO_LIGHTGBM",
    "ALGO_SCIKIT_RANDOMFOREST",
    "ALGO_SCIKIT_RIDGE",
    "ALGO_SCIKIT_LINEAR",
    "ALGO_TENSORFLOW",
    "ALGO_NEURALPROPHET",
    "CONFIG_BUILDING_BLOCKS",
    "CONFIG_ARCHITECTURE",
    "CONFIG_LOSS",
    "CONFIG_OPTIMIZER",
    "CONFIG_EPOCHS",
    "CONFIG_BATCH_SIZE",
    "CONFIG_BLOCK",
    "CONFIG_BLOCK_NAME",
    "CONFIG_BLOCK_COMPOSITE_LAYERS",
    "CONFIG_BLOCK_OVERRIDE",
    "CONFIG_LAYER",
    "CONFIG_LAYER_NAME",
    "CONFIG_LAYER_TYPE",
    "OPTIMIZER_ADAM",
    "OPTIMIZER_SGD",
    "OPTIMIZER_RMSPROP",
    "LOSS_BINARYCROSSENTROPY",
    "LOSS_MEANSQUAREDERROR",
    "LOSS_HUBER",
    "LAYER_DENSE",
    "LAYER_DROPOUT",
    "LAYER_BATCHNORM",
    "LAYER_FLATTEN",
    "TF_PATIENCE",
    "TF_MIN_DELTA",
    "FREQUENCY",
    "CONFIG_ID_COLUMN",
    "LAGGED_REGRESSORS",
    "FUTURE_REGRESSORS",
    "EVENTS",
    "COUNTRY_HOLIDAYS",
    "SEASONALITY",
    "METRIC_AUC",
    "METRIC_R2_SCORE",
    "METRIC_R2",
    "METRIC_FALSE_POSITIVES",
    "METRIC_FALSE_NEGATIVES",
    "TASK_CLASSIFICATION",
    "TASK_REGRESSION",
    "SAMPLING_SMOTE",
    "TRANSFORMER_MINMAX",
    "TRANSFORMER_QUANTILE",
    "TRANSFORMER_ONEHOT",
    "TRANSFORMER_FUNCTION",
    "TRANSFORMER_SIMPLEIMP",
    "TRANSFORMER_LOG",
    "TRANSFORMER_CLIP",
    "TRANSFORMER_ROBUST",
    "TRANSFORMER_STANDARD",
    "TRANSFORMER_ORDINAL",
]
