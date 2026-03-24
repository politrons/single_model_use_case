from typing import Any
import logging

from sklearn.base import BaseEstimator  # type: ignore # noqa
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    CONFIG_SECTION_HYPERPARAM_SEARCH,
    SCIKIT_LIBRARY,
    SCIKIT_GRID,
    SCIKIT_HALVING_RANDOM,
    RAY_LIBRARY,
    RAY_SEARCH,
    SPLIT_GENERAL,
    CONFIG_SEARCH_STRATEGY,
    CONFIG_NUM_FOLDS,
    CONFIG_SPLIT_STRATEGY,
    CONFIG_FACTOR,
    CONFIG_MAX_EVALS,
    SPLIT_TIME,
    SPLIT_STRATIFY,
    CONFIG_PARALLELISM,
    CONFIG_N_JOBS,
    CONFIG_NUM_SAMPLES,
    CONFIG_SCHEDULER,
    CONFIG_SEARCH_MODEL_SPACE,
    CONFIG_SEARCH_ALGO,
    CONFIG_EVAL_METRIC,
    CONFIG_SCORING,
    CONFIG_SECTION_MODEL,
)
from databricks_mlops_stack.training.model.hyperparam_searches.scikit_search import scikit_hyperparam_search  # type: ignore # noqa
from databricks_mlops_stack.training.model.hyperparam_searches.ray_search import RaySearchCV   # type: ignore # noqa
from databricks_mlops_stack.utils.scoring import (   # type: ignore # noqa
    get_scorer_info_from_config,
    get_scorer_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.hyperparam_searches.base_search")

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_SEARCH_STRATEGIES = {
    SCIKIT_GRID: SCIKIT_LIBRARY,
    SCIKIT_HALVING_RANDOM: SCIKIT_LIBRARY,
    RAY_SEARCH: RAY_LIBRARY,
}

AVAILABLE_SEARCH_SPLITS = {
    SPLIT_GENERAL: KFold,
    SPLIT_TIME: TimeSeriesSplit,
    SPLIT_STRATIFY: StratifiedKFold,
}

def hyperparam_search(
        config: dict[str, Any],
        pipe: BaseEstimator,
        random_state: int,
    ) -> BaseEstimator:

    hconf = config.get(CONFIG_SECTION_HYPERPARAM_SEARCH)
    search = pipe

    if hconf is None:
        return search

    model_params_space = hconf.get(CONFIG_SEARCH_MODEL_SPACE)
    transformers_space: dict[str, Any] = dict() # placeholder for future implementations
    search_strategy = hconf.get(CONFIG_SEARCH_STRATEGY, SCIKIT_GRID)
    num_folds = hconf.get(CONFIG_NUM_FOLDS, 3)
    split_strategy = hconf.get(CONFIG_SPLIT_STRATEGY, SPLIT_GENERAL)
    parallelism = hconf.get(CONFIG_PARALLELISM)
    n_jobs = hconf.get(CONFIG_N_JOBS)
    factor = hconf.get(CONFIG_FACTOR, 3)
    num_samples = hconf.get(CONFIG_NUM_SAMPLES)
    max_evals = hconf.get(CONFIG_MAX_EVALS)
    scheduler = hconf.get(CONFIG_SCHEDULER, None)
    search_algo = hconf.get(CONFIG_SEARCH_ALGO)
    scoring: str = hconf.get(CONFIG_SCORING) # when adding support for custom/multiple metrics, change here
    eval_metric: str = hconf.get(CONFIG_EVAL_METRIC) # when adding support for custom/multiple metrics, change here

    ######################################
    # basic validation
    ######################################
    if search_strategy not in AVAILABLE_SEARCH_STRATEGIES:
        raise ValueError(f"Unsupported search strategy: {search_strategy}")

    if split_strategy not in AVAILABLE_SEARCH_SPLITS:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    
    if not any([model_params_space, transformers_space]):
        raise ValueError("At least one space for search should be provided")
    
    if parallelism and n_jobs:
        raise ValueError(f"Cannot specify both '{CONFIG_PARALLELISM}' and '{CONFIG_N_JOBS}', they are aliases")
    
    if num_samples and max_evals:
        raise ValueError(f"Cannot specify both '{CONFIG_NUM_SAMPLES}' and '{CONFIG_MAX_EVALS}', they are aliases")
    
    if scoring and eval_metric:
        raise ValueError(f"Cannot specify both '{CONFIG_SCORING}' and '{CONFIG_EVAL_METRIC}', they are aliases")

    ######################################
    # init and setting defaults
    ######################################
    strategy_library = AVAILABLE_SEARCH_STRATEGIES[search_strategy]
    LOG.info(f'Strategy library: {strategy_library}')

    split_object = AVAILABLE_SEARCH_SPLITS[split_strategy]
    if split_strategy == SPLIT_TIME:
        cv_splitter = split_object(
            n_splits=num_folds,
        )
    else:
        cv_splitter = split_object(
            n_splits=num_folds,
            shuffle=False,
        )
    LOG.info(f'CV splitter: {cv_splitter}')

    model_params_space = model_params_space if model_params_space else dict()
    transformers_space = transformers_space if transformers_space else dict()
    all_spaces = model_params_space | transformers_space
    LOG.info(f'all spaces: {all_spaces}')

    parallelism = parallelism or n_jobs
    LOG.info(f'parallelism: {parallelism}')

    max_evals = max_evals or num_samples
    max_evals = max_evals or 1
    LOG.info(f'max_evals: {max_evals}')

    ######################################
    # specific for scorer
    ######################################
    scoring = scoring or eval_metric
    pipe_scorer_info = get_scorer_info(pipe, True)
    config_scorer_info = get_scorer_info_from_config(scoring)
    if pipe_scorer_info['is_default_scorer'] and config_scorer_info['is_default_scorer']:
        final_scorer_info = pipe_scorer_info
    else:
        if not pipe_scorer_info['is_default_scorer']:
            final_scorer_info = pipe_scorer_info
        elif not config_scorer_info['is_default_scorer']:
            final_scorer_info = config_scorer_info
        if pipe_scorer_info['is_default_scorer'] == config_scorer_info['is_default_scorer'] and pipe_scorer_info['function_name'] != config_scorer_info['function_name']:
            error_msg = f"'{CONFIG_EVAL_METRIC}' set in {CONFIG_SECTION_MODEL} is different from "
            error_msg += f"'{CONFIG_EVAL_METRIC}/{CONFIG_SCORING}' specified in {CONFIG_SECTION_HYPERPARAM_SEARCH}"
            raise ValueError(error_msg)
    LOG.info(f'Final scorer: {final_scorer_info}')

    ######################################
    # select requested library
    ######################################
    if strategy_library == SCIKIT_LIBRARY:
        search = scikit_hyperparam_search(
            search_strategy=search_strategy,
            pipe=pipe,
            all_spaces=all_spaces,
            cv_splitter=cv_splitter,
            random_state=random_state,
            parallelism=parallelism,
            factor=factor,
            scoring=final_scorer_info['function_name'],
        )
    elif strategy_library == RAY_LIBRARY:
        search = RaySearchCV(
            estimator=pipe,
            user_param_space=all_spaces,
            cv_splitter=cv_splitter,
            random_state=random_state,
            num_samples=max_evals,
            scorer_info=final_scorer_info,
            scheduler_name=scheduler,
            search_algo_name=search_algo,
        )
    else:
        raise ValueError(f"Strategy library {strategy_library} not implemented.")

    return search