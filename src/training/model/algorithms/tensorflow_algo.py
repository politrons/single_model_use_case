import logging
from typing import Any, Final
import random

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # type: ignore # noqa
import pandas as pd  # type: ignore # noqa
import numpy as np  # type: ignore # noqa
from sklearn.metrics import accuracy_score  # type: ignore # noqa
import tensorflow as tf  # type: ignore # noqa
from tensorflow.keras import (  # type: ignore # noqa
    layers,
    Sequential,
    losses,
    metrics,
)
from tensorflow.keras.optimizers import Optimizer, Adam, SGD, RMSprop  # type: ignore # noqa
from tensorflow.keras.losses import Loss as tfLoss  # type: ignore # noqa
from tensorflow.keras.metrics import Metric as tfMetric  # type: ignore # noqa
from tensorflow.keras.callbacks import History as tfHistory  # type: ignore # noqa
from tensorflow.keras.callbacks import EarlyStopping as tfEarlyStopping  # type: ignore # noqa

from databricks_mlops_stack.utils.scoring import get_metric_info  # type: ignore # noqa
from databricks_mlops_stack.utils.constants.model import (  # type: ignore # noqa
    OPTIMIZER_ADAM,
    OPTIMIZER_SGD,
    OPTIMIZER_RMSPROP,
    LOSS_BINARYCROSSENTROPY,
    LOSS_MEANSQUAREDERROR,
    LOSS_HUBER,
    LAYER_DENSE,
    LAYER_DROPOUT,
    LAYER_BATCHNORM,
    LAYER_FLATTEN,
    CONFIG_EPOCHS,
    CONFIG_BATCH_SIZE,
    CONFIG_BUILDING_BLOCKS,
    CONFIG_BLOCK_NAME,
    CONFIG_BLOCK_COMPOSITE_LAYERS,
    CONFIG_ARCHITECTURE,
    CONFIG_BLOCK,
    CONFIG_LAYER,
    CONFIG_LAYER_TYPE,
    CONFIG_LAYER_NAME,
    CONFIG_BLOCK_OVERRIDE,
    CONFIG_LOSS,
    CONFIG_EVAL_METRIC,
    CONFIG_SCORING,
    CONFIG_LOSS_CLASS,
    CONFIG_METRIC_CLASS,
    CONFIG_SCORING_CLASS,
    CONFIG_OPTIMIZER,
    CONFIG_OPTIMIZER_CLASS,
    METRIC_AUC,
    METRIC_R2_SCORE,
    CONFIG_OPTIMISE_THRESHOLD,
    METRIC_R2,
    METRIC_FALSE_POSITIVES,
    METRIC_FALSE_NEGATIVES,
    CONFIG_EVAL_SET,
    TF_PATIENCE,
    TF_MIN_DELTA,
    TRANSFORMER_ONEHOT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("framework.training.model.algorithms.tensorflow_algo")

# -----------------------------------------------------------------------------
# Catalogs
# -----------------------------------------------------------------------------
AVAILABLE_OPTIMIZERS = {
    OPTIMIZER_ADAM: Adam,
    OPTIMIZER_SGD: SGD,
    OPTIMIZER_RMSPROP: RMSprop,
}

AVAILABLE_TF_LOSSES = {
    LOSS_BINARYCROSSENTROPY: losses.BinaryCrossentropy,
    LOSS_MEANSQUAREDERROR: losses.MeanSquaredError,
    LOSS_HUBER: losses.Huber,
}

AVAILABLE_TF_METRICS = { # flexibility: although string is used for either loss or metric, when importing use the right class
    LOSS_MEANSQUAREDERROR: metrics.MeanSquaredError,
    METRIC_AUC: metrics.AUC,
    METRIC_R2_SCORE: metrics.R2Score,
    METRIC_R2: metrics.R2Score,
    METRIC_FALSE_POSITIVES: metrics.FalsePositives,
    METRIC_FALSE_NEGATIVES: metrics.FalseNegatives,
}

AVAILABLE_LAYERS = {
    LAYER_DENSE: layers.Dense,
    LAYER_DROPOUT: layers.Dropout,
    LAYER_BATCHNORM: layers.BatchNormalization,
    LAYER_FLATTEN: layers.Flatten,  # In case needed, though for tabular usually not
}

_DEFAULT_BINARY_CLASSIFICATION_THRESHOLD: Final[float] = 0.5

class TFBaseModel(BaseEstimator):
    """
    Base class for TensorFlow models compatible with scikit-learn.
    """
    def __init__(
        self,
        config: dict[str, Any],
        random_state: int,
        params: dict,
        extra_params: dict,
    ) -> None:

        self.config: dict[str, Any] = config
        LOG.info(f"config: {self.config}")
        self.params: dict = params
        LOG.info(f"params: {self.params}")
        self.extra_params: dict = extra_params
        LOG.info(f"extra params: {self.extra_params}")

        self.random_state: int = random_state
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self.loss: tfLoss | None = None
        self.metric: tfMetric | None = None
        self.metric_is_greater_is_better: bool = True
        self.optimizer: Optimizer | None = None
        self.history: tfHistory | None = None
        self.epochs_completed: int = -1

        self.epochs: int = config.get(CONFIG_EPOCHS, 10)
        LOG.info(f"number epochs: {self.epochs}")

        self.batch_size: int = config.get(CONFIG_BATCH_SIZE, 32)
        LOG.info(f"batch size: {self.batch_size}")

        self.optimise_threshold: bool = config.get(CONFIG_OPTIMISE_THRESHOLD, False)
        if not isinstance(self.optimise_threshold, bool):
            raise ValueError(f"{CONFIG_OPTIMISE_THRESHOLD} needs to be boolean")
        if not self.able_to_optimise_threshold and self.optimise_threshold:
            raise ValueError(f"Cannot set {CONFIG_OPTIMISE_THRESHOLD} to True for this type of task")
        LOG.info(f"optimise threshold: {self.optimise_threshold}")

        for transformer in self.extra_params.get('all_feature_transformers', []):
            if transformer in [TRANSFORMER_ONEHOT]:
                raise ValueError(f"Currently the transformer {transformer} is not supported for TensorFlow models")
            
        if self.extra_params.get('has_hyper_search', False):
            raise ValueError("Currently hyperparam search is not supported for TensorFlow models")

        # Build the model
        self.model = self._build_model()

        return

    def _resolve_blocks(
            self,
        ) -> dict[str, list[dict[str, Any]]]:
        """
        Parse building_blocks into a dict of name to list of layer configs.
        """
        blocks = {}
        building_blocks = self.config.get(CONFIG_BUILDING_BLOCKS, [])
        for block in building_blocks:
            name = block[CONFIG_BLOCK_NAME]
            if name in blocks:
                raise ValueError(f"Duplicate block {CONFIG_BLOCK_NAME}: {name}")
            blocks[name] = block[CONFIG_BLOCK_COMPOSITE_LAYERS]
        return blocks
    
    def _resolve_layer(
            self,
            added_layers: int,
            layer_config: dict[str, Any],
            override_config: dict[str, Any] | None = None,
            maybe_prefix_name: str | None = None,
        ) -> layers.Layer:

        layer_class_name = layer_config.get(CONFIG_LAYER_TYPE)
        if not layer_class_name:
            raise ValueError(f"Parameter {CONFIG_LAYER_TYPE} for layer is required")
        layer_class_name = layer_class_name.lower().strip()
        if layer_class_name not in AVAILABLE_LAYERS:
            raise ValueError(f"Unsupported layer {CONFIG_LAYER_TYPE}: {layer_class_name}")
        layer_class = AVAILABLE_LAYERS[layer_class_name]

        layer_params = layer_config.get("params", {})
        maybe_layer_name = layer_config.get(CONFIG_LAYER_NAME)
        if maybe_prefix_name or maybe_layer_name:
            defined_layer_name: str = ''
            if maybe_prefix_name:
                defined_layer_name = f'{maybe_prefix_name}_'
            if maybe_layer_name:
                defined_layer_name += f'{maybe_layer_name}_'
            defined_layer_name += f'{added_layers}'
            layer_params.update({'name': defined_layer_name})

        if override_config:
            layer_name = layer_config.get(CONFIG_LAYER_NAME)
            if layer_name and layer_name in override_config:
                layer_params.update(override_config[layer_name])

        return layer_class(**layer_params)

    def _resolve_hidden_layers(
            self,
        ) -> list[layers.Layer]:
        """
        Build the list of hidden layers based on architecture config.
        """
        blocks = self._resolve_blocks()
        architecture = self.config.get(CONFIG_ARCHITECTURE)
        if not architecture:
            raise ValueError(f"{CONFIG_ARCHITECTURE} is required")

        hidden_layers: list[layers.Layer] = []
        for item in architecture:
            if CONFIG_BLOCK in item:
                block_name = item[CONFIG_BLOCK]
                if block_name not in blocks:
                    raise ValueError(f"Unknown {CONFIG_BLOCK}: {block_name}")
                block_layers = [layer.copy() for layer in blocks[block_name]]  # Deep copy to avoid mutation
                override_config = item.get(CONFIG_BLOCK_OVERRIDE, {})

                for layer_config in block_layers:
                    this_layer = self._resolve_layer(
                        len(hidden_layers),
                        layer_config,
                        override_config,
                        block_name,
                    )
                    hidden_layers.append(this_layer)

            elif CONFIG_LAYER in item:
                layer_config = item[CONFIG_LAYER]
                this_layer = self._resolve_layer(len(hidden_layers), layer_config)
                hidden_layers.append(this_layer)

            else:
                raise ValueError(f"Architecture item must have '{CONFIG_BLOCK}' or '{CONFIG_LAYER}'")

        return hidden_layers
    
    def _resolve_metrics(self) -> None:

        def _parse_eval(
                eval_subsection: dict,
                subsection_name: str,
            ) -> tuple[str, dict]:
            class_name = eval_subsection.get(subsection_name)
            # currently, only 1 metric is supported (in addition to the loss)
            if not isinstance(class_name, str):
                error_msg = "TensorFlow model: Currently you can specify only one (string) metric, "
                error_msg += f"not multiple (array) for {CONFIG_SCORING}/{CONFIG_EVAL_METRIC}"
                raise ValueError(error_msg)
            class_name = class_name.lower().strip()
            class_params = eval_subsection.get("params", {})
            return class_name, class_params

        metric_class_subsection: dict | None = self.config.get(CONFIG_EVAL_METRIC)
        scoring_class_subsection: dict | None = self.config.get(CONFIG_SCORING)

        if metric_class_subsection and scoring_class_subsection:
            raise ValueError(f"Cannot specify both '{CONFIG_SCORING}' and '{CONFIG_EVAL_METRIC}', they are aliases")
        elif metric_class_subsection and metric_class_subsection.get(CONFIG_METRIC_CLASS):
            metric_class_name, metric_class_params = _parse_eval(metric_class_subsection, CONFIG_METRIC_CLASS)
        elif scoring_class_subsection and scoring_class_subsection.get(CONFIG_SCORING_CLASS):
            metric_class_name, metric_class_params = _parse_eval(scoring_class_subsection, CONFIG_SCORING_CLASS)
        else:
            metric_class_name, metric_class_params = None, {}

        if metric_class_name:
            if metric_class_name not in AVAILABLE_TF_METRICS:
                raise ValueError(f"Unsupported tensorflow metric: {metric_class_name}")
            else:

                metric_defined = AVAILABLE_TF_METRICS[metric_class_name](**metric_class_params)

                LOG.info(f"metric: {metric_defined}")

                self.metric_is_greater_is_better = get_metric_info(metric_class_name)['greater_is_better']
                self.metric = metric_defined

        return

    def _resolve_loss(self) -> None:

        loss_subsection: dict | None = self.config.get(CONFIG_LOSS)

        if not loss_subsection or not loss_subsection.get(CONFIG_LOSS_CLASS):
            raise ValueError(f"{CONFIG_LOSS} not set")

        loss_name = loss_subsection[CONFIG_LOSS_CLASS].lower().strip()

        if loss_name not in AVAILABLE_TF_LOSSES:
            raise ValueError(f"Unsupported tensorflow '{CONFIG_LOSS}': {loss_name}")

        loss_params = loss_subsection.get("params", {})

        loss_defined = AVAILABLE_TF_LOSSES[loss_name](**loss_params)

        LOG.info(f"'{CONFIG_LOSS}': {loss_defined}")

        self.loss = loss_defined

        return 
    
    def _resolve_optimizer(
            self,
        ) -> Optimizer:

        optimizer_section = self.config.get(CONFIG_OPTIMIZER)
        if not optimizer_section:
             raise ValueError(f"{CONFIG_OPTIMIZER} is required")

        optimizer_name = optimizer_section.get(CONFIG_OPTIMIZER_CLASS, OPTIMIZER_ADAM).lower().strip()
        if optimizer_name not in AVAILABLE_OPTIMIZERS:
            raise ValueError(f"Unsupported {CONFIG_OPTIMIZER}: {optimizer_name}")

        optimizer_params = optimizer_section.get("params", {})

        LOG.info(f"'{CONFIG_OPTIMIZER}': {optimizer_name}")

        return AVAILABLE_OPTIMIZERS[optimizer_name](**optimizer_params)

    def _build_model(
            self,
        ) -> Sequential:
        """
        Build the full Keras Sequential model.
        """
        model = Sequential()

        # first layer
        model.add(layers.Input(shape=(self.extra_params['number_used_features'],)))
        # hidden layers
        hidden_layers = self._resolve_hidden_layers()
        for layer in hidden_layers:
            model.add(layer)
        # output layer
        model.add(layers.Dense(
            self.number_outputs,
            activation=self.last_layer_activation,
            name='output_layer'
        ))

        # loss
        self._resolve_loss()

        # metric
        self._resolve_metrics()

        # optimizer
        self.optimizer = self._resolve_optimizer()

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])

        return model

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            **extra_params,
        ) -> None:

        ##########################################################
        # assuming validation set will be passed by 'eval_set',
        # to be compatible and standard through the framework
        ##########################################################

        validation_data: tuple | None = None
        call_back_early_stop: list[tfEarlyStopping] | None = None

        if CONFIG_EVAL_SET in extra_params:
            validation_data = (extra_params[CONFIG_EVAL_SET][0][0], extra_params[CONFIG_EVAL_SET][0][1])
            call_back_early_stop = [tfEarlyStopping(
                min_delta=extra_params.get(TF_MIN_DELTA, 0),
                patience=extra_params.get(TF_PATIENCE, 0),
            )]
            LOG.info(f"early stop set: {call_back_early_stop}")

        fit_result = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=call_back_early_stop,
        )

        self.history = fit_result
        self.epochs_completed = len(self.history.history['loss'])
        self.is_fitted_ = True

        self.find_optimal_threshold(X, y)

        return

    def predict(
            self,
            X: pd.DataFrame,
        ) ->  np.ndarray:
        # Serving endpoints may execute prediction in contexts where Keras `predict`
        # tries to iterate a tf.data.Dataset in graph mode. Using predict_on_batch
        # with normalized numpy input avoids that iteration path.
        model_input = self._to_model_input(X)
        try:
            raw_preds = self.model.predict_on_batch(model_input)
        except Exception as error:
            LOG.info(f"predict_on_batch failed, fallback to predict: {error}")
            raw_preds = self.model.predict(model_input, verbose=0)
        return self._to_numpy_output(raw_preds)

    @staticmethod
    def _to_model_input(
            X: pd.DataFrame | np.ndarray | Any,
        ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(copy=False)
        elif isinstance(X, pd.Series):
            arr = X.to_numpy(copy=False).reshape(-1, 1)
        elif isinstance(X, np.ndarray):
            arr = X
        elif hasattr(X, "to_numpy"):
            arr = np.asarray(X.to_numpy())
        else:
            arr = np.asarray(X)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr

    @staticmethod
    def _to_numpy_output(
            predictions: Any,
        ) -> np.ndarray:
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()
        return np.asarray(predictions)
    
    def find_optimal_threshold(
            self,
            X: pd.DataFrame,
            y: pd.Series,
        ) -> None:

        if not self.able_to_optimise_threshold:
            return
        if not self.optimise_threshold:
            return
        LOG.info("finding optimal threshold...")

        probabilities = self.predict(X)

        thresholds = np.arange(0.01, 1.00, 0.01)
        scores = []
        for thresh in thresholds:
            preds = (probabilities > thresh).astype(int).flatten()
            if self.metric:
                score = self.metric(y, preds)
                scores.append(score)
            else:
                score = accuracy_score(y, preds)
                scores.append(score)
        if scores:
            if self.metric_is_greater_is_better:
                best_idx = np.argmax(scores)
            else:
                best_idx = np.argmin(scores)
            self.optimal_threshold = thresholds[best_idx]
            LOG.info(f"threshold optimised: {self.optimal_threshold:.2f}")

        return

    def get_params(
            self,
            deep: bool = True,
        ) -> dict[str, Any]:
        LOG.info("getting params ... ")
        return {
            "config": self.config,
            "random_state": self.random_state,
            'params': self.params,
            'extra_params': self.extra_params,
        }

class TFClassifier(TFBaseModel, ClassifierMixin):
    def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:

        self.number_outputs: int = 1
        self.last_layer_activation: str = 'sigmoid'
        self.optimal_threshold: float = _DEFAULT_BINARY_CLASSIFICATION_THRESHOLD
        self.able_to_optimise_threshold: bool = True

        super().__init__(*args, **kwargs)

        return

    def predict(
            self,
            X: pd.DataFrame,
        ) -> np.ndarray:

        probabilities = super().predict(X)

        # to be compatible and standard through the framework
        preds = (probabilities > self.optimal_threshold).astype(int).flatten()

        LOG.info(f"threshold used: {self.optimal_threshold:.2f}")

        return preds

    def predict_proba(
            self,
            X: pd.DataFrame,
        ) -> np.ndarray:

        preds = super().predict(X)

        # to be compatible and standard through the framework
        scikit_shape = np.hstack([1 - preds, preds])

        return scikit_shape

class TFRegressor(TFBaseModel, RegressorMixin):
    def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:

        self.number_outputs: int = 1
        self.last_layer_activation: str | None = None
        self.able_to_optimise_threshold: bool = False

        super().__init__(*args, **kwargs)

        return
    
    def predict(
            self,
            X: pd.DataFrame,
        ) -> np.ndarray:

        preds = super().predict(X)

        # to be compatible and standard through the framework
        preds = preds.flatten()

        return preds
