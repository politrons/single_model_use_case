from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml   # type: ignore # noqa
import numpy as np   # type: ignore # noqa

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
LOG = logging.getLogger("framework.utils")

class YamlUtils:
    @staticmethod
    def yaml_to_dict(value: Optional[str]) -> Dict[str, Any]:
        """Load YAML from:
        - a file path (supports dbfs:/ and /dbfs/), resolved against several bases,
        - OR an inline YAML string.
        Returns a dict (empty if None). Raises if YAML root is not a mapping.
        """
        if not value:
            return {}
        v = value.strip()
        # Candidate file paths to try (in order)
        candidates = []
        if v.startswith("dbfs:/"):
            candidates.append(Path("/dbfs/" + v[len("dbfs:/"):]))
        candidates.append(Path(v))                    # as given (CWD)
        candidates.append(Path.cwd() / v)             # explicit CWD join
        try:
            base_dir = Path(__file__).parent
            candidates.append(base_dir / v)           # relative to module file
        except Exception as e:
            LOG.error(f"Failed to load YAML: {e}")
            pass

        text: Optional[str] = None
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    LOG.info(f"Loading YAML from {p}")
                    text = p.read_text(encoding="utf-8")
                    break
            except Exception as e:
                LOG.error(f"Failed to load candidates YAML: {e}")
                continue

        if text is None:
            # Fallback: treat as inline YAML
            text = v

        try:
            data = yaml.safe_load(text)
        except Exception as e:
            raise SystemExit(f"Invalid YAML: {e}") from e

        if not isinstance(data, dict):
            LOG.warning(f"YAML must be a mapping (key: value pairs). {data}")
            return {}
        return data

class DatabaseUtils:
    @staticmethod
    def table_exists(spark, table: str) -> bool:
        try:
            spark.sql(f"SELECT count(*) FROM {table}")
            return True
        except Exception as e:
            LOG.info(f"Table {table} does not exist. {e}")
            return False

class ParserUtils:
    @staticmethod
    def parse_dictionary_params(
        user_input: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Deeply parse a config dictionary loaded from YAML.
        Converts strings like "np.int8" → np.int8, "(0.1, 0.9)" → tuple, etc.
        Returns a new dict with parsed values.
        """
        def parse_param_value(
                value: Any
            ) -> Any:
            if not isinstance(value, str):
                return value

            value = value.strip()

            match value:
                case "True":
                    return True
                case "False":
                    return False
                case "None":
                    return None
                case _ if value.startswith("np."):
                    try:
                        return eval(value, {"__builtins__": {}, "np": np})
                    except Exception as e:
                        raise ValueError(f"Invalid numpy expression in config: {value!r}") from e
                case _ if value.startswith("(") and value.endswith(")"):
                    try:
                        parsed = eval(value, {"__builtins__": {}})
                        return parsed  # eval already returns tuple/list/dict/etc.
                    except Exception:
                        return value  # fallback to string if not parsable
                case _ if value.startswith("{") and value.endswith("}"):
                    try:
                        parsed = eval(value, {"__builtins__": {}})
                        return parsed  # eval already returns tuple/list/dict/etc.
                    except Exception:
                        return value  # fallback to string if not parsable
                case _:
                    return value

        # Create a new config to avoid mutating the original
        parsed_config = {}
        for k, v in user_input.items():
            parsed_config[k] = parse_param_value(v)

        return parsed_config 
