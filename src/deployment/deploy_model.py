from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional
from mlflow.tracking import MlflowClient # type: ignore

@dataclass(frozen=True)
class Config:
    env: str                      # dev | staging | prod (only for logging/context)
    model_uri: Optional[str]      # models:/<name>/<version>
    dependency_task_key: Optional[str]  # if model_uri is not provided, read from this upstream task

def deploy(model_uri: str, env: str) -> None:
    """Assign 'champion' alias to a registered model version (notebook behavior).
    - If alias 'champion' is newly set, remove 'challenger' if present.
    """
    print(f"Deployment running in env: {env}")
    # Expect models:/<name>/<version> -> ["models:", "<name>", "<version>"]
    _, model_name, version = model_uri.split("/")

    client = MlflowClient(registry_uri="databricks")
    mv = client.get_model_version(model_name, version)

    target_alias = "champion"
    aliases = set(mv.aliases or [])
    if target_alias not in aliases:
        client.set_registered_model_alias(name=model_name, alias=target_alias, version=version)
        print(f"Assigned alias '{target_alias}' to model version {model_uri}.")

        # remove "challenger" alias if assigning "champion" alias
        if "challenger" in aliases:
            print(f"Removing 'challenger' alias from model version {model_uri}.")
            client.delete_registered_model_alias(name=model_name, alias="challenger")
    else:
        print(f"Alias '{target_alias}' already set on {model_uri}; skipping.")

def _resolve_model_uri(cli_model_uri: str, dependency_task_key: Optional[str]) -> str:
    """Minimal helper: read model_uri from upstream task if not provided."""
    model_uri = (cli_model_uri or "").strip()
    if model_uri:
        return model_uri
    if dependency_task_key:
        try:
            from mlflow.utils.databricks_utils import dbutils  # type: ignore
            model_uri = dbutils.jobs.taskValues.get(dependency_task_key, "model_uri", debugValue="")
        except Exception:
            model_uri = ""
    if not model_uri:
        raise ValueError("model_uri must be provided via --model_uri or available in task values.")
    return model_uri

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal deployment (notebook-equivalent)")
    parser.add_argument("--env", default="dev")
    parser.add_argument("--model_uri", default="")
    parser.add_argument("--dependency_task_key", default="Training")
    args = parser.parse_args(argv)

    cfg = Config(env=args.env, model_uri=args.model_uri, dependency_task_key=args.dependency_task_key)
    model_uri = _resolve_model_uri(cfg.model_uri or "", cfg.dependency_task_key)
    deploy(model_uri, cfg.env)

    print("\n=== Deployment Complete ===")
    print(f"Model URI: {model_uri}")
    print(f"Environment: {cfg.env}")
    return 0

if __name__ == "__main__":
    main()
