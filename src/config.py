"""
Loads config/settings.yaml and the active prompt from config/prompts.yaml.

The prompt version can be overridden without editing files:
    PowerShell:  $env:PROMPT_VERSION = "answer_v2"; python app.py
"""
import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def _load_yaml(name):
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings():
    return _load_yaml("settings.yaml")


def load_prompt(version=None):
    """Returns (version, template). Fails loudly on a missing version or placeholder."""
    data = _load_yaml("prompts.yaml")
    version = version or os.getenv("PROMPT_VERSION") or data["active"]
    prompts = data.get("prompts", {})
    if version not in prompts:
        raise ValueError(f"Prompt version '{version}' not found in config/prompts.yaml. "
                         f"Available: {', '.join(prompts)}")
    template = prompts[version]["template"]
    for placeholder in ("{context}", "{question}"):
        if placeholder not in template:
            raise ValueError(f"Prompt '{version}' is missing the {placeholder} placeholder.")
    return version, template


SETTINGS = load_settings()
PROMPT_VERSION, PROMPT_TEMPLATE = load_prompt()