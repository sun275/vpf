#!/usr/bin/env python3
"""Load every public VPF task config without constructing datasets."""

from pathlib import Path

import yaml
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def main():
    config_roots = [
        ROOT / "tasks/detection/configs/coco",
        ROOT / "tasks/segmentation/configs/ade20k",
        ROOT / "tasks/autonomous_driving/configs/detection/bdd100k",
        ROOT / "tasks/autonomous_driving/configs/detection/acdc",
        ROOT / "tasks/autonomous_driving/configs/segmentation/acdc",
        ROOT / "tasks/autonomous_driving/configs/segmentation/cityscapes",
    ]
    python_configs = sorted(
        config_path
        for config_root in config_roots
        for config_path in config_root.glob("*.py")
    )
    for config_path in python_configs:
        Config.fromfile(config_path)

    yaml_configs = sorted((ROOT / "tasks/classification/configs").glob("*.yaml"))
    for config_path in yaml_configs:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if config.get("MODEL", {}).get("TYPE") != "vpf":
            raise ValueError(f"Unexpected model type in {config_path}")

    print(f"Loaded {len(python_configs)} Python configs and {len(yaml_configs)} YAML configs.")


if __name__ == "__main__":
    main()
