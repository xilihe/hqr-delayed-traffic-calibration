"""Resolve a study configuration and its isolated output directory."""
import json
from pathlib import Path


def local_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Study paths must remain inside the project')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Study path resolves outside the project')
    return path


def study_config(root, relative='configs/pilot-v1.json'):
    path = local_path(root, relative)
    config = json.loads(path.read_text())
    run = config.get('output_run', 'pilot-v1')
    if not isinstance(run,str) or Path(run).name != run or run in {'','.', '..'}:
        raise ValueError('Output run must be one directory name')
    return path, config, root/'outputs'/run
