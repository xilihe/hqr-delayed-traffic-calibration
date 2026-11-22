"""Verify an internally frozen confirmation protocol before accessing test rows."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_protocol(root, relative='research/frozen/confirmation-v1-seal.json'):
    root = root.resolve()
    seal_path = root/relative
    seal = json.loads(seal_path.read_text())
    if seal['status'] != 'INTERNAL_PROTOCOL_FROZEN_BEFORE_CONFIRMATION':
        raise ValueError('Confirmation protocol is not frozen')
    for relative, expected in seal['files'].items():
        path = root/relative
        if not path.resolve().is_relative_to(root) or sha256(path) != expected:
            raise ValueError('Frozen implementation or evidence differs: '+relative)
    protocol_path = root/seal['protocol']
    if sha256(protocol_path) != seal['protocol_sha256']:
        raise ValueError('Confirmation protocol identity differs')
    config = json.loads(protocol_path.read_text())
    if config['confirmation_start'] != '20241001' or config['confirmation_end'] != '20250531':
        raise ValueError('Unexpected confirmation date range')
    return config, seal
