"""Reuse the established signed-IAT QUIC representation and source-only scaling."""
import numpy as np


def canonical_inputs(ppi, lengths):
    """Final channels: log IPT, direction, log payload size, valid mask."""
    ppi = np.asarray(ppi)
    lengths = np.asarray(lengths)
    if ppi.shape != (len(lengths), 3, 30):
        raise ValueError('Expected exact PPI[N,3,30] source schema')
    if np.any(lengths != lengths.astype(int)) or np.any((lengths < 2) | (lengths > 30)):
        raise ValueError('Invalid PPI prefix length')
    mask = np.arange(30)[None, :] < lengths[:, None]
    ipt, direction, size = ppi[:, 0], ppi[:, 1], ppi[:, 2]
    if (not np.isfinite(ppi).all() or np.any(size[mask] < 0)
            or not np.isin(direction[mask], [-1, 1]).all()
            or np.any(ipt[:, 0] != 0)):
        raise ValueError('Malformed valid PPI values')
    # No clipping or fitted scalers. Padding has one canonical representation.
    out = np.zeros((len(ppi), 4, 30), dtype='<f4')
    out[:, 0] = np.where(mask, np.sign(ipt) * np.log1p(np.abs(ipt.astype(np.float64))) / np.log1p(60000.), 0)
    out[:, 1] = np.where(mask, direction, 0)
    out[:, 2] = np.where(mask, np.log1p(size.astype(np.float64)) / np.log1p(1500.), 0)
    out[:, 3] = mask
    out[out == 0] = 0  # Normalize signed zero before byte-level hashing.
    if not np.isfinite(out).all():
        raise ValueError('Non-finite final input')
    return out


def fit_preprocessor(train_inputs):
    mask=train_inputs[:,3].astype(bool)
    mean=np.zeros(4); scale=np.ones(4)
    for c in [0,2]:
        v=train_inputs[:,c][mask]; mean[c]=v.mean(); scale[c]=v.std()
        if scale[c]==0:scale[c]=1
    return {'mean':mean.tolist(),'scale':scale.tolist(),'fitted_on':'training_sample_only'}


def transform(inputs,params):
    result=inputs.copy(); mask=inputs[:,3].astype(bool)
    for c in [0,2]:result[:,c]=np.where(mask,(inputs[:,c]-params['mean'][c])/params['scale'][c],0)
    if not np.isfinite(result).all():raise ValueError('Nonfinite transformed input')
    return result


