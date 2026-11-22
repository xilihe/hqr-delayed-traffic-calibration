"""Fixed, class-specific reference-quantile maps with unbounded linear tails.

These maps put historical class score distributions on a common coordinate.
Their usefulness under later drift is empirical; no unconditional risk guarantee
is implied by this normalization alone.
"""
import numpy as np


def linear_extend(values, x, y):
    values=np.asarray(values,dtype=np.float64);x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if values.ndim!=1 or len(x)<2 or len(x)!=len(y) or not np.all(np.diff(x)>0):
        raise ValueError('Invalid monotone interpolation inputs')
    output=np.interp(values,x,y)
    left=values<x[0];right=values>x[-1]
    output[left]=y[0]+(values[left]-x[0])*(y[1]-y[0])/(x[1]-x[0])
    output[right]=y[-1]+(values[right]-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])
    return output


def fit_maps(scores, prediction, classes, probabilities, prior_count):
    scores=np.asarray(scores,dtype=np.float64);prediction=np.asarray(prediction)
    p=np.asarray(probabilities,dtype=np.float64)
    if len(scores)<20 or not np.isfinite(scores).all() or prior_count<=0 or not np.all((p>0)&(p<1)) or not np.all(np.diff(p)>0):
        raise ValueError('Invalid reference-fit sample or quantile anchors')
    if len(scores)!=len(prediction) or not np.isin(prediction,classes).all():raise ValueError('Invalid source prediction groups')
    global_x=np.quantile(scores,p,method='linear');target=np.log(p/(1-p));result={}
    for c in classes:
        values=scores[prediction==c];n=len(values);weight=n/(n+prior_count)
        x=weight*np.quantile(values,p,method='linear')+(1-weight)*global_x if n else global_x.copy()
        unique,inverse=np.unique(x,return_inverse=True)
        z=np.bincount(inverse,weights=target)/np.bincount(inverse)
        degenerate=len(unique)<2
        if degenerate:
            scale=max(float(np.ptp(scores)),1.0);center=float(unique[0])
            unique=np.asarray([center-scale,center+scale]);z=np.asarray([-1.,1.])
        if not np.all(np.diff(z)>0):raise ValueError('Reference coordinate is not strictly monotone')
        result[str(c)]=dict(x=unique.tolist(),z=z.tolist(),source_rows=n,source_weight=weight,degenerate=degenerate)
    return result


def transform(scores, prediction, maps):
    scores=np.asarray(scores,dtype=np.float64);prediction=np.asarray(prediction)
    output=np.empty_like(scores);covered=np.zeros(len(scores),dtype=bool)
    for c,mapping in maps.items():
        selected=prediction==int(c)
        output[selected]=linear_extend(scores[selected],mapping['x'],mapping['z']);covered|=selected
    if not covered.all() or not np.isfinite(output).all():raise ValueError('Reference transform left invalid scores')
    return output


def fit_shift_maps(scores, prediction, classes, probability, prior_count):
    """Additive conditional-quantile rectification control (historical only)."""
    scores = np.asarray(scores, dtype=np.float64)
    prediction = np.asarray(prediction)
    if len(scores) < 20 or len(scores) != len(prediction) or not np.isfinite(scores).all():
        raise ValueError('Invalid historical scores')
    if not 0 < probability < 1 or prior_count <= 0 or not np.isin(prediction, classes).all():
        raise ValueError('Invalid rectification settings')
    shared = float(np.quantile(scores, probability, method='linear'))
    maps = {}
    for c in classes:
        local = scores[prediction == c]; n = len(local); weight = n/(n+prior_count)
        center = weight*float(np.quantile(local, probability, method='linear'))+(1-weight)*shared if n else shared
        maps[str(c)] = dict(x=[center-1, center+1], z=[-1., 1.], source_rows=n, source_weight=weight, degenerate=False)
    return maps
