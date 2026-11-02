"""Pure python scoring and treeshap for an exported lightgbm model.

lightgbm itself needs openmp installed on the machine (brew install libomp on a
mac, usually missing on serverless hosts), so at runtime we walk the trees
ourselves. tests/test_trees.py checks this against lightgbm's own numbers.
"""
import json
import math


def load_trees(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _go_left(node, x):
    v = x[node["f"]]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return node["nan_left"]
    return v <= node["t"]


def raw_score(trees, x):
    total = 0.0
    for tree in trees:
        node = tree
        while "v" not in node:
            node = node["l"] if _go_left(node, x) else node["r"]
        total += node["v"]
    return total


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _count(node):
    return node["n"]


def _extend(path, zero_frac, one_frac, feature):
    path.append([feature, zero_frac, one_frac, 1.0 if not path else 0.0])
    depth = len(path) - 1
    for i in range(depth - 1, -1, -1):
        path[i + 1][3] += one_frac * path[i][3] * (i + 1) / (depth + 1)
        path[i][3] = zero_frac * path[i][3] * (depth - i) / (depth + 1)


def _unwind(path, i):
    depth = len(path) - 1
    one, zero = path[i][2], path[i][1]
    keep = path[depth][3]
    for j in range(depth - 1, -1, -1):
        if one != 0:
            hold = path[j][3]
            path[j][3] = keep * (depth + 1) / ((j + 1) * one)
            keep = hold - path[j][3] * zero * (depth - j) / (depth + 1)
        else:
            path[j][3] = path[j][3] * (depth + 1) / (zero * (depth - j))
    for j in range(i, depth):
        path[j][0], path[j][1], path[j][2] = path[j + 1][0], path[j + 1][1], path[j + 1][2]
    path.pop()


def _unwound_sum(path, i):
    depth = len(path) - 1
    one, zero = path[i][2], path[i][1]
    keep = path[depth][3]
    total = 0.0
    for j in range(depth - 1, -1, -1):
        if one != 0:
            part = keep * (depth + 1) / ((j + 1) * one)
            total += part
            keep = path[j][3] - part * zero * (depth - j) / (depth + 1)
        else:
            total += path[j][3] / (zero * (depth - j) / (depth + 1))
    return total


def _shap_walk(node, x, phi, path, zero_frac, one_frac, feature):
    path = [p[:] for p in path]
    _extend(path, zero_frac, one_frac, feature)

    if "v" in node:
        for i in range(1, len(path)):
            phi[path[i][0]] += _unwound_sum(path, i) * (path[i][2] - path[i][1]) * node["v"]
        return

    hot, cold = (node["l"], node["r"]) if _go_left(node, x) else (node["r"], node["l"])
    total = _count(node)
    seen_zero, seen_one = 1.0, 1.0

    # same feature further up this path: undo it before splitting again
    for i in range(1, len(path)):
        if path[i][0] == node["f"]:
            seen_zero, seen_one = path[i][1], path[i][2]
            _unwind(path, i)
            break

    _shap_walk(hot, x, phi, path, seen_zero * _count(hot) / total, seen_one, node["f"])
    _shap_walk(cold, x, phi, path, seen_zero * _count(cold) / total, 0.0, node["f"])


def contributions(trees, x, width):
    """Per feature contribution to the raw score, for one row."""
    phi = [0.0] * width
    for tree in trees:
        if "v" not in tree:
            _shap_walk(tree, x, phi, [], 1.0, 1.0, -1)
    return phi
