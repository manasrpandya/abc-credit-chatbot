import json


def compact_trees(booster):
    """Turn a lightgbm booster into the small nested dicts creditbot.trees walks."""
    dump = booster.dump_model()

    def walk(node):
        if "leaf_value" in node:
            return {"v": node["leaf_value"], "n": node.get("leaf_count", 1)}
        if node["decision_type"] != "<=":
            raise ValueError("expected numeric splits only")
        t = node["threshold"]
        if node["missing_type"] == "NaN":
            nan_left = node["default_left"]
        else:
            # lightgbm treats nan as 0 on features that had none in training
            nan_left = 0.0 <= t
        return {"f": node["split_feature"], "t": t, "nan_left": nan_left,
                "n": node["internal_count"],
                "l": walk(node["left_child"]), "r": walk(node["right_child"])}

    return [walk(tree["tree_structure"]) for tree in dump["tree_info"]]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
