def load_model_package(name):
    if name == "baseline":
        from models.baseline import parameters as p
        from models.baseline import esn
    elif name == "motion_only":
        from models.motion_only import parameters as p
        from models.motion_only import esn
    elif name == "timeindexed":
        from models.timeindexed import parameters as p
        from models.timeindexed import esn
    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Choose: baseline, motion_only, timeindexed"
        )

    return p, esn
