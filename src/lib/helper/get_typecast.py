def get_typecast(cfg, *keys, cast_fn=str, default=None, verbose=False):
    raw = cfg
    for k in keys:
        if isinstance(raw, dict):
            raw = raw.get(k, None)
        else:
            raw = None
        if raw is None:
            break
    try:
        return cast_fn(raw)
    except Exception:
        if verbose:
            print(f"⚠️  Failed to cast {keys}={raw!r}, defaulting to {default!r}")
        return default