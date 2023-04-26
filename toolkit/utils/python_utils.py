def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def merge_dict_with_prefix(a, b, prefix='', include=(), exclude=()):
    """
    Merge dictionary `b` into dictionary `a`, adding a prefix to the keys in `b` and
    optionally including or excluding specific keys.

    Args:
        a (dict): The dictionary to merge into.
        b (dict): The dictionary to merge from.
        prefix (str, optional): The prefix to add to the keys in `b`. Defaults to ''.
        include (tuple, optional): Tuple of keys to include. Defaults to ().
        exclude (tuple, optional): Tuple of keys to exclude. Defaults to ().

    Returns:
        dict: The updated dictionary `a`.

    """
    if not include:
        include = b.keys()

    for k, v in b.items():
        if k in include and k not in exclude:
            a[f'{prefix}{k}'] = v

    return a
