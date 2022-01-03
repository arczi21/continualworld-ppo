TASK_SEQS = {
    "CW10": [
        "hammer-v1",
        "push-wall-v1",
        "faucet-close-v1",
        "push-back-v1",
        "stick-pull-v1",
        "handle-press-side-v1",
        "push-v1",
        "shelf-place-v1",
        "window-close-v1",
        "peg-unplug-side-v1",
    ],
}

SUPPLEMENTARY_TRIPLETS = [
    # Good ABC triplets
    ['push-v1', 'window-close-v1', 'hammer-v1'],
    ['hammer-v1', 'window-close-v1', 'faucet-close-v1'],
    # ??? ABC triplets
    ['stick-pull-v1', 'push-back-v1', 'push-wall-v1'],
    ['push-wall-v1', 'shelf-place-v1', 'push-back-v1'],
    ['faucet-close-v1', 'shelf-place-v1', 'push-back-v1'],
    ['stick-pull-v1', 'peg-unplug-side-v1', 'stick-pull-v1'],
    ['window-close-v1', 'handle-press-side-v1', 'peg-unplug-side-v1'],
    ['faucet-close-v1', 'shelf-place-v1', 'peg-unplug-side-v1'],
    # Bad triplets
    ['peg-unplug-side-v1', 'push-v1', 'push-wall-v1'],
    ['window-close-v1', 'hammer-v1', 'push-v1'],
    ['push-back-v1', 'push-wall-v1', 'push-v1'],
]

TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]
