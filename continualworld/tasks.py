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
    "CWUNIQUE20": [
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
        # CW10 ends
        "peg-insert-side-v1",
        "pick-place-v1",
        "basketball-v1",
        "bin-picking-v1",
        "stick-push-v1",
        "sweep-v1",
        "coffee-pull-v1",
        "faucet-open-v1",
        "box-close-v1",
        "door-unlock-v1",
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
