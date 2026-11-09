"""Terminal-only progress indicators; these settings do not affect forecasts."""

import sys

from tqdm import tqdm

import experiment_config as cfg


def progress(iterable=None, **options):
    """Use live bars in terminals and keep redirected logs free of control codes."""
    return tqdm(iterable, disable=not cfg.SHOW_PROGRESS or not sys.stderr.isatty(),
                dynamic_ncols=True, mininterval=cfg.PROGRESS_REFRESH_SECONDS, **options)


def status(message):
    """Print a persistent message without overwriting active progress bars."""
    tqdm.write(message)
