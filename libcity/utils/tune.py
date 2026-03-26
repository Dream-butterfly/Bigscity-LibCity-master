from contextlib import contextmanager


@contextmanager
def checkpoint_dir(step=None):
    yield None


def report(**kwargs):
    return None
