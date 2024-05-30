import functools
from mindspore.communication import get_rank

def get_dist_info():
    try:
        rank = get_rank()
    except:
        rank = 0
    return rank

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper