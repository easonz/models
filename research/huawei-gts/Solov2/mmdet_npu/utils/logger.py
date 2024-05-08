import logging
# from ..mmcv_npu.runner import get_dist_info
def get_dist_info():
    # if torch.__version__ < '1.0':
    #     initialized = dist._initialized
    # else:
    #     if dist.is_available():
    #         initialized = dist.is_initialized()
    #     else:
    #         initialized = False
    # if initialized:
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()
    # else:
    #     rank = 0
    #     world_size = 1
    # return rank, world_size
    return 1,1
def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(__name__.split('.')[0])  # i.e., mmdet
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger