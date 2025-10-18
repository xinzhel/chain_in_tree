import os
import logging
def setup_logging(run_id, root_dir, add_console_handler=False, verbose=False):
    os.makedirs(root_dir, exist_ok=True)
    log_path = os.path.join(root_dir, f"logging_{run_id}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Create file handler
    log_path = os.path.join(root_dir, "mcts_debug.log")
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Formatters
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    # Create console handler if needed
    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger