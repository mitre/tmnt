"""
Copyright (c) 2019 The MITRE Corporation.
"""
import os
import logging
import inspect

__all__ = ['logging_config']

CONFIGURED = False

def get_level(ll):
    log_level = ll
    if isinstance(ll, int):
        return log_level
    elif ll.lower() == 'info':
        log_level = logging.INFO
    elif ll.lower() == 'debug':
        log_level = logging.DEBUG
    elif ll.lower() == 'error':
        log_level = logging.ERROR
    elif ll.lower() == 'warning':
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    return log_level

def logging_config(folder=None, name=None,
                   level=40,
                   console_level=40,
                   no_console=False):
    """ Config the logging.

    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level
    no_console: bool
        Whether to disable the console log
    Returns
    -------
    folder : str
        Folder that the logging file will be saved into.
    """
    global CONFIGURED
    CONFIGURED = True
    
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + '.log')
    log_level = get_level(level)
    console_level = get_level(console_level)
    logging.root.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(log_level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder
