import logging
import logging.config as config
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger.jsonlogger import JsonFormatter
import sys
import os 

def get_env_key_val(name: str, default: bool):
    return os.getenv(name) if os.getenv(name) else default

stdout_loglvl = get_env_key_val("STDOUT_LOG_LEVEL", logging.WARNING)
if not stdout_loglvl in logging._nameToLevel:
    stdout_loglvl = logging.DEBUG

print("Setting stdout loglvl to {}".format(stdout_loglvl))
file_loglvl = get_env_key_val("FILE_LOG_LEVEL", logging.DEBUG)
if not file_loglvl in logging._nameToLevel:
    file_loglvl = logging.DEBUG
print("Setting file loglvl to {}".format(file_loglvl))

base_directory = "logs"
dirname = os.path.dirname(os.path.abspath(__file__))
parent_folder_name = dirname.split('/')[-1] 
APP_NAME = parent_folder_name
APP_VERSION = "0.1"

def get_logger(name: str, add_stdout=True, add_file=True, default_log_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(stdout_loglvl)

    fmt = logging.Formatter('%(name)s | %(levelname)s  | %(asctime)s ｜ %(filename)s ｜ %(funcName)s ｜ %(lineno)s ｜ %(message)s')
    json_fmt = JsonFormatter('%(levelname)s %(message)s %(asctime)s %(name)s %(funcName)s %(lineno)d %(thread)d %(pathname)s', json_ensure_ascii=False)


    if add_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(stdout_loglvl)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    if add_file:
        # file_handler = logging.FileHandler(f"./logs_{name}.txt", encoding='utf-8')
        file_handler = TimedRotatingFileHandler(os.path.join(base_directory ,f"./logs_{name}.json"), when='d', interval=1, backupCount=60)
        file_handler.setLevel(file_loglvl)
        file_handler.setFormatter(json_fmt)
        logger.addHandler(file_handler)

    if not add_stdout and not add_file: 
        print("You should at least set one of the handlers to True")
        sys.exit(0)

    return logger