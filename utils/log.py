import logging, time, os


def get_logger_fgnet(filename, ver=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(asctime)s  %(message)s",
                              "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[ver])
 
    local_time = time.localtime()
    file_time =  time.strftime("%Y-%m-%d_%H:%M:%S", local_time)
    
    if not os.path.exists("./logs/for_LOPO_fgnet/"+file_time):
        os.makedirs("./logs/for_LOPO_fgnet/"+file_time)
    
    path = "./logs/for_LOPO_fgnet/" + file_time + "/" + filename 
 
    fh = logging.FileHandler(path, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    return logger

def get_logger(filename, ver=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(asctime)s  %(message)s",
                              "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[ver])
 
    local_time = time.localtime()
    file_time =  time.strftime("%Y-%m-%d_%H:%M:%S", local_time)
    
    if not os.path.exists("./logs/"+file_time):
        os.makedirs("./logs/"+file_time)
    
    path = "./logs/" + file_time + "/" + filename 
 
    fh = logging.FileHandler(path, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    return logger