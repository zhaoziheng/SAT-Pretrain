import os

def is_ddp():
    if "RANK" in os.environ:
        return True
    else:
        return False
    
def is_master():
    if int(os.environ["RANK"]) == 0:
        return True
    else:
        return False