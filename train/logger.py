import datetime
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

def set_up_log(args):
    # set exp time
    SHA_TZ = datetime.timezone(datetime.timedelta(hours=8),
                          name='Asia/Shanghai')   
    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
    exp_time = f'{beijing_now.year}-{beijing_now.month}-{beijing_now.day}-{beijing_now.hour}-{beijing_now.minute}'
    
    # set exp name
    if args.name is None:
        args.name = exp_time
        
    # step up log path
    log_path = os.path.join(args.log_dir, args.name)
    if os.path.exists(log_path):
        print(
            "WARNING. Experiment already exists."
        )
    os.makedirs(log_path, exist_ok=True)
    
    # record configs for reproducibility
    f_path = os.path.join(log_path, 'log.txt')
    with open(f_path, 'a') as f:
        configDict = args.__dict__
        f.write(f'{exp_time} \n Configs :\n')
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
    
    # checkpoint path
    checkpoint_dir = os.path.join(log_path, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # tensorboard path
    tensorboard_path = os.path.join(log_path, 'tensorboard_log')
    tb_writer = SummaryWriter(tensorboard_path)
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f'** LOG ** Save log to {log_path}')
    
    return checkpoint_dir, tb_writer, f_path