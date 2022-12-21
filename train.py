import json
import os
import argparse
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Model, context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
import mindspore.communication.management as D
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from elmo.data.reader import create_elmo_dataset
from elmo.model import LanguageModel
from ElmoTrainOne import ElmoTrainOnestepWithLoss
from elmo.utils.util import LossCallBack

parser = argparse.ArgumentParser(description="elmo")
parser.add_argument('--data_url', default='./dataset/train.mindrecord', help='Location of data.')
parser.add_argument('--train_url', default='./ckpt', help='Location of training outputs.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--lr', type=float, default=0.2, help='learning rate (default: 0.2)')
parser.add_argument('--epoch_num', type=int, default=1, help='epoch_num, default is 1')
parser.add_argument('--sink_size', type=int, default=100, help='Sink size for every iteration, default is 100')
args = parser.parse_args()

def train():
    set_seed(0)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    device_num = int(os.getenv('DEVICE_NUM'))
    device_id = int(os.getenv('DEVICE_ID'))
    if args.device_target == 'Ascend':
        print('id:', device_id, device_num)
        context.set_context(device_id=device_id)
        if device_num > 1:
            D.init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    elif args.device_target == "GPU":
        if device_num > 1:
            D.init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                gradients_mean=True)

    options_file = 'dataset/options.json'
    with open(options_file, 'r') as fin:
        options = json.load(fin)

    lm = LanguageModel(options=options, training=True)
    opt = nn.Adagrad(lm.trainable_params(), learning_rate=args.lr)
    
    dataset = create_elmo_dataset(batch_size=options['batch_size'], data_file_path=args.data_url)

    steps_per_epoch = dataset.get_dataset_size()
    #callback_size = opt.sink_size
    #actual_epoch_num = int(args.epoch_num * steps_per_epoch / callback_size)

    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="elmo", directory=args.train_url, config=config_ck)

    callback = [LossCallBack(steps_per_epoch), TimeMonitor(steps_per_epoch), ckpoint_cb]
    update_scale_cell = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    train_one_step = ElmoTrainOnestepWithLoss(lm, opt, update_scale_cell)
    model = Model(train_one_step)
    model.train(args.epoch_num, dataset, callbacks=callback)

if __name__=='__main__':
    train()