import h5py
import copy
import ast
import sys
import argparse
import numpy as np
import math
from sklearn.utils import shuffle
from random import shuffle as pyshuffle
import random
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter
import gc 
################################################################
## reco e+ KE using npe and first hit time map
################################################################
if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)


def init_weights(m):
    '''
    if type(m) == nn.Linear:
        #torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight,a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        m.bias.data.fill_(0.01)
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def load_data(data, max_n):
    d = h5py.File(data, 'r')
    max_evt = max_n if (max_n > 0 and max_n < d['data'].shape[0]) else d['data'].shape[0]
    df       = d['data' ][:max_evt,] 
    df[:,:,:,1]  = np.power(df[:,:,:,1], np.array([0.3]))## X^{alpha}
    df_label = d['label'][:max_evt,] 
    df_label[:,6] = df_label[:,6]/10
    d.close()
    return df, df_label
    

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers



class Vgg(torch.jit.ScriptModule):
    __constants__ = ['features','fcs']
    def __init__(self, in_channels, features_cfg, fcs_cfg):
        super(Vgg, self).__init__()
        mod_list = []
        mod_list += make_layers(features_cfg, in_channels, batch_norm=False)
        self.features = nn.ModuleList(mod_list)
        fcs_list = []
        for i in range(len(fcs_cfg)):
            if i == 0:
                fcs_list.append(  nn.Linear(512 * 7 * 3+1, fcs_cfg[i]) )# add total.p.e
                fcs_list.append( nn.ReLU(True) )
            elif i== (len(fcs_cfg)-1):
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
            else:
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
        self.fcs = nn.ModuleList(fcs_list)


    @torch.jit.script_method
    def forward(self, x):
        tot_pe = torch.sum(x[:,:,:,0:1], dim=(1,2), keepdim=False )/1000  
        x = x.transpose(1,3)
        y = x
        for i in self.features:
            y = i(y)
        y = y.view(y.size(0), -1)
        y = torch.cat((y,tot_pe),1)## add tot pe
        for i in self.fcs:
            y = i(y)
        return y


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run MDN training. '
        'Sensible defaults come from https://github.com/taboola/mdn-tensorflow-notebook-example/blob/master/mdn.ipynb',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--datafile', action='store', type=str,
                        help='HDF5 file paths')
    parser.add_argument('--valid_datafile', action='store', type=str,
                        help='HDF5 file paths')
    parser.add_argument('--test_datafile', action='store', type=str,
                        help='HDF5 file paths')
    parser.add_argument('--pmt_pos_file', action='store', type=str,
                        help='pmt_pos_file')
    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch_size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--test_batch', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--act_mode', action='store', type=int, default=0,
                        help='activation for last layer')
    parser.add_argument('--opt_mode', action='store', type=int, default=0,
                        help='optimizer')
    parser.add_argument('--early_stop_interval', action='store', type=int, default=10,
                        help='early_stop_interval')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output_dir file paths')
    parser.add_argument('--saveCkpt', action='store', type=ast.literal_eval, default=False,
                        help='save ckpt file')
    parser.add_argument('--savePb', action='store', type=ast.literal_eval, default=False,
                        help='save pb file')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False,
                        help='ckpt file paths')
    parser.add_argument('--Restore_opt', action='store', type=ast.literal_eval, default=True,
                        help='ckpt file paths')
    parser.add_argument('--restore_ckpt_path', action='store', type=str,
                        help='restore_ckpt_path ckpt file paths')
    parser.add_argument('--use_uniform', action='store', type=ast.literal_eval, default=True,
                        help='use uniform noise')
    parser.add_argument('--UseDDP', action='store', type=ast.literal_eval, default=False,
                        help='UseDDP')
    parser.add_argument('--produceEvent', action='store', type=int,
                        help='produceEvent')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt_path file paths')
    parser.add_argument('--log_out', action='store', type=str,
                        help='log_out file paths')
    parser.add_argument('--outFileName', action='store', type=str,
                        help='outFileName file paths')
    parser.add_argument('--pb_file_path', action='store', type=str,
                        help='pb_file_path file paths')
    parser.add_argument('--validation_file', action='store', type=str,
                        help='validation_file file paths')
    parser.add_argument('--test_file', action='store', type=str,
                        help='test_file file paths')
    parser.add_argument('--tag', action='store', type=str,
                        help='tag')
    parser.add_argument('--norm_mode', action='store', type=int, default=0,
                        help='mode of normalize.')
    parser.add_argument('--num_trials', action='store', type=int, default=40,
                        help='num of trials.')
    parser.add_argument('--max_training_evt', action='store', type=int, default=1000,
                        help='max_training_evt.')
    parser.add_argument('--local_rank', action='store', type=int, default=0,
                        help='local_rank')
    parser.add_argument('--Alphas', action='store', type=float, default=0.1,
                        help='Alphas for tsp')
    parser.add_argument('--Norm', action='store', type=int, default=2,
                        help='Norm for tsp')
    parser.add_argument('--kNN', action='store', type=int, default=1,
                        help='kNN for tsp')

    parser.add_argument('--doTraining', action='store', type=ast.literal_eval, default=False,
                        help='doTraining')
    parser.add_argument('--doValid', action='store', type=ast.literal_eval, default=False,
                        help='doValid')
    parser.add_argument('--doTest', action='store', type=ast.literal_eval, default=False,
                        help='doTest')
    parser.add_argument('--useFastLoss', action='store', type=ast.literal_eval, default=False,
                        help='useFastLoss')
    parser.add_argument('--LossType', action='store', type=int, default=0,
                        help='LossType')
    parser.add_argument('--saveToCPU', action='store', type=ast.literal_eval, default=False,
                        help='saveToCPU')
    parser.add_argument('--pt_file_path', action='store', type=str,
                        help='pt_file_path file paths')
    parser.add_argument('--outFilePath', action='store', type=str,
                        help='outFilePath file paths')





    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    return parser

def print_mem(in_devices):
    for dev in in_devices:
        total_memory = torch.cuda.get_device_properties(dev).total_memory
        total_memory = total_memory/math.pow(1024,3)
        max_memory = torch.cuda.max_memory_allocated(dev)
        max_memory = max_memory/math.pow(1024,3)
        allocated_memory = torch.cuda.memory_allocated(dev)
        allocated_memory = allocated_memory/math.pow(1024,3)
        cached_memory = torch.cuda.memory_cached(dev)
        cached_memory = cached_memory/math.pow(1024,3)
        print('dev %d, total=%f GB, max=%f GB, allocated=%f GB, cached=%f GB'%(in_devices.index(dev), total_memory, max_memory, allocated_memory, cached_memory) )
        logger.info('')
 

if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()
    UseDDP        = parse_args.UseDDP
    print('is CUDA available=',torch.cuda.is_available())
    if UseDDP:
        torch.distributed.init_process_group(backend="nccl")
    #####################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    devices = []
    for i in range( torch.cuda.device_count() ):
        tmp_device = torch.device("cuda:%d"%i)
        devices.append(tmp_device)
    #####################################
    epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    test_batch = parse_args.test_batch
    act_mode = parse_args.act_mode
    opt_mode = parse_args.opt_mode
    early_stop_interval = parse_args.early_stop_interval
    datafile = parse_args.datafile
    valid_datafile = parse_args.valid_datafile
    test_datafile = parse_args.test_datafile
    pmt_pos_file  = parse_args.pmt_pos_file
    output_dir = parse_args.output_dir
    saveCkpt  = parse_args.saveCkpt
    ckpt_path  = parse_args.ckpt_path
    log_out    = parse_args.log_out
    savePb    = parse_args.savePb
    Restore   = parse_args.Restore
    Restore_opt  = parse_args.Restore_opt
    restore_ckpt_path = parse_args.restore_ckpt_path
    doTraining    = parse_args.doTraining
    doValid       = parse_args.doValid
    doTest        = parse_args.doTest
    useFastLoss   = parse_args.useFastLoss
    LossType      = parse_args.LossType
    saveToCPU     = parse_args.saveToCPU
    pt_file_path  = parse_args.pt_file_path
    outFilePath       = parse_args.outFilePath
    use_uniform   = parse_args.use_uniform
    norm_mode     = parse_args.norm_mode
    produceEvent = parse_args.produceEvent
    outFileName = parse_args.outFileName
    pb_file_path = parse_args.pb_file_path
    validation_file = parse_args.validation_file
    test_file       = parse_args.test_file
    num_trials      = parse_args.num_trials
    max_training_evt   = parse_args.max_training_evt
    tag           = parse_args.tag
    Init_weights = True
    #####################################
    cfg = {
        'A': [64    , 'M', 128     , 'M', 256, 256          , 'M', 512, 512          , 'M', 512, 512          , 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256          , 'M', 512, 512          , 'M', 512, 512          , 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256     , 'M', 512, 512, 512     , 'M', 512, 512, 512     , 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    # set up all the logging stuff
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )

    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)
    #####################################
    logger.info('constructing graph')
    net0 = Vgg(in_channels=2, features_cfg=cfg['B'], fcs_cfg=[2048, 512, 1])
    if Init_weights:
        net0.apply(init_weights)
    logger.info('create net0')
    net0.cuda()
    logger.info('net0 to cuda')
    print('net0=',net0)
    netDDP = None
    if UseDDP:
        netDDP = DDP(net0, find_unused_parameters=True) # device_ids will include all GPU devices by default
    logger.info('net0 to DDP')
    if Restore:
        print('model restored from ',restore_ckpt_path)
        checkpoint = torch.load(restore_ckpt_path)
        netDDP.load_state_dict(checkpoint['model_state_dict'])
    net = net0
    if UseDDP:
        net = netDDP
    print('net=',net)
    print(net.parameters())
    scale_factor = 0.5
    optimizer = optim.Adam(net.parameters(), lr=0.00092966)## baseline
    criteria = nn.L1Loss()
    if Restore and Restore_opt:
        print('opt. restored from ',restore_ckpt_path)
        checkpoint = torch.load(restore_ckpt_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, threshold=0.0)
    logger.info('constructing optimizer')
    ########### for training dataset #############################
    datasets = []
    inFile = open(datafile,'r')
    lines = inFile.readlines()
    for line in lines:
        if '#' in line:continue
        tmp_data = line.replace('\n','') 
        tmp_data = tmp_data.replace(' ','') 
        datasets.append(tmp_data)
        tmp_df = h5py.File(tmp_data, 'r')
    inFile.close()
    print('len datasets=',len(datasets))
    #############for valid dataset ###########################
    datasets_valid = []
    inFile = open(valid_datafile,'r')
    lines = inFile.readlines()
    for line in lines:
        if '#' in line:continue
        tmp_data = line.replace('\n','') 
        tmp_data = tmp_data.replace(' ','') 
        datasets_valid.append(tmp_data)
        tmp_df = h5py.File(tmp_data, 'r')
    inFile.close()
    print('len valid datasets=',len(datasets_valid))
    #############for testing dataset ###########################
    datasets_test = []
    inFile = open(test_datafile,'r')
    lines = inFile.readlines()
    for line in lines:
        if '#' in line:continue
        tmp_data = line.replace('\n','') 
        tmp_data = tmp_data.replace(' ','') 
        datasets_test.append(tmp_data)
        tmp_df = h5py.File(tmp_data, 'r')
    inFile.close()
    print('len test datasets=',len(datasets_test))
    ########################################
    print_mem(devices)
    if doTraining:
        logger.info('Start training model')
        cost_list = []
        loss_previous = None
        for epoch in range(epochs):
            print('epoch=%d'%(epoch))
            logger.info('')
            total_cost = None
            count = 0
            val_total_cost = None
            val_count = 0
            pyshuffle(datasets)
            for dname in datasets:
                ########################################
                df_, df_label_ = load_data(dname, -1)
                for idf in range(0,df_.shape[0], batch_size):
                    ########################################
                    inputs = df_[idf:idf+batch_size,]
                    # forward pass
                    optimizer.zero_grad()
                    outputs = net((torch.Tensor(inputs)).cuda().requires_grad_(False))
                    outputs = np.squeeze(outputs)
                    labels = torch.Tensor( df_label_[idf:idf+batch_size,6] ).cuda()
                    loss = criteria(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss_c = loss.item()
                    total_cost = total_cost + loss_c if total_cost is not None else loss_c
                    count = count + 1
            avg_cost = total_cost/count
            scheduler.step(avg_cost)##FIXME
            ############ do validation here ###################
            if doValid:
                with torch.no_grad():
                    for dname in datasets_valid:
                        ########################################
                        df_, df_label_ = load_data(dname, -1)
                        for idf in range(0,df_.shape[0], batch_size):
                            ########################################
                            inputs = df_[idf:idf+batch_size,]
                            # forward pass
                            outputs = net((torch.Tensor(inputs)).cuda().requires_grad_(False))
                            outputs = np.squeeze(outputs)
                            labels = torch.Tensor( df_label_[idf:idf+batch_size,6] ).cuda()
                            loss = criteria(outputs, labels)
                            loss_c = loss.item()
                            val_total_cost = val_total_cost + loss_c if val_total_cost is not None else loss_c
                            val_count = val_count + 1
            val_avg_cost = val_total_cost/val_count
            ############ print and save ###################
            if epoch % 1 == 0 and saveCkpt:
                str_log = 'Epoch %d | training loss = %f, val loss=%f'%(epoch, avg_cost, val_avg_cost)
                print(str_log)
                fout = open(log_out,'a')
                fout.write(str_log+'\n')
                fout.close()
                torch.save({'model_state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
                print("Model saved in path: %s, write log to %s" % (ckpt_path, log_out))
                print_mem(devices)
                logger.info('')
            ############ early stop ###################
            '''
            if len(cost_list) < early_stop_interval: cost_list.append(avg_cost)
            else:
                for ic in range(len(cost_list)-1):
                    cost_list[ic] = cost_list[ic+1]
                cost_list[-1] = avg_cost
            if epoch > len(cost_list) and avg_cost >= cost_list[0]: break
            '''
            ############ early stop, using validation loss ###################
            if len(cost_list) < early_stop_interval: cost_list.append(val_avg_cost)
            else:
                for ic in range(len(cost_list)-1):
                    cost_list[ic] = cost_list[ic+1]
                cost_list[-1] = val_avg_cost
            if epoch > len(cost_list) and val_avg_cost >= cost_list[0]: break
            gc.collect() 
    #####################################
    with torch.no_grad():
        if doTest:
            net.eval() 
            print('Start testing model')
            logger.info('')
            for dname in datasets_test:
                ########################################
                outName0 = dname.split('/')[-2]
                outName = outName0+'_'+dname.split('/')[-1]
                df_, df_label_ = load_data(dname, -1)
                pred = copy.deepcopy(df_label_)
                mod_batch = df_label_.shape[0]%test_batch
                N_batch   = int( (df_label_.shape[0]-mod_batch)/test_batch )
                ########################################
                for i in range(N_batch+1):
                    inputs = df_[i*test_batch:(i+1)*test_batch] 
                    if inputs.shape[0] <= 0:continue
                    # forward pass
                    outputs = net((torch.Tensor(inputs)).cuda().requires_grad_(False))
                    outputs = np.squeeze(outputs)
                    pred[i*test_batch:(i+1)*test_batch,6] = outputs.cpu()
                pred[:,6]     =pred[:,6]     *10 ##scale back
                df_label_[:,6]=df_label_[:,6]*10
                hf = h5py.File("%s/%s_%s"%(outFilePath,tag, outName), 'w')
                hf.create_dataset('Pred' , data=pred)
                hf.create_dataset('Label', data=df_label_)
                hf.close()
                print('Saved pred data %s/%s_%s, from real %s'%(outFilePath, tag ,outName, dname))
        if saveToCPU and Restore:
            device = torch.device("cpu")
            net0.to(device)
            model = torch.jit.script(net0)
            model.save(pt_file_path)
            print('saved %s'%pt_file_path)
        print('done')
