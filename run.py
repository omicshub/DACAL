#%%


from os import path
from os.path import join as pj 
import time
import argparse 
import tracemalloc
from tqdm import tqdm 
import math
import numpy as np
import torch as th
import pandas as pd
import os
from torch import nn, autograd
import matplotlib.pyplot as plt
import umap
import random
import re
import itertools
from modules import models, utils
from modules.datasets import MultimodalDataset
from modules.datasets import MultiDatasetSampler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
parser = argparse.ArgumentParser()
## Task
parser.add_argument('--task', type=str, default='baron_single',
    help="Choose a task")
parser.add_argument('--reference', type=str, default='',
    help="Choose a reference task")
parser.add_argument('--exp', type=str, default='e0',
    help="Choose an experiment")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
parser.add_argument('--rf_experiment', type=str, default='',
    help="Choose a reference experiment") ###################################
# parser.add_argument('--data', type=str, default='sup',
#     help="Choose a data configuration")
parser.add_argument('--actions', type=str,nargs='+', default='train',
    help="Choose an action to run")
parser.add_argument('--method', type=str, default='scDAC',
    help="Choose an method to benchmark")
parser.add_argument('--init_model', type=str, default='',
    help="Load a saved model")
parser.add_argument('--mods-conditioned', type=str, nargs='+', default=[],
    help="Modalities conditioned for sampling")
parser.add_argument('--data-conditioned', type=str, default='prior.csv',
    help="Data conditioned for sampling")
parser.add_argument('--sample-num', type=int, default=0,
    help='Number of samples to be generated')
parser.add_argument('--input-mods', type=str, nargs='+', default=[],
    help="Input modalities for transformation")
## Training
parser.add_argument('--epoch-num', type=int, default=1000,
    help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=-1,
    help='Number of samples in a mini-batch') ########################
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
#parser.add_argument('--dim_logitx', type=int, default=64,
#    help='dim_logitx')
parser.add_argument('--grad-clip', type=float, default=-1,
    help='Gradient clipping value')
parser.add_argument('--s-drop-rate', type=float, default=0.1,
    help="Probility of dropping out subject ID during training")
parser.add_argument('--seed', type=int, default=-1,
    help="Set the random seed to reproduce the results")#############???
parser.add_argument('--drop_s', type=int, default=0,
    help="Force to drop s")###############################
parser.add_argument('--use-shm', type=int, default=1,
    help="Use shared memory to accelerate training")
parser.add_argument('--map_ref', type=int, default=0,
    help="Map query onto reference for transfer learning")
## Debugging
parser.add_argument('--print-iters', type=int, default=-1,
    help="Iterations to print training messages")
parser.add_argument('--log-epochs', type=int, default=100,
    help='Epochs to log the training states')
parser.add_argument('--save-epochs', type=int, default=1,
    help='Epochs to save the latest training states (overwrite previous ones)')
parser.add_argument('--time', type=int, default=0, choices=[0, 1],
    help='Time the forward and backward passes')
parser.add_argument('--debug', type=int, default=1, choices=[0, 1],
    help='Print intermediate variables')
# o, _ = parser.parse_known_args()  # for python interactive
o = parser.parse_args()
if o.task == 'mammary_epithelial'
    path_label = './data/mammary_epithelial/celltype.csv'
else o.task == 'human_pancreasn8'
    path_label = './data/human_pancreasn8/celltype.csv'
# tracemalloc.start()
# start_time = time.time()
# print("start_time:", start_time)
# Initialize global varibles
data_config = None
net = None
discriminator = None 
optimizer_net = None
optimizer_disc = None
benchmark = {
    "train_loss": [],
    "test_loss": [],
    "foscttm": [],
    "epoch_id_start": 0
}


def main():
    initialize()
    if o.actions == "print_model":
        print_model()
    if "train" in o.actions:
        train()
    if "test" in o.actions:
        test()
    if "infer_latent" in o.actions:
        infer_latent()###


def initialize():
    init_seed() ##
    init_dirs() ##
    load_data_config() ##
    load_model_config() ##
    get_gpu_config() ##
    init_model() ##


def init_seed():
    if o.seed >= 0:
        np.random.seed(o.seed) 
        th.manual_seed(o.seed) 
        th.cuda.manual_seed_all(o.seed)


def init_dirs():
    if o.use_shm == 1:
        o.data_dir = pj("./data",  o.task)
    else:
        o.data_dir = pj("./data",  o.task)
    o.result_dir = pj("result", o.task, o.exp, o.model)
    o.pred_dir = pj(o.result_dir, "predict", o.init_model)
    o.train_dir = pj(o.result_dir, "train")
    o.debug_dir = pj(o.result_dir, "debug")
    utils.mkdirs([o.train_dir, o.debug_dir])
    print("Task: %s\nExperiment: %s\nModel: %s\n" % (o.task, o.exp, o.model))


def load_data_config():
    # get_dims_x()
    # o.mods = list(o.dims_x.keys())
    # o.mod_num = len(o.dims_x)
    # global data_config
    # data_config = utils.load_toml("configs/data.toml")[o.task]
    # for k, v in data_config.items():
    #     vars(o)[k] = v

    # o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)
    

    
    # if o.reference != '':
    #     data_config_ref = utils.load_toml("configs/data.toml")[o.reference]
    #     _, _, _, o.dims_s = utils.gen_all_batch_ids(data_config_ref["s_joint"], 
    #                                                 data_config_ref["combs"])
    if o.reference == '':
        o.dims_x, o.dims_chr, o.mods = get_dims_x(ref=0)
        o.ref_mods = o.mods
    else:
        _, _, o.mods = get_dims_x(ref=0)
        o.dims_x, o.dims_chr, o.ref_mods = get_dims_x(ref=1)
    o.mod_num = len(o.mods)

    if o.rf_experiment == '':
        o.rf_experiment = o.exp
    
    global data_config
    data_config = utils.gen_data_config(o.task, o.reference)
    # data_config = utils.load_toml("configs/data.toml")[o.task]
    for k, v in data_config.items():
        vars(o)[k] = v
    if o.batch_size > 0:
        o.N = o.batch_size

    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)

    if "continual" in o.task:
        o.continual = True
        o.dim_s_query = len(utils.load_toml("configs/data.toml")[re.sub("_continual", "", o.task)]["s_joint"])
        o.dim_s_ref = len(utils.load_toml("configs/data.toml")[o.reference]["s_joint"])
    else:
        o.continual = False

    if o.reference != '' and o.continual == False and o.map_ref == 1:  # map query onto reference for transfer learning

        o.dims_s = {k: v + 1 for k, v in o.dims_s.items()}
        
        cfg_task_ref = re.sub("_atlas|_generalize|_transfer|_ref_.*", "", o.reference)
        data_config_ref = utils.load_toml("configs/data.toml")[cfg_task_ref]
        _, _, s_ref, dims_s_ref = utils.gen_all_batch_ids(data_config_ref["s_joint"], 
                                                    data_config_ref["combs"])
        o.subset_ids_ref = {m: [] for m in dims_s_ref}
        for subset_id, id_dict in enumerate(s_ref):
            for m in id_dict.keys():
                o.subset_ids_ref[m].append(subset_id)

    o.dim_s = o.dims_s["joint"]
    o.dim_b = 2



def load_model_config():
    model_config = utils.load_toml("configs/model.toml")["default"]
    if o.model != "default":
        model_config.update(utils.load_toml("configs/model.toml")[o.model])
    for k, v in model_config.items():
        vars(o)[k] = v
    o.dim_z = o.dim_c + o.dim_b
    o.dims_dec_x = o.dims_enc_x[::-1]
    o.dims_dec_s = o.dims_enc_s[::-1]
    if "dims_enc_chr" in vars(o).keys():
        o.dims_dec_chr = o.dims_enc_chr[::-1]
    o.dims_h = {}
    for m, dim in o.dims_x.items():
        o.dims_h[m] = dim if m != "atac" else o.dims_enc_chr[-1] * 22
    print("dims_h:", o.dims_h)

def get_gpu_config():
    o.G = 1  # th.cuda.device_count()  # get GPU number
    o.N = 512
    assert o.N % o.G == 0, "Please ensure the mini-batch size can be divided " \
        "by the GPU number"
    o.n = o.N // o.G
    print("Total mini-batch size: %d, GPU number: %d, GPU mini-batch size: %d" % (o.N, o.G, o.n))


def init_model():
    """
    Initialize the model, optimizer, and benchmark
    """
    # global net, discriminator, optimizer_net, optimizer_disc
    # net = models.Net_DP(o).cuda()
    # optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    # if o.init_model != '':
    #     fpath = pj(o.train_dir, o.init_model)
    #     savepoint = th.load(fpath+".pt")
    #     net.load_state_dict(savepoint['net_states'])
    #     optimizer_net.load_state_dict(savepoint['optim_net_states'])
    #     benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
    #     print('Model is initialized from ' + fpath + ".pt")
    # net_param_num = sum([param.data.numel() for param in net.parameters()])
    # print('Parameter number: %.3f M' % (net_param_num / 1e6))
    global net, discriminator, optimizer_net, optimizer_disc
    # global net, optimizer_net
    
    # Initialize models    
    net = models.Net_DP(o).cuda()

    discriminator = models.Discriminator(o).cuda()
    net_param_num = sum([param.data.numel() for param in net.parameters()])
    disc_param_num = sum([param.data.numel() for param in discriminator.parameters()])
    # print('Parameter number: %.3f M' % (net_param_num / 1e6))
    print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))

    # Load benchmark
    # if o.init_model != '':
    #     if o.init_from_ref == 0:
    #         fpath = pj(o.train_dir, o.init_model)
    #         savepoint_toml = utils.load_toml(fpath+".toml")
    #         benchmark.update(savepoint_toml['benchmark'])
    #         o.ref_epoch_num = savepoint_toml["o"]["ref_epoch_num"]
    #     else:
    #         fpath = pj("result", o.reference, o.rf_experiment, o.model, "train", o.init_model)
    #         benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
    #         o.ref_epoch_num = benchmark["epoch_id_start"]
    # else:
    #     o.ref_epoch_num = 0

    # Initialize optimizers
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    optimizer_disc = th.optim.AdamW(discriminator.parameters(), lr=o.lr)
    if o.init_model != '':
        # if o.init_from_ref == 0:
        fpath = pj(o.train_dir, o.init_model)
        savepoint_toml = utils.load_toml(fpath+".toml")
        savepoint = th.load(fpath+".pt")
        net.load_state_dict(savepoint['net_states'])
        optimizer_net.load_state_dict(savepoint['optim_net_states'])
        benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
        print('Model is initialized from ' + fpath + ".pt")
    else:
        o.ref_epoch_num = 0

def print_model():
    #global net, discriminator
    global net
    with open(pj(o.result_dir, "model_architecture.txt"), 'w') as f:
        print(net, file=f)
        print(discriminator, file=f)

def get_dims_x(ref):
    if ref == 0:
        feat_dims = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    else:
        feat_dims = utils.load_csv(pj("data", "processed", o.reference, "feat", "feat_dims.csv"))
    feat_dims = utils.transpose_list(feat_dims)
    
    dims_x = {}
    dims_chr = []
    for i in range(1, len(feat_dims)):
        m = feat_dims[i][0]
        if m == "atac":
            dims_chr = list(map(int, feat_dims[i][1:]))
            dims_x[m] = sum(dims_chr)
        else:   
            dims_x[m] = int(feat_dims[i][1])
    print("Input feature numbers: ", dims_x)

    mods = list(dims_x.keys())
    
    return dims_x, dims_chr, mods
    # dims_x = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    # dims_x = utils.transpose_list(dims_x)
    # o.dims_x = {}
    # for i in range(1, len(dims_x)):
    #     m = dims_x[i][0]
    #     if m == "atac":
    #         o.dims_chr = list(map(int, dims_x[i][1:]))
    #         o.dims_x[m] = sum(o.dims_chr)
    #     else:
    #         o.dims_x[m] = int(dims_x[i][1])


    # print("Input feature numbers: ", o.dims_x)


def train():
    train_data_loader_cat = get_dataloader_cat("train")
    epoch_id_list = []
    ari_list = []
    nmi_list = []
    sc_list = []

    for epoch_id in range(benchmark['epoch_id_start'], o.epoch_num):
        run_epoch(train_data_loader_cat, "train", epoch_id)
        if epoch_id >299:
            c = infer_latent_dp(save_input=False)
            net.loss_calculator_dp.mean_dp, net.loss_calculator_dp.weight_concentration_dp,net.loss_calculator_dp.mean_precision_dp,net.loss_calculator_dp.precisions_cholesky_dp, net.loss_calculator_dp.degrees_of_freedom_dp, net.scdp.predict_label = dp(c)
        else:
            pass
        # ari, nmi, sc = cluster_index_calculer(z, net.scdp.predict_label)
        # print("ari:", ari)
        # print("nmi:", nmi)
        # print("sc:", sc)
        # epoch_id_list.append(epoch_id)
        # ari_list.append(ari)
        # nmi_list.append(nmi)
        # sc_list.append(sc)
        # plt_ari(epoch_id_list, ari_list)
        # plt_nmi(epoch_id_list, nmi_list)
        # plt_sc(epoch_id_list, sc_list)
        check_to_save(epoch_id)

def dp(z):
    # z_np = (z).cpu().detach().numpy()   
    z_np = z.cpu().detach().numpy()   
    bgm = BayesianGaussianMixture(
        n_components=30, weight_concentration_prior=1e-200,mean_precision_prior = 200,covariance_type='diag',init_params ='kmeans', max_iter=100, warm_start = True
        ).fit(z_np)
    predict_label_array = bgm.predict(z_np)
    predict_label_array = bgm.predict(z_np)
    predict_label = th.Tensor(np.array(predict_label_array)).unsqueeze(1).cuda()
    mean_dp = th.Tensor(np.array(bgm.means_))
    weight_concentration_dp = th.Tensor(np.array(bgm.weight_concentration_))
    precisions_cholesky_dp = th.Tensor(np.array(bgm.precisions_cholesky_))
    degrees_of_freedom_dp = th.Tensor(np.array(bgm.degrees_of_freedom_))
    mean_precision_dp = th.Tensor(np.array(bgm.mean_precision_))   
    return mean_dp, weight_concentration_dp, mean_precision_dp, precisions_cholesky_dp, degrees_of_freedom_dp, predict_label

def dp_infer(z):
    # z_np = (z).cpu().detach().numpy()   
    z_np = z.cpu().detach().numpy()   
    bgm = BayesianGaussianMixture(
        n_components=30, weight_concentration_prior=1e-200,mean_precision_prior = 200, covariance_type='full',  init_params ='kmeans', random_state=42, max_iter=1000
        ).fit(z_np)
    predict_label_array = bgm.predict(z_np)
    predict_label = th.Tensor(np.array(predict_label_array)).unsqueeze(1).cuda()
    mean_dp = th.Tensor(np.array(bgm.means_))
    weight_concentration_dp = th.Tensor(np.array(bgm.weight_concentration_))
    precisions_cholesky_dp = th.Tensor(np.array(bgm.precisions_cholesky_))
    degrees_of_freedom_dp = th.Tensor(np.array(bgm.degrees_of_freedom_))
    mean_precision_dp = th.Tensor(np.array(bgm.mean_precision_))   
    return mean_dp, weight_concentration_dp, mean_precision_dp, precisions_cholesky_dp, degrees_of_freedom_dp, predict_label


def get_dataloaders(split, train_ratio=None):
    data_loaders = {}
    for subset in range(len(o.s)):
        data_loaders[subset] = get_dataloader(subset, split, train_ratio=train_ratio)
    return data_loaders


def get_dataloader(subset, split, train_ratio=None):
    dataset = MultimodalDataset(o.task, o.data_dir, subset, split, train_ratio=train_ratio)
    shuffle = True if split == "train" else False
    # shuffle = False
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N, shuffle=shuffle,
                                           num_workers=64, pin_memory=True)
    print("Subset: %d, modalities %s: %s size: %d" %
          (subset, str(o.combs[subset]), split, dataset.size))
    return data_loader


def get_dataloader_cat(split, train_ratio=None):
    datasets = []
    for subset in range(len(o.s)):
        datasets.append(MultimodalDataset(o.task, o.data_dir, subset, split, train_ratio=train_ratio))
        print("Subset: %d, modalities %s: %s size: %d" %  (subset, str(o.combs[subset]), split,
            datasets[subset].size))
    dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
    shuffle = True if split == "train" else False
    # shuffle = False
    sampler = MultiDatasetSampler(dataset_cat, batch_size=o.N, shuffle=shuffle)
    data_loader = th.utils.data.DataLoader(dataset_cat, batch_size=o.N, sampler=sampler, 
        num_workers=64, pin_memory=True)
    return data_loader



def get_eval_dataloader(train_ratio=False):
    data_config_new = utils.copy_dict(data_config)
    data_config_new.update({"combs": [o.mods], "comb_ratios": [1]})
    if train_ratio:
        data_config_new.update({"train_ratio": train_ratio})
    dataset = MultimodalDataset(data_config_new, o.mods, "test")
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N,
        shuffle=False, num_workers=64, pin_memory=True)
    print("Eval Dataset %s: test: %d\n" % (str(o.mods), dataset.size))
    return data_loader


def test():
    data_loaders = get_dataloaders()
    run_epoch(data_loaders, "test")


def run_epoch(data_loader, split, epoch_id=0):
    if split == "train":
        net.train()
        discriminator.train()
    elif split == "test":
        net.eval()
        discriminator.eval()
    else:
        assert False, "Invalid split: %s" % split
    net.o.epoch_id = epoch_id
    # losses = []
    # loss_total = 0
    # for i, data in enumerate(data_loader):
    #     loss = run_iter(split, data)
    #     losses.append(loss)
    #     loss_total += loss
    #     if o.print_iters > 0 and (i+1) % o.print_iters == 0:
    #         print('Epoch: %d/%d, Batch: %d/%d, %s loss: %.3f' % (epoch_id+1,
    #         o.epoch_num, i+1, len(data_loader), split, loss))
    # loss_avg = loss_total / len(data_loader)
    # print('Epoch: %d/%d, %s loss: %.3f\n' % (epoch_id+1, o.epoch_num, split, loss_avg))
    # benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    # return loss_avg
    losses = []
    for i, data in enumerate(data_loader):
        loss = run_iter(split, epoch_id, i, data)
        losses.append(loss)
        if o.print_iters > 0 and (i+1) % o.print_iters == 0:
            print('%s\tepoch: %d/%d\tBatch: %d/%d\t%s_loss: %.2f'.expandtabs(3) % 
                  (o.task, epoch_id+1, o.epoch_num, i+1, len(data_loader), split, loss))
    loss_avg = np.nanmean(losses)
    # epoch_time = (time.time() - start_time) / 3600 / 24
    # elapsed_time = epoch_time * (epoch_id+1)
    # total_time = epoch_time * o.epoch_num
    print('%s\t%s\tepoch: %d/%d\t%s_loss: %.2f\n'.expandtabs(3) % 
          (o.task, o.exp, epoch_id+1, o.epoch_num, split, loss_avg))
    benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    return loss_avg


def run_iter(split, epoch_id, iter_id, inputs):
    # inputs = utils.convert_tensors_to_cuda(inputs)
    # if split == "train":
    #     with autograd.set_detect_anomaly(o.debug == 1):
    #         loss_net = forward_net(inputs)
    #         loss = loss_net
    #         update_net(loss) 
            
            
            
    # else:
    #     with th.no_grad():
    #         loss_net = forward_net(inputs)
    #         loss = loss_net
    if split == "train":
        skip = False
        if o.continual and o.sample_ref == 1:
            subset_id_ref = inputs["s"]["joint"][0, 0].item() - o.dim_s_query
            cycle_id = iter_id // o.dim_s
            if subset_id_ref >= 0 and subset_id_ref != (cycle_id % o.dim_s_ref):
                skip = True
        
        if skip:
            return np.nan
        else:
            inputs = utils.convert_tensors_to_cuda(inputs)
            with autograd.set_detect_anomaly(o.debug == 1):
                loss_net, c_all = forward_net(inputs)
                discriminator.epoch = epoch_id - o.ref_epoch_num
                K = 10
                if o.exp == "k_1":
                    K = 1
                elif o.exp == "k_2":
                    K = 2
                elif o.exp == "k_4":
                    K = 4
                elif o.exp == "k_5":
                    K = 5

                for _ in range(K):
                    loss_disc = forward_disc(utils.detach_tensors(c_all), inputs["s"])
                    update_disc(loss_disc)
                # c = models.CheckBP('c')(c)
                loss_adv = forward_disc(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
                update_net(loss)
                
                print("loss_net: %.3f\tloss_adv: %.3f\tprob: %.4f".expandtabs(3) %
                      (loss_net.item(), loss_adv.item(), discriminator.prob))

            # If we want to map query onto reference for transfer learning, then we take the reference as an additional batch
            # and train the discriminator after the last training subset
            if o.reference != '' and o.continual == False and o.map_ref == 1 and inputs["s"]["joint"][0, 0].item() == o.dim_s - 2:
                
                # Randomly load c inferred from the reference dataset
                c_all_ref = {}
                subset_ids_sampled = {m: random.choice(ids) for m, ids in o.subset_ids_ref.items()}
                for m, subset_id in subset_ids_sampled.items():
                    z_dir = pj("result", o.reference, o.rf_experiment, o.model, "predict", o.init_model,
                            "subset_"+str(subset_id), "z", m)
                    filename = random.choice(utils.get_filenames(z_dir, "csv"))
                    z = th.from_numpy(np.array(utils.load_csv(pj(z_dir, filename)), dtype=np.float32))
                    # z = th.tensor(utils.load_csv(pj(z_dir, filename)), dtype=th.float32)
                    c_all_ref[m] = z[:, :o.dim_c]
                c_all_ref = utils.convert_tensors_to_cuda(c_all_ref)

                # Generate s for the reference dataset, which is treated as the last subset
                s_ref = {}
                tmp = inputs["s"]["joint"]
                for m, d in o.dims_s.items():
                    s_ref[m] = th.full((c_all_ref[m].size(0), 1), d-1, dtype=tmp.dtype, device=tmp.device)

                with autograd.set_detect_anomaly(o.debug == 1):
                    for _ in range(K):
                        loss_disc = forward_disc(c_all_ref, s_ref)
                        update_disc(loss_disc)
            
    else:
        with th.no_grad():
            inputs = utils.convert_tensors_to_cuda(inputs)
            loss_net, c_all = forward_net(inputs)
            loss_adv = forward_disc(c_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv
            
            print("loss_net: %.3f\tloss_adv: %.3f\tprob: %.4f".expandtabs(3) %
                  (loss_net.item(), loss_adv.item(), discriminator.prob))    
    return loss.item()


def forward_net(inputs):
    return net(inputs)

def forward_disc(c, s):
    return discriminator(c, s)

def update_net(loss):
    update(loss, net, optimizer_net)

def update_disc(loss):
   update(loss, discriminator, optimizer_disc)    

def update(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()

    if o.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    optimizer.step()


def check_to_save(epoch_id):
    if (epoch_id+1) % o.log_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_%08d" % epoch_id)
    if (epoch_id+1) % o.save_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_latest")


def save_training_states(epoch_id, filename):
    benchmark['epoch_id_start'] = epoch_id
    utils.save_toml({"o": vars(o), "benchmark": benchmark}, pj(o.train_dir, filename+".toml"))
    th.save({"net_states": net.state_dict(),
             "optim_net_states": optimizer_net.state_dict(),
            }, pj(o.train_dir, filename+".pt"))




def infer_latent_dp(save_input=False):
    print("Inferring ...")
    dirs = {}
    base_dir = pj(o.result_dir, "represent", o.init_model)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}}
            dirs[subset_id]["z"]["rna"] = pj(base_dir, "subset_"+str(subset_id), "z", "rna")
            utils.mkdirs(dirs[subset_id]["z"]["rna"], remove_old=False)          
            z_list = []
            if save_input:
                for m in o.combs[subset_id]:
                    dirs[subset_id]["x"][m] = pj(base_dir, "subset_"+str(subset_id), "x", m)
                    utils.mkdirs(dirs[subset_id]["x"][m], remove_old=True)
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"    
            for i, data in enumerate(data_loader):
                data = utils.convert_tensors_to_cuda(data)
                _,_,_,_,c,_= net.scdp(data)      
                z_list.append(c)
                z_all = th.cat(z_list, dim = 0)
    return(z_all)

def infer_latent(only_joint=False, impute=False, save_input=False):
    print("Inferring ...")
    dirs = {}
    base_dir = pj(o.result_dir, "represent", o.init_model)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        z_all_large = []
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            
            #dirs[subset_id] = {"z": {}, "x_r": {}, "x": {}, "w_lis1": {}}
            # dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}, "y_cat_list": {}, "c_ymu": {}, "n_covariance2": {}, "n_mu": {}, "d2": {}, "w_covariance3": {}}
            # dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}, "predict_label":{}}
            dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}}
            dirs[subset_id]["z"]["joint"] = pj(base_dir, "subset_"+str(subset_id), "z", "joint")
            dirs[subset_id]["y"] = pj(base_dir, "subset_"+str(subset_id), "y")
            dirs[subset_id]["zz"] = pj(base_dir, "subset_"+str(subset_id), "zz")
            utils.mkdirs(dirs[subset_id]["z"]["joint"], remove_old=True)
            if not only_joint:
                for m in o.combs[subset_id]:
                    dirs[subset_id]["z"][m] = pj(base_dir, "subset_"+str(subset_id), "z", m)
                    utils.mkdirs(dirs[subset_id]["z"][m], remove_old=True)
                    # dirs[subset_id]["predict_label"][m] = pj(base_dir, "subset_"+str(subset_id), "predict_label", m)
                    # utils.mkdirs(dirs[subset_id]["predict_label"][m], remove_old=True)
                    dirs[subset_id]["x_r_pre"][m] = pj(base_dir, "subset_"+str(subset_id), "x_r_pre", m)
                    utils.mkdirs(dirs[subset_id]["x_r_pre"][m], remove_old=True)
                    # dirs[subset_id]["c_ymu"][m] = pj(base_dir, "subset_"+str(subset_id), "c_ymu", m)
                    # utils.mkdirs(dirs[subset_id]["c_ymu"][m], remove_old=True)     
                    # dirs[subset_id]["n_covariance2"][m] = pj(base_dir, "subset_"+str(subset_id), "n_covariance2", m)
                    # utils.mkdirs(dirs[subset_id]["n_covariance2"][m], remove_old=True)  
                    # dirs[subset_id]["n_mu"][m] = pj(base_dir, "subset_"+str(subset_id), "n_mu", m)
                    # utils.mkdirs(dirs[subset_id]["n_mu"][m], remove_old=True)         
                    # dirs[subset_id]["d2"][m] = pj(base_dir, "subset_"+str(subset_id), "d2", m)
                    # utils.mkdirs(dirs[subset_id]["d2"][m], remove_old=True)  
                    # dirs[subset_id]["w_covariance3"][m] = pj(base_dir, "subset_"+str(subset_id), "w_covariance3", m)
                    # utils.mkdirs(dirs[subset_id]["w_covariance3"][m], remove_old=True)            
            if impute:
                for m in o.mods:
                    dirs[subset_id]["x_r"][m] = pj(base_dir, "subset_"+str(subset_id), "x_r", m)
                    utils.mkdirs(dirs[subset_id]["x_r"][m], remove_old=True)
            if save_input:
                for m in o.combs[subset_id]:
                    dirs[subset_id]["x"][m] = pj(base_dir, "subset_"+str(subset_id), "x", m)
                    utils.mkdirs(dirs[subset_id]["x"][m], remove_old=True)
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
            for i, data in enumerate(tqdm(data_loader)):
                data = utils.convert_tensors_to_cuda(data)
                # conditioned on all observed modalities
                #x_r_pre, _, _, _, z, _, _, *_ = net.dpmm(data)  # N * K
                x_r_pre, z, *_= net.scdp(data) 
                utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                # utils.save_tensor_to_csv(x_r_pre[m], pj(dirs[subset_id]["x_r_pre"][m], fname_fmt) % i)
                # utils.save_tensor_to_csv(predict_label, pj(dirs[subset_id]["predict_label"], fname_fmt) % i)
                if impute:
                    x_r = models.gen_real_data(x_r_pre, sampling=True)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_r"][m], fname_fmt) % i)
                if save_input:
                    for m in o.combs[subset_id]:
                        utils.save_tensor_to_csv(data["x"][m], pj(dirs[subset_id]["x"][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if not only_joint:
                    for m in data["x"].keys():
                        input_data = {
                            "x": {m: data["x"][m]},
                            "s": data["s"],  #这儿可能得删掉，
                            "e": {}
                        }
                        if m in data["e"].keys():
                            input_data["e"][m] = data["e"][m]
                        _, _, _, _,c, _= net.scdp(input_data)  # N * K
                     

                    
                    if i >0:
                        z_all = th.cat((z_all, c), dim = 0)
                    else:
                        z_all = c

                # print(z_all.shape)
            

            z_all_large.append(z_all)
            # print(len(z_all_large))
        # print("111111111",z_all_large.shape)
        z_all_large = th.cat(z_all_large)
        _, _, _, _, _, predict_label = dp_infer(z_all_large)
        # print(z_all_large.shape)
        utils.save_tensor_to_csv(z_all_large, './result/mammary_epithelialn/e0/default/represent/z.csv')
        utils.save_tensor_to_csv(predict_label, './result/mammary_epithelialn/e0/default/represent/y.csv')
        label_true = utils.load_csv(path_label)
        label_tlist = utils.transpose_list(label_true)[1][1:]
        predict_label_cpu = predict_label.cpu()
        label_plist = utils.transpose_list(predict_label_cpu)[0]
        ari = adjusted_rand_score(label_tlist, label_plist) #l1 kpca20
        nmi = normalized_mutual_info_score(label_tlist, label_plist)
        z_all_cpu = z_all_large.cpu()
        sc = silhouette_score(z_all_cpu, label_plist)
        print("ari:", ari)
        print("nmi:", nmi)
        print("sc:", sc)
# def infer_latent(only_joint=True, impute=False, save_input=False):
#     print("Inferring ...")
#     dirs = {}
#     base_dir = pj(o.result_dir, "represent", o.init_model)
#     data_loaders = get_dataloaders("test", train_ratio=0)
#     net.eval()
#     with th.no_grad():
#         for subset_id, data_loader in data_loaders.items():
#             print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            
#             #dirs[subset_id] = {"z": {}, "x_r": {}, "x": {}, "w_lis1": {}}
#             # dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}, "y_cat_list": {}, "c_ymu": {}, "n_covariance2": {}, "n_mu": {}, "d2": {}, "w_covariance3": {}}
#             # dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}, "predict_label":{}}
#             dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}}
#             dirs[subset_id]["z"]["joint"] = pj(base_dir, "subset_"+str(subset_id), "z", "joint")
#             utils.mkdirs(dirs[subset_id]["z"]["joint"], remove_old=True)
#             if not only_joint:
#                 for m in o.combs[subset_id]:
#                     dirs[subset_id]["z"][m] = pj(base_dir, "subset_"+str(subset_id), "z", m)
#                     utils.mkdirs(dirs[subset_id]["z"][m], remove_old=True)
#                     # dirs[subset_id]["predict_label"][m] = pj(base_dir, "subset_"+str(subset_id), "predict_label", m)
#                     # utils.mkdirs(dirs[subset_id]["predict_label"][m], remove_old=True)
#                     dirs[subset_id]["x_r_pre"][m] = pj(base_dir, "subset_"+str(subset_id), "x_r_pre", m)
#                     utils.mkdirs(dirs[subset_id]["x_r_pre"][m], remove_old=True)
#                     # dirs[subset_id]["c_ymu"][m] = pj(base_dir, "subset_"+str(subset_id), "c_ymu", m)
#                     # utils.mkdirs(dirs[subset_id]["c_ymu"][m], remove_old=True)     
#                     # dirs[subset_id]["n_covariance2"][m] = pj(base_dir, "subset_"+str(subset_id), "n_covariance2", m)
#                     # utils.mkdirs(dirs[subset_id]["n_covariance2"][m], remove_old=True)  
#                     # dirs[subset_id]["n_mu"][m] = pj(base_dir, "subset_"+str(subset_id), "n_mu", m)
#                     # utils.mkdirs(dirs[subset_id]["n_mu"][m], remove_old=True)         
#                     # dirs[subset_id]["d2"][m] = pj(base_dir, "subset_"+str(subset_id), "d2", m)
#                     # utils.mkdirs(dirs[subset_id]["d2"][m], remove_old=True)  
#                     # dirs[subset_id]["w_covariance3"][m] = pj(base_dir, "subset_"+str(subset_id), "w_covariance3", m)
#                     # utils.mkdirs(dirs[subset_id]["w_covariance3"][m], remove_old=True)            
#             if impute:
#                 for m in o.mods:
#                     dirs[subset_id]["x_r"][m] = pj(base_dir, "subset_"+str(subset_id), "x_r", m)
#                     utils.mkdirs(dirs[subset_id]["x_r"][m], remove_old=True)
#             if save_input:
#                 for m in o.combs[subset_id]:
#                     dirs[subset_id]["x"][m] = pj(base_dir, "subset_"+str(subset_id), "x", m)
#                     utils.mkdirs(dirs[subset_id]["x"][m], remove_old=True)
#             fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
#             for i, data in enumerate(tqdm(data_loader)):
#                 data = utils.convert_tensors_to_cuda(data)
#                 # conditioned on all observed modalities
#                 #x_r_pre, _, _, _, z, _, _, *_ = net.dpmm(data)  # N * K
#                 x_r_pre, z= net.scdp(data) 
#                 utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
#                 # utils.save_tensor_to_csv(x_r_pre[m], pj(dirs[subset_id]["x_r_pre"][m], fname_fmt) % i)
#                 # utils.save_tensor_to_csv(predict_label, pj(dirs[subset_id]["predict_label"], fname_fmt) % i)
#                 if impute:
#                     x_r = models.gen_real_data(x_r_pre, sampling=True)
#                     for m in o.mods:
#                         utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_r"][m], fname_fmt) % i)
#                 if save_input:
#                     for m in o.combs[subset_id]:
#                         utils.save_tensor_to_csv(data["x"][m], pj(dirs[subset_id]["x"][m], fname_fmt) % i)

#                 # conditioned on each individual modalities
#                 if not only_joint:
#                     for m in data["x"].keys():
#                         input_data = {
#                             "x": {m: data["x"][m]},
#                             "s": data["s"],  #这儿可能得删掉，
#                             "e": {}
#                         }
#                         if m in data["e"].keys():
#                             input_data["e"][m] = data["e"][m]
#                         #_, _, _, _, z, c, b, *_ = net.sct(input_data)  # N * K
#                         # _, c_ymu, _, _, _, z, y_cat_list, _, _, _, n_covariance2, n_mu, d2, w_covariance3, _ = net.dpmm(input_data)  # N * K
#                         _, z= net.scdp(input_data)  # N * K
                    
#                         # print(input_data['x']['rna'].shape, z.shape)
#                         utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"][m], fname_fmt) % i)    
#                         # utils.save_tensor_to_csv(predict_label, pj(dirs[subset_id]["predict_label"][m], fname_fmt) % i)  
#                 if i >0:
#                     z_all = th.cat((z_all, z), dim = 0)
#                 else:
#                     z_all = z
#             _, _, _, _, _, predict_label = dp_infer(z_all)
            
#                 # print("predict_label", predict_label1[0:8])
#                 # print("predict_label", predict_label1.size())
#                 # predict_label_list = utils.convert_tensor_to_list((th.argmax(predict_label1, dim = 1).unsqueeze(1)).type_as(predict_label1))
#                 # print("predict_labellist:", predict_label_list[1:10]) 
#             predict_label_list = utils.convert_tensor_to_list(predict_label)
#             if o.task == 'chen_10':
#                 utils.save_list_to_csv(predict_label_list, "./data/chen_10/predict_label.csv")
#                 # path_label = '/root/data/asj/2023/0118/sc-transformer-gmvaextoytoz/data/tab1/label_seurat/label2_10.csv'
#             elif o.task == 'tabula2_8':
#                 utils.save_list_to_csv(predict_label_list, "/root/data/asj/2023/0118/sc-transformer-gmvaextoytoz/data/tab1/label_seurat/predict_label2_80224less.csv")
#             elif o.task == 'tabula2_6':
#                 utils.save_list_to_csv(predict_label_list, "/root/data/asj/2023/0118/sc-transformer-gmvaextoytoz/data/tab1/label_seurat/predict_label2_60224less.csv")   
#             elif o.task == 'tabula2_4':
#                 utils.save_list_to_csv(predict_label_list, "/root/data/asj/2023/0118/sc-transformer-gmvaextoytoz/data/tab1/label_seurat/predict_label2_4save.csv") 
#             else:
#                 # utils.save_list_to_csv(predict_label_list, "/root/data/asj/2023/0721_2/scDAC/result/wnn_t/e0/default/represent/sp_latest/predict_label.csv")    
#                 utils.save_list_to_csv(predict_label_list, "/root/data/asj/2023/0721_2/scDAC/result/pancreas_t/e0/default/represent/sp_latest/predict_label.csv")         
#             # utils.save_list_to_csv(predict_label_list, "/root/data/asj/2023/0118/sc-transformer-gmvaextoytoz/data/tab1/label_seurat/predict_label02093.csv")
#                 # print("data_list:", data_list[1:8])
#                 # if i > 33:
#             z_all_cpu = z_all.cpu()
#             predict_label_cpu = predict_label.cpu()
#             # path_label = '/root/data/asj/2023/0118/sc-transformer-gmvaextoytoz/data/m1/label_seurat/label_Macosko.csv'
#             label_true = utils.load_csv(path_label)
#             label_tlist = utils.transpose_list(label_true)[1][1:]
#             label_plist = utils.transpose_list(predict_label_cpu)[0]
#             ari = adjusted_rand_score(label_tlist, label_plist) #l1 kpca20
#             nmi = normalized_mutual_info_score(label_tlist, label_plist)
#             sc = silhouette_score(z_all_cpu, label_plist)
#             print("ari:", ari)
#             print("nmi:", nmi)
#             print("sc:", (sc+1)/2)
main()
# CPU memory and time
# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()
# end_time = time.time()
# print("end_time:", end_time)
# print("Training time: {:.2f} seconds".format(end_time - start_time))
# %%
