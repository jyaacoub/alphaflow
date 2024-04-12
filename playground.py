#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='splits/transporters_only.csv')
parser.add_argument('--templates_dir', type=str, default='./data')
parser.add_argument('--msa_dir', type=str, default='./alignment_dir')
parser.add_argument('--mode', choices=['alphafold', 'esmfold'], default='alphafold')
parser.add_argument('--samples', type=int, default=10)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--outpdb', type=str, default='./outpdb/default')
parser.add_argument('--weights', type=str, default=None)
#parser.add_argument('--ckpt', type=str, default=None)
#parser.add_argument('--original_weights', action='store_true')
parser.add_argument('--pdb_id', nargs='*', default=[])
#parser.add_argument('--subsample', type=int, default=None) # for MSA subsampling
#parser.add_argument('--resample', action='store_true')     # simple resampling of MSA
parser.add_argument('--tmax', type=float, default=1.0)
parser.add_argument('--templates', action='store_true')
parser.add_argument('--no_diffusion', action='store_true', default=False)
parser.add_argument('--self_cond', action='store_true', default=False)
parser.add_argument('--noisy_first', action='store_true', default=False)
parser.add_argument('--runtime_json', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
parser.add_argument('--low_mem', action='store_true', default=False)

parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)

io_dir="/home/jyaacoub/projects/def-sushant/jyaacoub/data/pdbbind/alphaflow_io/"

args = parser.parse_args(args=f"--mode alphafold \
                  --input_csv {io_dir}/input_19.csv \
                  --msa_dir {io_dir}/msa_dir \
                  --weights weights/alphaflow_md_distilled_202402.pt \
                  --samples 50 \
                  --outpdb {io_dir}/out_pdb_MD-distilled \
                  --no_overwrite \
                  --low_mem \
                  \
                  --rank 0 \
                  --world_size 2".split())

import functools
import torch, tqdm, os, json, time
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, wrap, always_wrap_policy, transformer_auto_wrap_policy

from collections import defaultdict
from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import AlphaFoldWrapper, ESMFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
import alphaflow.utils.protein as protein
from alphaflow.data.inference import AlphaFoldCSVDataset, CSVDataset
from collections import defaultdict
from alphaflow.config import model_config

from alphaflow.utils.logging import get_logger
logger = get_logger(__name__)
torch.set_float32_matmul_precision("medium")

config = model_config(
    'initial_training',
    train=False, 
    low_prec=True,
    long_sequence_inference=args.low_mem # only modifies c.globals and c.models
) 
schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Use a free port

    # Initializes the default distributed process group, and this will also initialize the distributed package
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
                                                                                       
def cleanup():
    torch.distributed.destroy_process_group()
    
def custom_autowrap_policy(module, recurse, unwrapped_params):
    """
    FSDP Wrapping policy to target only the AlphaFold.extra_msa_stack module
    """
    if recurse:
        return True
    result = isinstance(module, torch.nn.Module) and 'AlphaFold' in module.__class__.__name__
    if result and args.rank == 0:
        print(module.__class__.__name__)
    return result
    # return isinstance(module, torch.nn.Module) and ('ExtraMSAStack' in module.__class__.__name__ 
    #                                                    or 'EvoformerStack' in module.__class__.__name__
    #                                                    or 'InputPairStack' in module.__class__.__name__)

# %%
if args.rank != 0:
    time.sleep(1)
print(args.rank, args.world_size)
setup(args.rank, args.world_size)

################## GET DATASET ##################
valset = {
    'alphafold': AlphaFoldCSVDataset,
    'esmfold': CSVDataset,
}[args.mode](
    data_cfg,
    args.input_csv,
    msa_dir=args.msa_dir,
    data_dir=args.templates_dir,
    templates=args.templates,
)

#%%
with torch.no_grad():
    torch.cuda.set_device(args.rank)
    ################## LOAD MODEL ##################
    logger.info("Loading the model")
    model_class = {'alphafold': AlphaFoldWrapper, 'esmfold': ESMFoldWrapper}[args.mode]

    ckpt = torch.load(args.weights, map_location='cpu')
    model = model_class(**ckpt['hyper_parameters'], training=False)
    model.model.load_state_dict(ckpt['params'], strict=False)

    # loop through params to ensure that ALL requires_grad are set to false
    for p in model.parameters():
        p.requires_grad = False

    # wrap model with FSDP
    #%%
    model = FSDP(model, 
                    cpu_offload=CPUOffload(offload_params=True),
                    # limit_all_gathers=True, # only in v2.0+
                    auto_wrap_policy=custom_autowrap_policy,
                    # auto_wrap_policy=always_wrap_policy,
                    # auto_wrap_policy=functools.partial(
                    #        size_based_auto_wrap_policy,
                    #        min_num_params=int(1e8))
                )

    model.eval()
    # barrier to ensure all processes have loaded the model
    torch.distributed.barrier()
    # %%

    logger.info("Model has been loaded")
    ################## INFERENCE ##################
    if args.rank == 0: os.makedirs(args.outpdb, exist_ok=True)

    item = valset[92] #valset[0]
    result = []
    for j in tqdm.trange(args.samples, disable=args.rank != 0):
        batch = collate_fn([item])
        batch = tensor_tree_map(lambda x: x.cuda(), batch)  
        start = time.time()
        torch.distributed.barrier()
        prots = model.inference(batch, as_protein=True, noisy_first=args.noisy_first,
                    no_diffusion=args.no_diffusion, schedule=schedule, self_cond=args.self_cond)
        result.append(prots[-1])
    # %%
