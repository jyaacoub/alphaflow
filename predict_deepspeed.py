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
parser.add_argument('--pdb_id', nargs='*', default=[]) # pdbs to ignore
parser.add_argument('--tmax', type=float, default=1.0)
parser.add_argument('--templates', action='store_true')
parser.add_argument('--no_diffusion', action='store_true', default=False)
parser.add_argument('--self_cond', action='store_true', default=False)
parser.add_argument('--noisy_first', action='store_true', default=False)
parser.add_argument('--runtime_json', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
parser.add_argument('--low_mem', action='store_true', default=False)

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=2)

args = parser.parse_args()

import deepspeed
import torch, tqdm, os, json, time
import numpy as np

from collections import defaultdict
from alphaflow.config import set_inf
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
    long_sequence_inference=args.low_mem # modifies c.globals and c.models
) 
schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # init_method="tcp://localhost:12355"

    # Initializes the default distributed process group, and this will also initialize the distributed package
    deepspeed.init_distributed("nccl", rank=rank, world_size=world_size)

@torch.no_grad()
def main():
    if args.local_rank != 0:
        time.sleep(1)
    print(args.local_rank, args.world_size)
    setup(args.local_rank, args.world_size)

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
    
    ################## LOAD MODEL ##################
    logger.info("Loading the model")
    model_class = {'alphafold': AlphaFoldWrapper, 'esmfold': ESMFoldWrapper}[args.mode]

    ckpt = torch.load(args.weights, map_location='cpu')
    
    if args.low_mem:
        c = ckpt['hyper_parameters']['config']
        
        c.globals.offload_inference = True
        c.globals.use_lma = True # Use Staats & Rabe's low-memory attention algorithm.
        c.globals.use_flash = False # flash attention doesnt work well for long sequences
        
        c.model.template.offload_inference = True
        
        # not sure why this matters:
        c.model.template.template_pair_stack.tune_chunk_size = False
        c.model.extra_msa.extra_msa_stack.tune_chunk_size = False
        c.model.evoformer_stack.tune_chunk_size = False
        
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)
        if args.local_rank == 0: print(c.globals)
        
    model = model_class(**ckpt['hyper_parameters'], training=False)
    model.model.load_state_dict(ckpt['params'], strict=False)
    
    # loop through params to ensure that ALL requires_grad are set to false
    for p in model.parameters():
        p.requires_grad = False
    
    engine = deepspeed.init_inference(model,
                tensor_parallel={"enabled": True, "tp_size": args.world_size},
                zero={"stage": 3,
                     "offload_optimizer": {
                         "device": "cpu"
                         },
                     "contiguous_gradients": True, # prevents fragmentation of memory
                     "reduce_bucket_size": 5e8, # 500MB
                    "offload_param": {
                        "device": "cpu",
                        "buffer_size": 1e9,                 # 100MB - The size of the buffer for offloading parameters
                        "max_in_cpu": 1e10,                 # 10GB - The maximum number of elements in the CPU buffer
                    }
                }, 
                dtype=torch.bfloat16, # half percision
                replace_with_kernel_inject=True
                )
    model = engine.module
    model.eval()
    
    logger.info("Model has been loaded")
    ################## INFERENCE ##################
    if args.local_rank == 0: os.makedirs(args.outpdb, exist_ok=True)
    runtime = defaultdict(list)
    for i, item in enumerate(valset):
        if (args.pdb_id and item['name'] not in args.pdb_id):
            continue
        
        out_fp = f'{args.outpdb}/{item["name"]}.pdb'
        if os.path.exists(out_fp) and args.no_overwrite:
            if args.local_rank == 0: print(f"{i}:{item['name']} already exists, skipping...")
            continue
        result = []
        for j in tqdm.trange(args.samples, disable=args.local_rank != 0, ncols=10, 
                             desc=f"{i}:{item['name']}:{len(item['seqres'])}"):
            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)  
            start = time.time()
            prots = model.inference(batch, as_protein=True, noisy_first=args.noisy_first,
                        no_diffusion=args.no_diffusion, schedule=schedule, self_cond=args.self_cond)
            runtime[item['name']].append(time.time() - start)
            result.append(prots[-1])
        
        # Save pdb
        if args.local_rank == 0:
            with open(out_fp, 'w') as f:
                f.write(protein.prots_to_pdb(result))

    if args.runtime_json and args.local_rank == 0:
        with open(args.runtime_json, 'w') as f:
            f.write(json.dumps(dict(runtime)))

try:
    main()
except Exception as e:
    if args.local_rank == 0:
        raise e
    time.sleep(1)
    print(f"rank {args.local_rank} also ran into exception '{e}'")

