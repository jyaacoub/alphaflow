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

attn_method = parser.add_mutually_exclusive_group()
attn_method.add_argument('--lma', action='store_true', default=False, help="Uses bfloat16 and LMA")
attn_method.add_argument('--flash', action='store_true', default=False, help="Uses bfloat16 and flash attention (requires CUDA >= 11.6 and torch >= 1.12)")
parser.add_argument('--chunk_size', type=int, default=None, 
                    help="chunk size for reducing memory overhead (lower=less mem; 4 is usually good)")

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=2)

args = parser.parse_args()

import socket
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
torch.set_float32_matmul_precision(("medium" if args.lma or args.flash else 'high'))
    

config = model_config(
    'initial_training',
    train=False, 
    low_prec=True,
    long_sequence_inference=args.lma # modifies c.globals and c.models to use low-mem attention
) 
schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # 0 means we ask the OS to bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.

def setup(rank, world_size, shared_port=29500):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(shared_port)  # Use a centrally managed shared port

    # Attempt to initialize distributed processing
    try:
        deepspeed.init_distributed("nccl", rank=rank, world_size=world_size)
    except RuntimeError as e:
        print(f"Failed to initialize on port {shared_port}: {str(e)}")
        if rank == 0:
            # Only retry for rank 0 or a central node if applicable
            new_port = find_free_port()
            print(f"Retrying on new port {new_port}")
            os.environ['MASTER_PORT'] = str(new_port)
            deepspeed.init_distributed("nccl", rank=rank, world_size=world_size)

@torch.no_grad()
def main():
    if args.local_rank != 0:
        time.sleep(1)
    print(args.local_rank, args.world_size)
    # Note: setup is not needed if we use deepspeed at CLI
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
    
    c = ckpt['hyper_parameters']['config']
    c.globals.chunk_size = args.chunk_size # setting to None means no chunking
    if args.lma:
        c.globals.offload_inference = True
        c.globals.use_lma = True # Use Staats & Rabe's low-memory attention algorithm.
        # Default to DeepSpeed memory-efficient attention kernel unless use_lma is explicitly set
        # evo attn is hard to setup
        #c.globals.use_deepspeed_evo_attention = True if not c.globals.use_lma else False

        c.globals.use_flash = False # flash attention doesnt work well for long sequences
        
        c.model.template.offload_inference = True
        
        # TUNING CHUNK SIZE IS IMPORTANT TO FIND THE RIGHT CHUNK SIZE FOR OUR MODEL 
        # SO THAT NO MEMORY ERRORS OCCUR
        # but it just wastes time for longer sequences (see openfold docs: https://github.com/aqlaboratory/openfold?tab=readme-ov-file#monomer-inference)
        c.model.template.template_pair_stack.tune_chunk_size = False
        c.model.extra_msa.extra_msa_stack.tune_chunk_size = False
        c.model.evoformer_stack.tune_chunk_size = False
        
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)
        if args.local_rank == 0: print(c.globals)
    elif args.flash:
        c.globals.use_flash = True
        
    model = model_class(**ckpt['hyper_parameters'], training=False)
    model.model.load_state_dict(ckpt['params'], strict=False)
    
    # loop through params to ensure that ALL requires_grad are set to false
    for p in model.parameters():
        p.requires_grad = False
    
    engine = deepspeed.init_inference(model,
                tensor_parallel={"enabled": True, "tp_size": args.world_size},
                zero={"stage": 3,
                     "offload_optimizer": {
                         "pin_memory": True, # pins mem to cpu
                        "device": "cpu",
                        },
                     "contiguous_gradients": True, # prevents fragmentation of memory
                     "reduce_bucket_size": 5e8, # 500MB
                    "offload_param": {
                        "buffer_size": 1e9,                 # 1GB - The size of the buffer for offloading parameters
                        "max_in_cpu": 22e9,                 # 18GB - The maximum number of elements in the CPU buffer
                        "pin_memory": True,
                        "device": "cpu",
                    }
                }, 
                dtype=(torch.bfloat16 if args.lma or args.flash else torch.float32),
                replace_with_kernel_inject=True,
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
        desc = f"{i}:{item['name']}:{len(item['seqres'])}"
        if os.path.exists(out_fp) and args.no_overwrite:
            if args.local_rank == 0: print(f"{desc} already exists, skipping...")
            continue
        result = []
        for _ in tqdm.trange(args.samples, disable=args.local_rank != 0, ncols=100, desc=desc):
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

