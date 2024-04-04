# This script takes a msa_dir path (flat path) and prepares a directory for alphaflow input containing:
# .
# ├── input.csv
# ├── msa_dir
# │   └── <prot_id>
# │       └── a3m
# │           └── <prot_id>.a3m -> ~[...]/msa_dir/<prot_id>.msa.a3m *
#[...]
# └── out_pdb
#
# * is a symbolic link to the original msa file from the msa_dir
#
# 4 directories, 2 files
# input_csv has columns name,seqres
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Process MSA directories for AlphaFold input.')
parser.add_argument('MSA_DIR_IN', type=str, help='Input directory containing MSA files.')
parser.add_argument('OUT_DIR', type=str, help='Output directory for processed files (symlink to originals).')
parser.add_argument('--n_splits', type=int, default=0, help='Number of splits for output (optional).')
args = parser.parse_args()

MSA_DIR_IN = args.MSA_DIR_IN
OUT_DIR = args.OUT_DIR
n_splits = args.n_splits

MSA_DIR_OUT = os.path.join(OUT_DIR, 'msa_dir')
os.makedirs(MSA_DIR_OUT, exist_ok=True)

def process_file(filename):
    # get prot_id and target seq
    prot_id, ext = os.path.splitext(filename)
    fp = os.path.join(MSA_DIR_IN, filename)
    dst = os.path.join(MSA_DIR_OUT, prot_id, 'a3m')
    t_seq = None

    if ext == '.a3m':
        with open(fp, 'r') as f:
            for i in range(2):  # skip header
                t_seq = f.readline()
            t_seq = t_seq.strip()

        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, f'{prot_id}.a3m')
        # create symlink in correct dir structure
        if not os.path.exists(dst):
            os.symlink(fp, dst)

    elif ext == '.aln':
        with open(fp, 'r') as f_aln:
            aln_seqs = [l for l in f_aln.readlines()]
        t_seq = aln_seqs[0].strip()

        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, f'{prot_id}.a3m')
        if not os.path.exists(dst):
            with open(dst, 'w') as f:
                for i, l in enumerate(aln_seqs):
                    f.write(f"> {i}\n")
                    f.write(l)
    else:
        return None

    return prot_id, t_seq

files = os.listdir(MSA_DIR_IN)
with Pool(processes=None) as pool:
    results = list(tqdm(pool.imap(process_file, files), total=len(files)))

# Filter out None results and create CSV
csv_p = os.path.join(OUT_DIR, 'input.csv')
results = [r for r in results if r is not None]
df = pd.DataFrame(results, columns=['name', 'seqres'])
df = df.sort_values(by='seqres', key=lambda x: x.str.len())
df.to_csv(csv_p, index=False)

# Save split CSV files if n_splits is provided and greater than 0
if n_splits > 0:
    chunk_size = len(df) // n_splits
    for i in range(n_splits):
        df_chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
        df_chunk.to_csv(os.path.join(OUT_DIR, f'input_{i}.csv'), index=False)

    # Handle any remainder
    if len(df) % n_splits != 0:
        df_chunk = df.iloc[n_splits*chunk_size:]
        df_chunk.to_csv(os.path.join(OUT_DIR, f'input_{n_splits}.csv'), index=False)
