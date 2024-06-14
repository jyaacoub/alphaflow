# Setting up DeepSpeed on compute canada for AlphaFlow
See [issue#84](https://github.com/jyaacoub/MutDTA/issues/84#issuecomment-2059882818) on how to set up alphaflow.

The steps are fairly simple after installing alphaflow:
1. Load correct modules
2. pip install deepspeed
3. test that environment worked by sallocing to GPU node


### 1. load modules

**MPI python bindings and cuda are required by DeepSpeed for the pip install:**
```
module load mpi4py/3.1.3
module load cuda/11.4
```

### 2. pip install deepspeed
Make sure to load up the environment you created earlier for alphaflow

From the yml for [OpenFoldv1.0.1](https://github.com/aqlaboratory/openfold/blob/42e71db7fa327e0810eb0e371abc9f82aa9b7a6a/environment.yml) they specify v0.5.10, but I was able to get it working with v0.12.4 so we will stick to that.

```
pip install deepspeed==0.12.4
```

This might gives some errors about rust package manager not being installed but at the end it will still work out nicely (some of the versions are different though):
```
Successfully installed annotated-types-0.6.0+computecanada deepspeed-0.12.4 hjson-3.1.0+computecanada ninja-1.11.1+computecanada py-cpuinfo-9.0.0+computecanada pydantic-2.5.2+computecanada pydantic-core-2.14.5+computecanada pynvml-11.5.0+computecanada
```

**Also note we need huggingface's transformers module for this:**
```
pip install transformers==4.39.3
```

### 3. Test evironment
```
ds_report
```

1. Connect to node with a GPU (just use salloc)
2. activate the environment you just created
3. run `ds_report`, the output should match the following 
![alt text](<ds_report.png>)

### 4. Running Deepspeed
Example `multi_run_alphaflow.sh` script can be modified for your purposes. HOWEVER, for LMA (only neccessary for long sequences; 1200+) you have to make a small modification to the installed OpenFold libraries as mentioned in https://github.com/aqlaboratory/openfold/pull/435