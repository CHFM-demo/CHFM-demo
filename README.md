# CHFM

This directory contains the implementation of CHFM for continual learning with
T5. It includes the Householder adapter, feature-memory loss, accuracy-based
evaluation, task configurations, and training scripts. Ablation variants,
visualization code, parameter-orthogonality loss, and ROUGE evaluation are not
included.

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run task order 1 on GPU 4 with seed 42:

```bash
./run_order1_seed42_gpu4.sh /path/to/t5-large
```

The script expects the continual-learning benchmark at `../CL_Benchmark` by
default. Set `DATA_DIR` to override this location.

## Acknowledgements

This implementation is developed with reference to the following work and
repositories:

- [O-LoRA](https://github.com/cmnfriend/O-LoRA)
- [ASO-LoRA](https://github.com/736619821/ASO-LORA)
