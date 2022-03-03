## Hypergraph representation

`python hypersets.py`


Note:
[`baselines.py`](baselines.py) assumes the presence of two baseline repos: [HyperGCN](https://github.com/malllabiisc/HyperGCN) and [HGNN](https://github.com/iMoonLab/HGNN). If one wishes to run these baseline models, their directories should be named as `hypergcn` and `hgnn`, respectively, and should be placed in the parent directory of the current repo, as indicated in the [_init_paths.py script](_init_paths.py).


## Data processing

[data.py](data.py) extracts and processes raw data. For instance the [Cora Information Extraction data](https://people.cs.umass.edu/mccallum/data.html).

For an example of processed data please see the [CiteSeer data](data/citeseer6cls3703.pt) ([citeseer.pt](data/citeseer.pt) contains the same hypergraph but with reduced feature dimension).
