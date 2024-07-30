# BrainBuildingPaper
A repository for the software used in the paper "Building a small brain with a simple stochastic generative model"

## Installtion, dependencies and requirements
The repository is implemented in Python (version 3.9).
It was developed on a Windows 10 computer. Some of the logic is intended to run on a single machine (and there may be system-speific issues such as path syntax, but the most is system-agnostic). These parts can run on a standard computer in a reasonable time.

There are parts that require a computing cluster for parallel computations, and were specifically developed for an IBM LSF cluster.

To install the dependencies run the following command in the terminal: `pip install -r requirements.txt`. This should take a few minutes.

## Short description of files and folders
### CElegansData
Contains the data (raw and processed) that is used for analyses.
### Figures
Contains the figures in the paper, including files and saved calculations to create them. Ignored by git due to size limits.
### ParamFiles
Contains a parameters file that is used to train models (the values for the grid search in the plane of S- and beta)
### SavedOutputs
Contains output files of distributed runs over the WIS computing cluster. They store calculations that were run in parallel. Ignored by git due to size limits.
### c_elegans_constants.py
Constants that are used throughout.
### c_elegans_data_parsing.py
Preprocess and parse raw data files.
### c_elegans_independent_model_training.py
Train single-epoch models with statiscally independent synapses.
### c_elegans_reciprocal_model_training.py
Train models with statiscally dependent reciprocal synapses.
### CElegansNeuronsAdder.py
A class that implements the operator for neurogenesis throughout development.
### cluster_cmds.txt
Explanations and commands for running calculations and analyses over an IBM LSF computing cluster.
### compact_model_representation.py
Compress models by quantizing the S+ matrix.
### ConnectomeDeveloper.py
A class that implements the model (grows a synthetic connectome over time using operators that implement the building rules).
### distance_model.py
Train a model that relies on distances between cell bodies and predict using it.
### er_block_model.py
Train a model that relies on distances between cell bodies and predict using it.
Implements the agglomerative clustering algorithm for inferring neuronal cell types.
### graph_statistics.py
Calculate various graph measures.
### paper_figures.py
Generate all the paper figures.
Note that the path parameters in all functions (e.g. `out_path`, `saved_calcs_path`) must be specified to an existing path
on the machine.
Additionally, the figures may require outputs that are pre-calculated and stored in the SavedOutputs folder, which is not under version control due to
its size. All necessary files can be calculated following the instructions in the file cluster_cmds.txt. However, they require parallel computations over a cluster of computers, and their generation may take time.
Figure specific calculations (to which the flags `is_saved` and `do_save`, as well as the path `saved_calcs_path` refer) are not under version control, but can be computed locally given the necessary files in SavedOutputs, stored and reused.
### SynapsesAdder.py
A class that implements the operator for synaptic formation throughout development.
### SynapsesRemover.py
A class that implements the operator for synaptic pruning throughout development.
### train_c_elegans_independent_model.py
Run the distributed calculations over an IBM-LSF computing cluster for training the single-epoch model with statistically independent synapses and submodels, compress models, simulate them many times and more.
### train_c_elegans_reciprocal_model.py
Run the distributed calculations over an IBM-LSF computing cluster for training models with statistically dependent synapses, single and multiple epochs.
