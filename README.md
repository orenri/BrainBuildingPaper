# BrainBuildingPaper
A repository for the software used in the paper "Building a small brain with a simple stochastic generative model"

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
### SynapsesAdder.py
A class that implements the operator for synaptic formation throughout development.
### SynapsesRemover.py
A class that implements the operator for synaptic pruning throughout development.
### train_c_elegans_independent_model.py
Run the distributed calculations over an IBM-LSF computing cluster for training the single-epoch model with statistically independent synapses and submodels, compress models, simulate them many times and more.
### train_c_elegans_reciprocal_model.py
Run the distributed calculations over an IBM-LSF computing cluster for training models with statistically dependent synapses, single and multiple epochs.
