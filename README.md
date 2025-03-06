# Molecular Graph Transformer (MGT)

This repository contains the implementation of the Molecular Graph Transformer (MGT) that can 
be used to predic material properties. There are two functionalities provided with this package:

- Train and Test a MGT model for your own dataset
- Run pretrained models to predict material properties for new materials

## Table of Contents
* [Usage and Examples](#-usage-and-examples)
  * [Installation](#-installation)
  * [Dataset](#-dataset)
  * [Using Pre-Trained Models](#-using-pre-trained-models)
  * [Training and Testing your own model](#-training-and-testing-your-own-model)
* [Funding](#-funding)

 <a name="usage"></a>
# Usage and Examples
-------------------------

<a name="install"></a>
## Installation
-------------------------
First create a conda environment:
Install the miniconda environment from https://docs.conda.io/en/latest/miniconda.html or the 
anaconda environment from https://www.anaconda.com/products/distribution

Now create a conda environment and activate it (substitute my_env with your preferred name):
```
conda create -n my_env
conda activate my_env
```

Now install the necessary libraries into the environment:

- [Pytorch](https://pytorch.org/)
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

- [Fabric](https://lightning.ai/docs/fabric/stable/)
```
conda install lightning -c conda-forge
```

- [Deep Graph Library](https://www.dgl.ai/)
```
conda install -c dglteam/label/th24_cu124 dgl
```

- [Pymatgen](https://pymatgen.org/)
```
conda install pymatgen -c conda-forge
```

<a name="dataset"></a>
## Dataset
-------------------------
A user needs the following to set-up a dataset to train, test and run inference using the model 
(all of this should be inside the same directory): 
 
1. `id_prop.csv` with name of the files of each structure and corresponding truth value/s for 
   each structure, 
2. `atom_init.json` a file to initialize the feature vector for each atom type. (can be found 
   in examples)
3. A folder contatining the structure files (accepted formats: `.cif`, `.xyz`, `.pdb`, `POSCAR`)
 
An example dataset can be found in [examples/example_data](examples/example_data), testing and 
training dataset have to be saved in different folders, each with all three components.

<a name="pretrain"></a>
## Using Pre-Trained Models
-------------------------
All the pre-trained models can be found in the [pretrained](pretrained), and they are saved with 
name of the task/dataset on which they were trained.

The [run.py](run.py) document can be used to get predictions using the pre-trained or 
custom-trained models. An example of using the pretrained model to predict the BANDGAP, HOMO and 
LUMO of the files in the example dataset (found in [examples/example_data](examples/example_data)
) is shown below:

```
run.py --root ./examples/example_data/ --model_path ./pretrained/ --model_name qmof.ckpt --out_dims 3 --out_names BANDGAP HOMO LUMO 
```

Help for the [run.py](run.py) file and its command line arguments can be obtained using ``` 
run.py -h ```

<a name="test_train"></a>
## Training and Testing your own model
-------------------------

### Training

To train your own model you'll first need to have made a [custom dataset](#-dataset), you can 
then run the training and validation by running:

```
python training.py --root ./examples/example_data/ --model_path ./saved_models/
```

You can specify the train and validation splits of the dataset by running:

```
python training.py --root ./examples/example_data/ --model_path ./saved_models/ --train_split 0.8 --val_split 0.2 
```

or the splits can also be set as absolute values (example assumes a dataset with 100 systems)

```
python training.py --root ./examples/example_data/ --model_path ./saved_models/ --train_split 80 --val_split 20
```

after running the [training.py](training.py) file, you will obtain:

- ```model.ckpt```: contains the MGT model at the last epoch (stored in the ```--model_path``` directory)
- ```lowest.ckpt```: contains the MGT model with the lowest validation error (stored in the ```--model_path``` directory)
- ```results.csv```: contains the training and validation losses (stored in the ```--save_dir``` 
  directory, if no ```--save_dir``` is specified it will save the results in ```.
  /output/train/```)

Help for the [training.py](training.py) file and its command line arguments can be obtained using ``` 
training.py -h ```

### Testing

To test your own model you'll first need to have made a [custom dataset](#-dataset), you can 
then run the testing by running:

```
python testing.py --root ./examples/example_data/ --model_path ./saved_models/ --model_name model.ckpt
```

after running the [testing.py](testing.py) file, you will obtain:

- ```results.csv```: contains the test results (structure ID, target value, predicted value, 
  overall error, per property error) for each structure in the test database (stored in 
  the ```--save_dir``` 
  directory, if no ```--save_dir``` is specified it will save the results in ```.
  /output/test/```)

Help for the [testing.py](testing.py) file and its command line arguments can be obtained using ``` 
testing.py -h ```

<a name="fund"></a>
# Funding
-------------------------
This project is funded by Queen Mary University of London
