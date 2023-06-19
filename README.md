# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

### Installation and setup

To clone the git repository, run the following 

```
git clone git@bitbucket.org:ml6team/biographs.git
```

#### Multiscale Interactoms
The multiscale interactome is hosted by Stanford University. All data is available at
[http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz](http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz).To download and unpack the data, please run the following:

```
mkdir multiscale_interactome
cd multiscale_interactome
wget http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz
tar -xvf data.tar.gz
```

#### Setup
Code is written in Python3. Please install the packages present in the requirements.txt file. You may use:

```
pip install -r requirements.txt
```

###### Note: 
If you installed your own NVIDIA Drivers / CUDA Toolkit, it may be important to manually uninstall the 
Pytorch CUDA tools that are downloaded when installing ```torch```. This can be accomplished with the followin command.
```
pip uninstall nvidia_cublas_cu11
```

###### Note:
The installed versions of `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, and `torch_spline_conv` must match the 
installed versions of `torch` and CUDA. If issues persist after installing the required packages using the `requirements.txt` file,try running the following command:

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
```
For example, if using `torch` version `1.13.1` with CUDA version `cu117`, run the following command:
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```


* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
