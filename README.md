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
[http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz](http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz).To download and unpack the data, first enter the root folder of the cloned repository and then run the following:

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

Next, install additional required packages using the following command:
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```


###### Note:
The installed versions of `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, and `torch_spline_conv` must match the 
installed versions of `torch` and CUDA. If issues persist after the previous steps, try reinstalling the following packages using the following command:

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
```
where TORCH should be replaced with the correct torch version (`1.13.1`) above, and CUDA should be replaced with the correct CUDA version (`cu117`) above.


###### Note: 
If you installed your own NVIDIA Drivers / CUDA Toolkit, it may be important to manually uninstall the 
Pytorch CUDA tools that are downloaded when installing ```torch```. This can be accomplished with the followin command.
```
pip uninstall nvidia_cublas_cu11
```

### Miscellaneous
##### Accessing Google Cloud
To authorize access to google cloud, run the following:
```
gcloud auth login
```

Files can then be copied to and from from any buckets that are authorized with the same account using the following:
```
gsutil cp {path/to/file} gs://{bucket-name}
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
