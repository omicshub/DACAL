# DACAL
Batch correction and deep adaptive clustering method for single-cell RNA data 
# Requirements
Python --- 3.7.11

Pytorch --- 1.12.0

Sklearn --- 1.0.2

Numpy --- 1.21.6

# Run DACAL

Take the dataset mammary_epithelial as an example.

Decompress the expression matrix file in "DACAL/data/mammary_epithelial.zip".

 Run the following command to train the DACAL model:
 
    CUDA_VISIBLE_DEVICES=0 py run.py --task mammary_epithelial 

To infer the labels after training, run the following command:

    CUDA_VISIBLE_DEVICES=0 py run.py --actions infer_latent --task mammary_epithelial --init_model sp_latest

We can obtain the predicted clustering labels and the ARI, NMI and SC metrics are also calculated in this step. 

# outputs

We can obtain the predicted clustering labels in the file '/DACAL/result/mammary_epithelial/e0/default/represent/y.csv'. The indicator of Deviation Ratio (DR) can be calculated by running "DACAL/metrics_DR.ipynb".

# Data availability

In the "DACAL/data" folder, we provide the compressed format of both of the real datasets used in the paper. If you want to use them, please download and decompress them first.

The simulated data is  used in the paper can be found at (https://drive.google.com/file/d/1PLjWY6mx2iCuvoC4XSVRGmR0A9TqdzjV/view?usp=drive_link)

# Using for new dataset

If you want to use DACAL to analyse other data, you can running the code as the following:

1.Please preprocess the data by Seurat referring to https://satijalab.org/seurat/articles/pbmc3k_tutorial, or use the codes provided by DACAL in the "preprocess" folder.

Whether we do the preprocessing step ourselves or with the DACAL's code in the "preprocess" folder, we need to get the following files:

a. The expression matrices of highly variable genes, where the data from different batches consists to different matrix.

b. The feature number and feature name of the highly variable genes and the cellnames of every expression matrix.

We can use "DACAL/preprocess_split.ipynb" to split the matrix to vectors. The vectors are the inneed input of DACAL.

2.Please supplement the file "DACAL/configs/data.toml" with information about the new dataset.

3.Please train the model:

       CUDA_VISIBLE_DEVICES=0 py run.py --task [datasetname]

4. To infer the labels after training, run the following command:

       CUDA_VISIBLE_DEVICES=0 py run.py --action infer_latent --task [datasetname] --init-model sp_latest

Then we can obtain the predict label and ARI, NMI and SC sore.


5.The indicator of Deviation Ratio (DR) can be calculated by running "DACAL/metrics_DR.ipynb".

# Available Options

There are some available options in DACAL.

1. The hyperparameter α is defined in "DACAL/run.py" on line 310, which is is indicated as "weight_concentration_prior". The larger α leads to the more uniform distribution of weights, which means the greater number of clusters. However, DACAL is robust to hyperparameter α changes.

2. The dimension of low-dimensional representation and network are given at "DACAL/configs/model.toml", which can be adjusted if necessary.

