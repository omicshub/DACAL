# DACAL
batch correction and deep adaptive clustering method for single-cell RNA data 
# Requirements
Python --- 3.7.11

Pytorch --- 1.12.0

Sklearn --- 1.0.2

Numpy --- 1.21.6

# data avaluablity
In the "data" folder, we provide the compressed format of all the datasets used in the paper. If you want to use them, please download and decompress them first.

If you want to use DACAL to analyse other data, please preprocess the data by Seurat referring to https://satijalab.org/seurat/articles/pbmc3k_tutorial, or use the codes provided by DACAL in the "preprocess" folder.

Whether we do the preprocessing step ourselves or with the DACAL's code in the "preprocess" folder, we need to get the following files:

1. The expression matrices of highly variable genes, where the data from different batches consists to different matrix.

2. The cellnames of every expression matrix and the feature number and feature name of the highly variable genes.

After decompress or preprocess the data, we can split the expression matrices by running the DACAL/preprocess/preprocess_split.ipynb to obtain the input for DACAL.

# Run DACAL
Take the dataset mammary_epithelialn we provided here as an example.

Decompress the expression matrix file in DACAL/data

 Run the following command to train the DACAL model:
 
$ CUDA_VISIBLE_DEVICES=0 py run.py --task mammary_epithelialn --exp e0

To infer the labels after training, run the following command:

$ CUDA_VISIBLE_DEVICES=0 py run.py --actions infer_latent --task mammary_epithelialn --exp e0  --init_model sp_latest

You can obtain the predicted clustering labels under the folder /data/mammary_epithelialn, and the ARI, NMI and SC metrics. 
