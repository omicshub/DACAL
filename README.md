# DACAL
batch correction and deep adaptive clustering method for single-cell RNA data 
# Requirements
Python --- 3.7.11

Pytorch --- 1.12.0

Sklearn --- 1.0.2

Numpy --- 1.21.6

# Run DACAL
Take the dataset mammary_epithelialn we provided here as an example.

Decompress the expression matrix file in DACAL/data

 Run the following command to train the DACAL model:
$ CUDA_VISIBLE_DEVICES=0 py run.py --task mammary_epithelialn --exp e0
To infer the labels after training, run the following command:
$ CUDA_VISIBLE_DEVICES=0 py run.py --actions infer_latent --task mammary_epithelialn --exp e0  --init_model sp_latest
You can obtain the predicted clustering labels under the folder /data/mammary_epithelialn, and the ARI, NMI and SC metrics. 
