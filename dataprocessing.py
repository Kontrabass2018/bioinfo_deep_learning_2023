import numpy as np
import matplotlib.pyplot as plt
import datetime 
import h5py
def main():
    pass 
class MLdataset:
    def __init__(self, data, samples, genes, labels):
        self.labels = labels
        self.data = data 
        self.samples = samples 
        self.genes = genes 

def load_datasets():
    return dict([("TCGA", load_tcga()), 
                 ("BRCA", load_tcga_brca()),
                 ("ALL", load_target_all())])

def load_tcga(inpath = "/cours/a23_bin3002-a/cours/TP6/TCGA_TPM_hv_subset.h5"):
    dataset = h5py.File(inpath,"r")
    expr_data = dataset['data'][:,:] 
    labels = np.array(dataset["labels"][:], dtype = str)
    samples = np.array(dataset["cols"][:], dtype = str)
    genes = np.array(dataset["rows"][:], dtype = str)
    return MLdataset(expr_data, samples, genes, labels) 

def load_tcga_brca(inpath = "TCGA_BRCA_fpkm_hv_norm_PAM50.h5"):
    dataset = h5py.File(inpath,"r")
    expr_data = dataset['data'][:,:] 
    labels = np.array(dataset["labels"][:], dtype = str)
    samples = np.array(dataset["samples"][:], dtype = str)
    genes = np.array(dataset["genes"][:], dtype = str)
    return MLdataset(expr_data, samples, genes, labels) 

def load_target_all(inpath = "TARGET_ALL_264_norm_tpm_lab.h5"):
    dataset = h5py.File(inpath,"r")
    expr_data = dataset['data'][:,:] 
    labels = np.array(dataset["labels"][:], dtype = str)
    samples = np.array(dataset["samples"][:], dtype = str)
    genes = np.array(dataset["genes"][:], dtype = str)
    return MLdataset(expr_data, samples, genes, labels) 

if __name__ == "__main__":
    main()