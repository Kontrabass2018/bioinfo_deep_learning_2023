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

class MLSurvDataset(MLdataset):
    def __init__(self, data, samples, genes, labels, survt, surve):
        super(MLSurvDataset, self).__init__(data, samples, genes, labels)
        self.surve = survt
        self.survt = surve
        
def load_datasets():
    return dict([("TCGA", load_tcga()), 
                 ("BRCA", load_tcga_brca()),
                 ("TALL", load_target_all()),
                 ("LAML", load_leucegene()) 
                 ])

def load_leucegene():
    infile = "./data/leucegene_GE_CDS_TPM_clinical.h5"
    inf = h5py.File(infile, "r")
    tpm_data = np.log10(inf["data"][:,:] + 1)
    genes = np.array(inf["genes"][:], dtype = str)
    samples = np.array(inf["samples"][:], dtype = str)
    labels = np.array(inf["labels"][:], dtype = str)
    survt = np.array(inf["survt"][:], dtype = int)
    surve = np.array(inf["surve"][:], dtype = int)
    
    return MLSurvDataset(tpm_data, samples, genes, labels, survt, surve)

def load_tcga(inpath = "./data/TCGA_TPM_hv_subset.h5"):
    dataset = h5py.File(inpath,"r")
    expr_data = dataset['data'][:,:] 
    labels = np.array(dataset["labels"][:], dtype = str)
    samples = np.array(dataset["rows"][:], dtype = str)
    genes = np.array(dataset["cols"][:], dtype = str)
    return MLdataset(expr_data, samples, genes, labels) 

def load_tcga_brca(inpath = "./data/TCGA_BRCA_fpkm_hv_norm_PAM50.h5"):
    dataset = h5py.File(inpath,"r")
    expr_data = dataset['data'][:,:] 
    labels = np.array(dataset["labels"][:], dtype = str)
    samples = np.array(dataset["samples"][:], dtype = str)
    genes = np.array(dataset["genes"][:], dtype = str)
    return MLdataset(expr_data, samples, genes, labels) 

def load_target_all(inpath = "./data/TARGET_ALL_264_norm_tpm_lab.h5"):
    dataset = h5py.File(inpath,"r")
    expr_data = dataset['data'][:,:] 
    labels = np.array(dataset["labels"][:], dtype = str)
    samples = np.array(dataset["samples"][:], dtype = str)
    genes = np.array(dataset["genes"][:], dtype = str)
    return MLdataset(expr_data, samples, genes, labels) 

if __name__ == "__main__":
    main()