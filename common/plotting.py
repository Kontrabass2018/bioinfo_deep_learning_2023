import numpy as np
import matplotlib.pyplot as plt
import datetime 
import random

def main():
    pass 

def random_color_generator():
    r = random.randint(0, 255)/256
    g = random.randint(0, 255)/256
    b = random.randint(0, 255)/256
    return (r, g, b)

def plot_tsne(X_tr_tsne, labels):
    fig, ax1 = plt.subplots(figsize=(8, 7))
    ax1.grid(visible=True,linestyle="--")
    ax1.set_axisbelow(True)
    markers_ = np.concatenate([['o',"v","^","<",">","8","p","s","h","D","P","X"] for i in range(10)])
    colors_by_ctype = [random_color_generator() for  i in range(len(np.unique(labels)))]
    for (i,lbl) in enumerate(np.unique(labels)):
        #defin_ = tcga_abbrevs.loc[tcga_abbrevs["abbrv"] == lbl,"def"].values[0] 
        #count_ = counts_df.loc[counts_df["c_type"] == lbl,"count"].values[0]
        #tag = f"{defin_} ({count_})"
        #print(X_tr_tsne[labels ==lbl,0])
        ax1.scatter(X_tr_tsne[labels ==lbl,0], 
                    X_tr_tsne[labels ==lbl,1], 
                    s = 8, marker=markers_[i], label = lbl)
    ax1.axis("equal")
    ax1.legend(bbox_to_anchor=(1,1),fontsize = 8)
    ax1.set_xlabel("TSNE1")
    ax1.set_ylabel("TSNE2")
    plt.tight_layout()
    plt.savefig(f"figures/figTSNE_{str(datetime.datetime.now())}.pdf")


def plot_umap(X_train_umap, X_test_umap, Y_train, Y_test, labels, s= 8, figsize = (6,5)):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.grid(visible=True,linestyle="--")
    ax1.set_axisbelow(True)
    markers_ = np.concatenate([['o',"v","^","<",">","8","p","s","h","D","P","X"] for i in range(10)])
    colors_by_ctype = [random_color_generator() for  i in range(len(np.unique(labels)))]
    for (i,lbl) in enumerate(np.unique(labels)):
        #defin_ = tcga_abbrevs.loc[tcga_abbrevs["abbrv"] == lbl,"def"].values[0] 
        #count_ = counts_df.loc[counts_df["c_type"] == lbl,"count"].values[0]
        #tag = f"{defin_} ({count_})"
        #print(X_tr_tsne[labels ==lbl,0])
        ax1.scatter(X_train_umap[Y_train ==lbl,0], 
                X_train_umap[Y_train ==lbl,1], 
                s = s, color = colors_by_ctype[i], linewidth = 0.5, marker=markers_[i], label = lbl)
        ax1.scatter(X_test_umap[Y_test ==lbl,0], 
                X_test_umap[Y_test ==lbl,1], 
                s = s, edgecolors = colors_by_ctype[i], color ="white", linewidth = 0.5, marker=markers_[i])

    ax1.axis("equal")
    ax1.legend(bbox_to_anchor=(1,1),fontsize = 8)
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    #fig.tight_layout()
    plt.savefig(f"figures/figUMAP_{str(datetime.datetime.now())}.pdf")

def plot_learning_curves(trl, tstl, trc, tstc, tpm_data, X_train, X_test):
    steps = np.arange(len(trl))
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 8))
    axes[0].plot(steps, trl, label = "train")
    axes[0].plot(steps, tstl, label= "test")
    axes[1].plot(steps, trc * 100, label = "train")
    axes[1].plot(steps, tstc * 100, label= "test")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[1].set_ylabel("Pearson Correlation")
    axes[1].set_ylim((0,100))
    axes[1].set_xlabel("Gradient step")
    axes[0].legend()
    axes[0].set_title(f"Learning curves of DNN on ML data\nN={tpm_data.shape[1]}, N(train)={X_train.shape[0]}, N(test)={X_test.shape[0]}")
    plt.savefig(f"figures/DNN_learning_curves{str(datetime.datetime.now())}.pdf")

def plot_ae_performance(mm, X_test):
    y_tst_out = mm(X_test)
    outs = y_tst_out.flatten().detach().numpy()
    trues = X_test.flatten().detach().numpy()
    corr =  pearsonr(outs,trues).statistic
    plt.figure(figsize = (9,7))
    plt.grid(visible =True, alpha = 0.5, linestyle = "--")
    plt.plot([0,1],[0,1], color = "blue", alpha =0.5, linestyle = "--")
    plt.hexbin(outs, trues, bins = "log")
    plt.xlabel("Predicted Expressions (normalised TPM)")
    plt.ylabel("True expressions")
    plt.colorbar(label='log10(N)')
    plt.axis("equal")
    plt.title(f"Auto-Encoder performance of reconstruction on test set.\nPearson Correlation: {round(corr,4)}")
    
if __name__ == "__main__":
    main()