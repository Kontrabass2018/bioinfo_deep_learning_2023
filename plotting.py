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


def plot_umap(X_train_umap, X_test_umap, labels, s= 8, figsize = (6,5)):
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

if __name__ == "__main__":
    main()