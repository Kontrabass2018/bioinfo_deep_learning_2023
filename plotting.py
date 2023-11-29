import numpy as np
import matplotlib.pyplot as plt
import datetime 
def main():
    pass 
def plot_tsne(X_tr_tsne, labels):
    fig, ax1 = plt.subplots(figsize=(5, 7))
    ax1.grid(visible=True,linestyle="--")
    ax1.set_axisbelow(True)
    markers_ = np.concatenate([['o',"v","^","<",">","8","p","s","h","D","P","X"] for i in range(10)])
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

if __name__ == "__main__":
    main()