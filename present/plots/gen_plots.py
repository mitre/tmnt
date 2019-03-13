import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import argparse
from matplotlib.lines import Line2D
import matplotlib.cm as cm

def get_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--in_files', type=str, help='Input csv files')
    parser.add_argument('--plot_file', type=str, help='Output png plot')
    parser.add_argument('--names', type=str, help='Name list')
    parser.add_argument('--title', type=str, help='Title for Plot')
    return parser

def plot_trace_1(csv_file1, csv_file2, fig, ax, r=0,g=0,b=1,show=True):
    df1  = pd.read_csv(csv_file1)
    df2  = pd.read_csv(csv_file2)
    df1['Source'] = 'LogNormal'
    df2['Source'] = 'vMF'
    colors1 = df1.Epoch.apply(lambda x: (r,g,b,(x+1)/len(df1.Epoch))).tolist()
    colors2 = df2.Epoch.apply(lambda x: (g,b,r,(x+1)/len(df2.Epoch))).tolist()    
    df = pd.concat([df1, df2])
    colors1.extend(colors2)
    x = df['PPL']
    y = df['NPMI']
    l = df['Source']
    ax.scatter(x,y, label=l, c=colors1)
    ax.legend()
    #df.plot(kind='scatter', x='PPL', y='NPMI', color=colors1)


def plot_trace(csv_files, fig, ax, names, cmap_names, title, show=True):
    cmaps = [cm.get_cmap(cmn) for cmn in cmap_names]
    custom_lines = [Line2D([0], [0], color=cmm(0.5), lw=4) for cmm in cmaps]
    dfs = [pd.read_csv(f) for f in csv_files]
    #names = ['LogNormal', 'vMF']
    for (dd,n) in zip(dfs,names):
        dd['Source'] = n
    df = pd.concat(dfs)    
    groups = df.groupby('Source')
    i = 0
    a_names = []
    for name, group in groups:
        a_names.append(name)
        ax.scatter(group.PPL, group.NPMI, c=group.Epoch, cmap=cmap_names[i])
        i += 1
    ax.set_xlabel('Perplexity', fontsize=16)
    ax.set_ylabel('NPMI', fontsize=16)
    ax.set_title(title,fontsize=18)
    ax.legend(handles=custom_lines, labels=a_names, loc='upper right')
    


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    fig, ax = plt.subplots()
    names = args.names.split(',')
    files = args.in_files.split(',')
    colors = ['Blues','Oranges','Greens', 'Spectral']
    assert(len(files) == len(names))
    plot_trace(files, fig, ax, names, colors, args.title)
    show = False
    if show:
        plt.show()
    else:
        plt.savefig('pngs/'+args.plot_file, dpi=300)
