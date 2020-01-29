import os
import argparse
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll', help="Directory containing the dataset")


def lineplot(model_dir):

    df = pd.read_csv(os.path.join(model_dir, 'val_f1.csv'), index_col=0)
    ax = sns.lineplot(data=df, palette="tab10", linewidth=2)
    ax.set(ylim=(0.6, 1.0), xlim=(10, 60))
    ax.set(xlabel='Batch', ylabel='Val F1')
    ax.get_figure().savefig(os.path.join(model_dir, 'val_f1.png'))


def barplot(model_dir):

    df = pd.read_csv(os.path.join(model_dir, 'test_f1.csv'))

    g = sns.catplot(x="strategy", y="f1", hue="class", data=df,
                    height=4, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("Test F1")
    g.set(ylim=(0.8, 0.92))
    g.savefig(os.path.join(model_dir, 'test_f1.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.dataset)

    lineplot(model_dir)
    barplot(model_dir)
