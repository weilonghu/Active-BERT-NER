import os
import argparse
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='conll', help="Directory containing the dataset")
parser.add_argument('--xlim', type=str, default='20,60', help='Range of x axis')
parser.add_argument('--ylim', type=str, default='0.6,1.0', help='Range of y axis')


def lineplot(model_dir, xlim, ylim):

    # Support 8 kinds of lines
    dash_styles = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1, 1.5, 1),
        (5, 1, 1, 1),
        (5, 1, 2, 1, 2, 1),
        (2, 2, 3, 1.5),
        (1, 2.5, 3, 1.2)]

    df = pd.read_csv(os.path.join(model_dir, 'val_f1.csv'), index_col=0)
    ax = sns.lineplot(data=df, palette="tab10", linewidth=2, dashes=dash_styles)
    ax.set(ylim=ylim, xlim=xlim)
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

    xlim = [int(x) for x in args.xlim.split(',')]
    ylim = [float(x) for x in args.ylim.split(',')]

    lineplot(model_dir, xlim=tuple(xlim), ylim=tuple(ylim))

    if os.path.exists(os.path.join(model_dir, 'test_f1.csv')):
        barplot(model_dir)
