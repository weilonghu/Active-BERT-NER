import numpy as np
import os
import argparse
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll', help="Directory containing the dataset")


def lineplot(files, model_dir, labels):

    values = [np.load(os.path.join(model_dir, npy_file)) for npy_file in files]
    values = np.array(values).transpose()

    x = np.arange(len(values))
    data = pd.DataFrame(values, x, columns=labels)
    ax = sns.lineplot(data=data, palette="tab10", linewidth=2)
    ax.set(ylim=(0.6, 1.0), xlim=(10, 60))
    ax.set(xlabel='Batch', ylabel='Val F1')
    ax.get_figure().savefig(os.path.join(model_dir, 'f1.png'))


def barplot(labels):

    # Load the example Titanic dataset
    titanic = sns.load_dataset("titanic")

    # Draw a nested barplot to show survival for class and sex
    g = sns.catplot(x="class", y="survived", hue="sex", data=titanic,
                    height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("survival probability")


def to_csv(self, files, labels):
    values = [np.load(os.path.join(model_dir, npy_file)) for npy_file in files]
    batches = [np.arange(len(values[0]))]
    values = np.array(batches + values).transpose()
    df = pd.DataFrame(data=values, columns=['Batch'] + labels)


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.dataset)

    label_map = {
        'random_select': 'Random',
        'least_confidence': 'Least',
        'token_entropy': 'Entropy',
        'random_select_crf': 'Random+CRF',
        'least_confidence_crf': 'Least+CRF',
        'token_entropy_crf': 'Entropy+CRF'
    }

    # Find all .npy files
    files = []
    labels = []
    for f in os.listdir(model_dir):
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '.npy':
            files.append(f)
            labels.append(label_map[file_name])

    lineplot(files, model_dir, labels)
    barplot(labels)
