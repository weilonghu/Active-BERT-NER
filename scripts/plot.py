import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll', help="Directory containing the dataset")


def plot(files, model_dir, labels):

    color = ['cornflowerblue', 'turquoise', 'darkorange', 'red', 'teal', 'blueviolet', 'black', 'green', 'slategray', 'brown']
    markers = ['<', 's', 'o', '*', 'x', '^', 'd', 'v', 'p', 'h', '1', '2', '3', '4', '>']

    plt.clf()
    for i, npy_file in enumerate(files):
        f1s = np.load(os.path.join(model_dir, npy_file))
        batchs = np.arange(1, len(f1s) + 1)
        plt.plot(batchs, f1s, color = color[i], marker=markers[i], markevery=10, lw=1, label=labels[i])

    plt.xlabel('Batch')
    plt.ylabel('Val F1')
    plt.ylim([0.0, 1.0])
    plt.xlim([0, 70])
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'f1.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.dataset)

    label_map = {
        'random_select': 'Random',
        'least_confidence': 'Least',
        'token_entropy': 'Entropy'
    }

    # Find all .npy files
    files = []
    labels = []
    for f in os.listdir(model_dir):
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '.npy':
            files.append(f)
            labels.append(label_map[file_name])

    plot(files, model_dir, labels)
