import numpy as np
import matplotlib.pyplot as plt
import json
import torchvision
import torchvision.transforms.functional
import seaborn as sns
import pandas as pd
from torchvision.utils import make_grid
from pathlib import Path
from toolkit.utils import plt_settings, TryExcept

plt.rcParams["savefig.bbox"] = 'tight'


@TryExcept()
@plt_settings()
def show(imgs, save_dir=Path(''), name=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig(save_dir / f'{"show" if name is None else name}.png', dpi=200)
    plt.close()


@TryExcept()
@plt_settings()
def plot_txt(file_path, keyword, save_dir=Path(''), name=None):
    with open(file_path) as f:
        first_line = f.readline().strip()
        first_dict = json.loads(first_line)

        keys = [key for key in first_dict if keyword in key]

        data = [first_dict] + [json.loads(line) for line in f]

    epochs = [d['epoch'] for d in data]

    for key in keys:
        values = [float(d[key]) for d in data]

        plt.plot(epochs, values, label=key.capitalize())

    plt.xlabel('Epoch')
    plt.ylabel(keyword.capitalize())
    plt.title(f'{keyword.capitalize()} Curves')
    plt.legend()
    plt.savefig(save_dir / f'{"show" if name is None else name}.png', dpi=200)
    plt.close()


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


@TryExcept()
@plt_settings()
def tsne_plot(tsne, labels, save_dir, fname):
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    class_name = {0: 'Benign', 1: 'Malignant'}

    colors_per_class = {
        0: [0, 0, 255],
        1: [255, 0, 0]}

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[label], dtype=float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color.reshape(1, -1), label=class_name[label])

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    if save_dir is not None:
        fig.savefig(save_dir / f'{fname}-tsne.png', dpi=200)


@TryExcept()
@plt_settings()
def plot_confusion_matrix(cm, name, save_dir=Path()):
    cm = pd.DataFrame(cm, name, name)
    plt.figure(figsize=(9, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig(save_dir / "best_confusion_matrix.png")
    plt.close()


@TryExcept()
@plt_settings()
def plot_cam(visualization, save_dir=Path(), patient_id: str = None, disease_type: str = None, name: str = None,
             predict: str = None):
    plt.imshow(visualization)
    plt.title("ID: {}  type: {} pred: {}".format(patient_id, disease_type, predict))
    plt.show()
    plt.savefig(save_dir / f"{name}-cam.png")
    plt.close()


