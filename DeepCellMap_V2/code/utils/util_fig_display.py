
from utils.util_colors_drawing import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from shapely.geometry import (
    MultiPoint,
    MultiPolygon,
    polygon,
    Point,
    LineString,
    Polygon,
)
from shapely import geometry

from matplotlib import cm
from matplotlib.colors import ListedColormap
from utils.util import np_to_pil


from config.html_generation_config import HtmlGenerationConfig



# Contain all the displaying functions of images 



def barplot_df(df, titre="K vecteur"):
    nb_col = df.shape[1]
    fig, axes = plt.subplots(nrows=nb_col, ncols=1, figsize=(20, 7 * nb_col))
    for i in range(nb_col):
        p = sns.barplot(x=df.index, y=df[df.columns[i]], ax=axes[i]).set(
            title=titre + " - " + df.columns[i]
        )
        axes[i].set_title(df.columns[i] + " - " + titre, size=24)


# Visualisation d'une image rgb
def display_rgb(image1, title="image1", grid=False, figsize=(7, 7), pathsave = None):
    """
    Input : Image rgb, title
    """
    f = plt.figure(figsize=figsize)
    plt.imshow(image1)
    plt.title(title, fontsize=15)
    if grid:
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, image1.shape[1], ROW_TILE_SIZE))
        ax.set_yticks(np.arange(0, image1.shape[0], COL_TILE_SIZE))
        # Gridlines based on minor ticks
        ax.grid(which="major", color="w", linestyle="-", linewidth=1)
    plt.show()
    if pathsave is not None:
        f.savefig(pathsave,
                    facecolor='white',
                    dpi="figure",
                    bbox_inches='tight',
                    pad_inches=0.1)
    return f


# Visualisation d'une image rgb et un mask
def display_rgb_mask(
    rgb_img,
    mask,
    titre1="image1",
    titre2="image2",
    use_background=False,
    legend=False,
    state_names="",
    crop_grid=True,
    figsize=(10, 7),
    cmap=CMAP,
):
    """
    TODO
    """
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(titre1, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")
    if crop_grid:
        ax = plt.gca()
        ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
        ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
        ax.grid(color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 2, 2)
    ff = plt.imshow(mask, cmap=cmap, vmin=0, vmax=256)
    if use_background:
        plt.imshow(rgb_img, alpha=0.6)
    plt.title(titre2, fontsize=20)
    if legend:
        values = np.unique(mask.ravel())
        colors = [ff.cmap(ff.norm(value)) for value in values]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values[i])])
            for i in range(1, len(values))
        ]
        plt.legend(handles=patches)
    # else :
    # plt.colorbar(shrink = 0.5)

    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.show()
    return fig1


# Visualisation d'une image rgb et un mask
def display_rgb_mask_mask(
    image1,
    mask,
    mask_pred,
    titre1="image1",
    titre2="image2",
    titre3="image3",
    legend=False,
    state_names="",
    figsize=(20, 12),
):
    """
    TODO

    """
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    plt.subplot(1, 3, 1)
    plt.imshow(image1)
    plt.title(titre1, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(which="major", color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 3, 2)
    ff = plt.imshow(mask, cmap="Purples")
    plt.title(titre2, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")

    ax = plt.gca()
    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(which="major", color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 3, 3)
    # ff = plt.imshow(mask_pred, cmap=CMAP,vmin=0,vmax=256)
    ff = plt.imshow(mask_pred, cmap="Purples")
    plt.title(titre3, fontsize=20)
    if legend:
        values = np.unique(mask_pred.ravel())
        colors = [ff.cmap(ff.norm(value)) for value in values]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values[i])])
            for i in range(1, len(values))
        ]
        plt.legend(handles=patches)
    else:
        plt.colorbar(shrink=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()

    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(color="black", linestyle="-", linewidth=1)
    plt.show()


# Visualisation d'une image rgb et un mask
def display_rgb_classified_detected_cells(
    image1,
    mask,
    mask_pred,
    titre1="image1",
    titre2="image2",
    titre3="image3",
    legend=False,
    state_names="",
    figsize=(20, 12),
):
    """
    TODO

    """
    labels_name = state_names.copy()
    labels_name.append("detected")
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    plt.subplot(1, 3, 1)
    plt.imshow(image1)
    plt.title(titre1, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(which="major", color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 3, 2)
    ff = plt.imshow(mask, cmap="Purples")
    plt.title(titre2, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")

    ax = plt.gca()
    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(which="major", color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 3, 3)
    ff = plt.imshow(mask_pred, cmap=CMAP, vmin=0, vmax=256)
    plt.title(titre3, fontsize=20)

    values = np.unique(mask_pred.ravel())
    colors = [ff.cmap(ff.norm(value)) for value in values]
    patches = [
        mpatches.Patch(color=colors[i], label=labels_name[int(values[i]) - 1])
        for i in range(1, len(values))
    ]
    plt.legend(handles=patches)

    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()

    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(color="black", linestyle="-", linewidth=1)
    plt.show()


def display_rgb_mask_mask_mask(
    image1,
    mask,
    mask_pred,
    mask_3,
    titre1="image1",
    titre2="image2",
    titre3="image3",
    titre4="",
    legend=False,
    state_names="",
    figsize=(20, 12),
):
    """
    TODO

    """
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    plt.subplot(1, 4, 1)
    plt.imshow(image1)
    plt.title(titre1, fontsize=20)

    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()

    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 4, 2)
    ff = plt.imshow(mask, cmap="Purples")
    plt.title(titre2, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()

    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 4, 3)
    ff = plt.imshow(mask_pred, cmap="Purples")
    plt.title(titre3, fontsize=20)
    if legend:
        values = np.unique(mask_pred.ravel())
        colors = [ff.cmap(ff.norm(value)) for value in values]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values[i])])
            for i in range(1, len(values))
        ]
        plt.legend(handles=patches)
    # else :
    # plt.colorbar(shrink = 0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()

    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(color="black", linestyle="-", linewidth=1)

    plt.subplot(1, 4, 4)
    ff = plt.imshow(mask_3, cmap="Purples")
    plt.title(titre4, fontsize=20)
    plt.xlabel("x")
    plt.ylabel("y")
    ax = plt.gca()

    ax.set_xticks(np.arange(0, mask.shape[1], SIZE_CROP))
    ax.set_yticks(np.arange(0, mask.shape[0], SIZE_CROP))
    ax.grid(color="black", linestyle="-", linewidth=1)
    plt.show()

    plt.show()


def display_rgb_mask_mask_mask_mask_mask(
    rgb,
    eosin,
    mask1,
    mask2,
    mask3,
    mask4,
    mask5,
    mask6,
    titre1="rgb",
    titre2="eosin",
    titre3="mask",
    titre4="mask",
    titre5="mask",
    titre6="mask",
    titre7="mask",
    titre8="mask",
    figsize=(20, 12),
):
    """
    TODO

    """
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    plt.subplot(2, 5, 1)
    plt.imshow(rgb)
    plt.title(titre1, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 2)
    ff = plt.imshow(mask1, cmap="Purples")
    plt.title(titre3, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 3)
    ff = plt.imshow(mask2, cmap="Purples")
    plt.title(titre4, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 4)
    ff = plt.imshow(mask3, cmap="Purples")
    plt.title(titre5, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 7)
    ff = plt.imshow(mask4, cmap="Purples")
    plt.title(titre6, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 8)
    ff = plt.imshow(mask5, cmap="Purples")
    plt.title(titre7, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 9)
    ff = plt.imshow(mask6, cmap="Purples")
    plt.title(titre8, fontsize=20)
    plt.axis("off")

    plt.subplot(2, 5, 5)
    ff = plt.imshow(eosin, cmap="Purples")
    plt.title(titre2, fontsize=20)
    plt.axis("off")
    plt.show()


def display_rgb_mask_eosin(
    image1,
    mask,
    image_eosin,
    title="RGB",
    title2="Mask Prediction",
    title3="Eosin canal",
    add_square=False,
    colorbar=False,
    legend=False,
    state_names="",
    cmap="Purples",
    figsize=(20, 10),
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)
    plt.subplot(1, 3, 1)
    plt.imshow(image1)
    plt.title(title, fontsize=20)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    ff = plt.imshow(mask, cmap=cmap)
    # if add_square:
    # ax.add_patch(patches.Rectangle((int(mask_binaire1.shape[0]/2-int(SIZE_CROP/2)), int(mask_binaire1.shape[1]/2-int(SIZE_CROP/2))),SIZE_CROP,SIZE_CROP,edgecolor = 'blue',facecolor = 'red',fill=False) )

    if legend:
        values = np.unique(mask.ravel())
        colors = [ff.cmap(ff.norm(value)) for value in values]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values[i])])
            for i in range(1, len(values))
        ]
        plt.legend(handles=patches)
    if colorbar:
        plt.colorbar(ff, shrink=0.5)

    plt.title(title2, fontsize=20)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_eosin)
    plt.title(title3, fontsize=20)
    plt.axis("off")
    plt.show()


def display_rgb_mask_mask_eosin(
    image1,
    mask,
    mask2,
    image_eosin,
    title="rgb",
    title2="old mask",
    title3="new mask",
    title4="eosin",
    cmap="Purples",
    figsize=(20, 10),
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)
    plt.subplot(1, 4, 1)
    plt.imshow(image1)
    plt.title(title, fontsize=20)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    ff = plt.imshow(mask, cmap=cmap)
    plt.title(title2, fontsize=20)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    ff = plt.imshow(mask2, cmap=cmap)
    plt.title(title3, fontsize=20)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(image_eosin)
    plt.title(title4, fontsize=20)
    plt.axis("off")
    plt.show()


def display_eosin_mask_pred(
    img_eosin,
    img_rgb,
    mask_pred,
    state_names,
    figsize=(20, 20),
    title="Canal eosin",
    title2="Prediction of the network",
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    plt.subplot(1, 2, 1)
    plt.imshow(img_eosin)
    plt.title(title, fontsize=20)
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    ff = plt.imshow(mask_pred, alpha=0.6, cmap="YlOrBr")
    values = np.unique(mask_pred.ravel())
    colors = [ff.cmap(ff.norm(value)) for value in values]
    patches = [
        mpatches.Patch(color=colors[i], label=state_names[int(values[i])])
        for i in range(1, len(values))
    ]
    plt.legend(handles=patches)  # , bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0. )
    plt.title(title2, fontsize=20)
    plt.show()


def display_mask_on_rgb(
    img_rgb_background,
    mask_binaire1,
    mask_binaire2=None,
    use_background=True,
    alpha=0.8,
    titre="Mask on RGB img",
    add_square=False,
    liste_center_of_mass=None,
    grid=False,
    legend=False,
    state_names="",
    nb_cells_by_subpop=None,
    figsize=(7, 7),
    fontsize_title=20,
    cmap=CMAP,
):
    import matplotlib.patches as patches

    fig1, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if legend:
        state_names2 = state_names + ["disk"]
    # state_names2.insert(len(state_names) + 1, "disks")
    if use_background:
        ax.imshow(img_rgb_background, alpha=0.9)

    if cmap == CMAP:
        ff = ax.imshow(
            mask_binaire1,
            alpha=alpha * (mask_binaire1 > 0),
            cmap=cmap,
            vmin=0,
            vmax=255,
        )
    else:
        ff = ax.imshow(mask_binaire1, alpha=alpha * (mask_binaire1 > 0), cmap=cmap)

    if mask_binaire2 is not None:
        ax.imshow(mask_binaire2, alpha=0.2, cmap=cmap)

    if add_square:
        ax.add_patch(
            patches.Rectangle(
                (
                    int(mask_binaire1.shape[0] / 2 - int(SIZE_CROP / 2)),
                    int(mask_binaire1.shape[1] / 2 - int(SIZE_CROP / 2)),
                ),
                SIZE_CROP,
                SIZE_CROP,
                edgecolor="blue",
                facecolor="red",
                fill=False,
            )
        )

    if liste_center_of_mass is not None:
        for coord in range(len(liste_center_of_mass)):
            x, y = (
                liste_center_of_mass["x_coord"][coord] - dataset_config.roi_border_size,
                liste_center_of_mass["y_coord"][coord] - dataset_config.roi_border_size,
            )
            if (
                x > 0
                and y > 0
                and x < mask_binaire1.shape[0]
                and y < mask_binaire1.shape[1]
            ):
                ax.plot(y, x, marker="o", color="r", markersize=10, alpha=1)

    if legend:
        values = np.unique(mask_binaire1.ravel())
        colors = [ff.cmap(ff.norm(value)) for value in values]
        if nb_cells_by_subpop is not None:
            patches = [
                mpatches.Patch(
                    color=colors[i],
                    label=state_names[int(values[i])]
                    + " ("
                    + str(nb_cells_by_subpop[i - 1])
                    + " cells)",
                )
                for i in range(1, len(values))
            ]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names2[int(values[i])])
            for i in range(1, len(values))
        ]

        plt.legend(handles=patches, prop={"size": 10})

    if grid:
        print("La grid est True")
        # Major ticks
        ax.set_xticks(np.arange(0, mask_binaire1.shape[1], ROW_TILE_SIZE))
        ax.set_yticks(np.arange(0, mask_binaire1.shape[0], COL_TILE_SIZE))

        # Gridlines based on minor ticks
        ax.grid(which="major", color="black", linestyle="-", linewidth=1)
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        plt.axis("off")

    plt.title(titre, fontsize=fontsize_title)

    plt.show()

    return fig1
    # else:
    #     plt.colorbar(
    #         ff,
    #         shrink=0.5,
    #     )


def display_liste_mask_on_rgb(
    img_rgb_background,
    mask_binaire1,
    liste_mask,
    alpha=0.3,
    titre="RGB et plein de masks",
    add_square=False,
    legend=False,
    state_names="",
    nb_cells_by_subpop=None,
    figsize=(7, 7),
    cmap=CMAP,
):
    import matplotlib.patches as patches

    fig1, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.imshow(img_rgb_background)
    ff = ax.imshow(mask_binaire1, alpha=alpha, cmap=cmap, vmin=0, vmax=256)

    for idx_mask in range(len(liste_mask)):
        ax.imshow(liste_mask[idx_mask], alpha=0.3, cmap=cmap)

    if legend:
        values = np.unique(mask_binaire1.ravel())
        colors = [ff.cmap(ff.norm(value)) for value in values]
        print(values)
        print(state_names)
        if nb_cells_by_subpop is not None:
            patches = [
                mpatches.Patch(
                    color=colors[i],
                    label=state_names[int(values[i])]
                    + " ("
                    + str(nb_cells_by_subpop[i - 1])
                    + " cells)",
                )
                for i in range(1, len(values))
            ]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values[i])])
            for i in range(1, len(values))
        ]

        plt.legend(handles=patches)
    else:
        plt.colorbar(
            ff,
            shrink=0.5,
        )
    plt.title(titre, fontsize=20)
    return fig1


# Figure de visualisation avec gridspec 2 images
def display_mask_mask(
    image1,
    es,
    title="Mask 1",
    title2="Mask 2",
    colorbar=True,
    figsize=(10, 8),
    cmap="viridis",
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])
    f1_ax2 = fig1.add_subplot(spec1[0, 1])

    if colorbar:
        divider = make_axes_locatable(f1_ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    f1 = f1_ax1.imshow(image1, cmap=cmap)
    if colorbar:
        fig1.colorbar(f1, cax=cax, orientation="vertical")
    f1_ax1.set_title(title, fontsize=20)
    # f1_ax1.axis('off')

    if colorbar:
        divider = make_axes_locatable(f1_ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    f2 = f1_ax2.imshow(es, cmap=cmap)
    if colorbar:
        fig1.colorbar(f2, cax=cax, orientation="vertical")
    f1_ax2.set_title(title2, fontsize=20)
    f1_ax2.axis("off")
    plt.show()

# Figure de visualisation avec gridspec 2 images
def display_mask(
    image1,
    image_background=None,
    title="Mask",
    liste_center_of_mass=[],
    add_squares=False,
    add_numbers=False,
    add_center_of_mass=False,
    figsize=(8, 8),
    vmax=None,
    pathsave=None,
    cmap="Purples",
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])

    divider = make_axes_locatable(f1_ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if vmax is not None:
        f1 = f1_ax1.imshow(image1, cmap=cmap, vmin=np.min(image1), vmax=vmax)
    else:
        f1 = f1_ax1.imshow(image1, cmap=cmap, vmin=np.min(image1), vmax=np.max(image1))

    fig1.colorbar(f1, cax=cax, orientation="vertical")

    if image_background is not None:
        f1_ax1.imshow(image_background, alpha=0.5)

    if len(liste_center_of_mass) > 0:
        for coord in range(len(liste_center_of_mass)):
            x, y = (
                liste_center_of_mass["x_coord"][coord],
                liste_center_of_mass["y_coord"][coord],
            )
            if add_numbers:
                f1_ax1.text(y, x, str(coord), color="black", fontsize=12)
            if add_center_of_mass:
                f1_ax1.plot(y, x, "+", color="darkgreen", markersize=10)
            if add_squares:
                f1_ax1.add_patch(
                    patches.Rectangle(
                        (int(y - int(SIZE_CROP / 2)), int(x - int(SIZE_CROP / 2))),
                        SIZE_CROP,
                        SIZE_CROP,
                        edgecolor="blue",
                        facecolor="red",
                        fill=False,
                    )
                )

    f1_ax1.set_title(title, fontsize=20)

    if pathsave is not None:
        fig1.savefig(pathsave)

    else:
        plt.show()


# Figure de visualisation avec gridspec 2 images
def display_mask_tissueee(
    image1,
    title="Mask",
    liste_center_of_mass=[],
    add_squares=False,
    add_numbers=False,
    add_center_of_mass=False,
    figsize=(8, 8),
    savepath=None,
    cmap=CMAP_TISSU_SLIDE,
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)

    f1 = plt.imshow(image1, cmap=cmap, vmin=0, vmax=255)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, image1.shape[1], 32))
    ax.set_yticks(np.arange(0, image1.shape[0], 32))
    # Gridlines based on minor ticks
    # f1_ax1.grid(which='minor', color='w', linestyle='-', linewidth=10)
    ax.grid(which="major", color="black", linestyle="-", linewidth=1)

    ax.set_title(title, fontsize=20)
    values = np.unique(image1.ravel())
    colors = [f1.cmap(f1.norm(value)) for value in values]
    patches = [
        mpatches.Patch(color=colors[i], label=LABELS_TISSU_SLIDE[int(values[i])])
        for i in range(len(values))
    ]
    plt.legend(handles=patches, prop={"size": 20})

    plt.show()
    if savepath is not None:
        fig1.savefig(savepath)
        plt.close("All")
    plt.close("All")
    return fig1


# Figure de visualisation avec gridspec 2 images
def display_rgb_rgb(
    image1, image2, title="Image 1", title2="Image 2", figsize=(10, 8)
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])
    f1_ax2 = fig1.add_subplot(spec1[0, 1])

    f1_ax1.imshow(image1)
    f1_ax1.set_title(title, fontsize=20)
    f1_ax1.axis("off")

    f1_ax2.imshow(image2)
    f1_ax2.set_title(title2, fontsize=20)
    f1_ax2.axis("off")
    plt.show()


# Avec la colorbar
def display_mask_mask_cb(
    image1,
    es,
    title="Image avant dilatation",
    title2="Élément structurant",
    cmap="plasma",
    alpha1=1,
    alpha2=1,
):
    fig1 = plt.figure(figsize=(15, 8), constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])
    f1_ax2 = fig1.add_subplot(spec1[0, 1])

    divider = make_axes_locatable(f1_ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f1 = f1_ax1.imshow(image1, cmap=cmap, alpha=alpha1)
    fig1.colorbar(f1, cax=cax, orientation="vertical")
    f1_ax1.set_title(title, fontsize=20)
    f1_ax1.axis("off")

    divider = make_axes_locatable(f1_ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f2 = f1_ax2.imshow(es, cmap=cmap, alpha=alpha2)
    fig1.colorbar(f1, cax=cax, orientation="vertical")
    f1_ax2.set_title(title2, fontsize=20)
    f1_ax2.axis("off")

    plt.show()


# Figure de visualisation avec gridspec 2 images
def display_mask_mask_maks_cb(
    image1,
    es,
    image3,
    title="Image avant dilatation",
    title2="Élément structurant",
    title3="Image dilatée",
    cmap="plasma",
):
    fig1 = plt.figure(figsize=(19, 11), constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])
    f1_ax2 = fig1.add_subplot(spec1[0, 1])
    f1_ax3 = fig1.add_subplot(spec1[0, 2])

    divider = make_axes_locatable(f1_ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f1 = f1_ax1.imshow(image1, cmap=cmap, vmin=0, vmax=image1.max())
    fig1.colorbar(f1, cax=cax, orientation="vertical")
    f1_ax1.set_title(title, fontsize=20)
    f1_ax1.axis("off")

    divider = make_axes_locatable(f1_ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f2 = f1_ax2.imshow(es, cmap=cmap, vmin=0, vmax=es.max())
    fig1.colorbar(f1, cax=cax, orientation="vertical")
    f1_ax2.set_title(title2, fontsize=20)
    f1_ax2.axis("off")

    divider = make_axes_locatable(f1_ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f3 = f1_ax3.imshow(image3, cmap=cmap, vmin=0, vmax=image3.max())
    fig1.colorbar(f1, cax=cax, orientation="vertical")
    f1_ax3.set_title(title3, fontsize=20)
    f1_ax3.axis("off")

    plt.show()


# Figure de visualisation avec gridspec 4 images


def display_4_mask(
    image1,
    image2,
    image3,
    image4,
    title="Image 1",
    title2="Image 2",
    title3="Image 3",
    title4="Image 4",
):
    fig1 = plt.figure(figsize=(20, 15), constrained_layout=True)

    spec1 = gridspec.GridSpec(ncols=4, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])
    f1_ax2 = fig1.add_subplot(spec1[0, 1])
    f1_ax3 = fig1.add_subplot(spec1[0, 2])
    f1_ax4 = fig1.add_subplot(spec1[0, 3])

    f1_ax1.imshow(image1, cmap="plasma")
    f1_ax1.set_title(title, fontsize=20)
    f1_ax1.axis("off")

    f1_ax2.imshow(image2, cmap="plasma")
    f1_ax2.set_title(title2, fontsize=20)
    f1_ax2.axis("off")

    f1_ax3.imshow(image3, cmap="plasma")
    f1_ax3.set_title(title3, fontsize=20)
    f1_ax3.axis("off")

    f1_ax4.imshow(image4, cmap="plasma")
    f1_ax4.set_title(title4, fontsize=20)
    f1_ax4.axis("off")

    plt.show()


# Visualisation des images RGV et HED


def display_4_rgb(ima1, ima1_h, ima1_e, ima1_d):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(ima1)
    ax[0].set_title("Original image")

    ax[1].imshow(ima1_h)
    ax[1].set_title("Hematoxylin")

    ax[2].imshow(ima1_e)
    ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

    ax[3].imshow(ima1_d)
    ax[3].set_title("DAB")

    for a in ax.ravel():
        a.axis("off")

    fig.tight_layout()


#### claissif
def display_prediction_22(
    rgb,
    prediction,
    ground_true=None,
    final_pred=None,
    state_names="",
    mode_prediction="",
    epoch_number=None,
    display=False,
):
    """
    Visualise les prediction du model avec les classe - utilisé pendant le trianing
    """
    eosin = rgb_to_eosin(rgb)
    num_classes = prediction.shape[-1]
    nb_subplots = num_classes + 1
    if ground_true is not None:
        nb_subplots += 1
    if final_pred is not None:
        nb_subplots += 1
    f = plt.figure(figsize=(25, 10), constrained_layout=False)
    if epoch_number is not None:
        f.suptitle("Epoch " + str(epoch_number), fontsize=50, fontweight="bold")
    spec = gridspec.GridSpec(ncols=num_classes, nrows=2, figure=f)
    f_ax1 = f.add_subplot(spec[0, 0])
    a = f_ax1.imshow(rgb)
    f_ax1.set_title("rgb", fontsize=12)
    f_ax1.axis("off")

    f_ax2 = f.add_subplot(spec[0, 1])
    a = f_ax2.imshow(eosin)
    f_ax2.set_title("eosin", fontsize=15)
    f_ax2.axis("off")

    if ground_true is not None:
        f_ax3 = f.add_subplot(spec[0, 2])
        b = f_ax3.imshow(ground_true, cmap=CMAP, vmin=0, vmax=255)
        values_on_gt = np.unique(ground_true)
        colors = [b.cmap(b.norm(value)) for value in values_on_gt]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values_on_gt[i])])
            for i in range(1, len(values_on_gt))
        ]
        plt.legend(handles=patches, prop={"size": 13})
        f_ax3.set_title("Ground Truth", fontsize=15)
        f_ax3.axis("off")

    if final_pred is not None:
        f_ax4 = f.add_subplot(spec[0, 3])
        # divider = make_axes_locatable(f_ax4)
        # cax = divider.append_axes('right', size='5%', pad=0.1)
        a = f_ax4.imshow(final_pred, cmap=CMAP, vmin=0, vmax=255)
        values_on_final_pred = np.unique(final_pred)
        colors = [a.cmap(a.norm(value)) for value in values_on_final_pred]
        patches = [
            mpatches.Patch(
                color=colors[i], label=state_names[int(values_on_final_pred[i])]
            )
            for i in range(1, len(values_on_final_pred))
        ]
        plt.legend(handles=patches, prop={"size": 13})
        f_ax4.set_title("Prediction final", fontsize=15)
        f_ax4.axis("off")

    # if epoch_number is not None :
    #     f_ax4 = f.add_subplot(spec[0, 4])
    #     f_ax4.set_title("Epochhh \n"+str(epoch_number), fontsize=50)
    #     f_ax4.axis('off')

    for classe_i in range(num_classes):
        f_ax2 = f.add_subplot(spec[1, classe_i])
        divider = make_axes_locatable(f_ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        a = f_ax2.imshow(prediction[0, :, :, classe_i], vmin=0, vmax=1, cmap="viridis")
        f.colorbar(a, cax=cax, orientation="vertical")
        f_ax2.set_title(
            "proba : "
            + state_names[classe_i]
            + "\nmax:"
            + str(np.round(np.max(prediction[0, :, :, classe_i]), 5)),
            fontsize=15,
        )
        f_ax2.axis("off")

    # plt.text(0.6, 0.7, "eggs", size=50, rotation=0,
    #      ha="center", va="center",
    #      bbox=dict(boxstyle="round",
    #                ec=(1., 0.5, 0.5),
    #                fc=(1., 0.8, 0.8),
    #                ))
    # plt.text(0.55, 0.6, "spam", size=50, rotation=0,
    #         ha="right", va="top",
    #         bbox=dict(boxstyle="square",
    #                 ec=(1., 0.5, 0.5),
    #                 fc=(1., 0.8, 0.8),
    #                 ))
    if display:
        plt.show()

    return f


def display_prediction_23(
    rgb,
    prediction,
    ground_true=None,
    final_pred=None,
    mask_cell=None,
    compute_weiht=True,
    state_names="",
    title="",
    mode_prediction="summed_proba",
    cmap="viridis",
):
    """
    Visualise les prediction du model avec les classe
    """
    eosin = rgb_to_eosin(rgb)

    num_classes = prediction.shape[-1]
    nb_subplots = num_classes + 1
    if ground_true is not None:
        nb_subplots += 1
    if final_pred is not None:
        nb_subplots += 1

    f = plt.figure(figsize=(25, 8), constrained_layout=False)
    spec = gridspec.GridSpec(ncols=num_classes, nrows=2, figure=f)

    f_ax1 = f.add_subplot(spec[0, 0])
    a = f_ax1.imshow(rgb)
    f_ax1.set_title("rgb " + title, fontsize=12)
    f_ax1.axis("off")

    f_ax2 = f.add_subplot(spec[0, 1])
    a = f_ax2.imshow(eosin)
    f_ax2.set_title("eosin", fontsize=15)
    f_ax2.axis("off")

    if ground_true is not None:
        f_ax3 = f.add_subplot(spec[0, 2])
        b = f_ax3.imshow(ground_true, cmap=CMAP, vmin=0, vmax=255)
        values_on_gt = np.unique(ground_true)
        colors = [b.cmap(b.norm(value)) for value in values_on_gt]
        patches = [
            mpatches.Patch(color=colors[i], label=state_names[int(values_on_gt[i])])
            for i in range(1, len(values_on_gt))
        ]
        plt.legend(handles=patches, prop={"size": 13})
        f_ax3.set_title("Ground Truth", fontsize=15)
        f_ax3.axis("off")

    if final_pred is not None:
        f_ax4 = f.add_subplot(spec[0, 3])
        # divider = make_axes_locatable(f_ax4)
        # cax = divider.append_axes('right', size='5%', pad=0.1)
        a = f_ax4.imshow(final_pred, cmap=CMAP, vmin=0, vmax=255)
        values_on_final_pred = np.unique(final_pred)
        colors = [a.cmap(a.norm(value)) for value in values_on_final_pred]
        patches = [
            mpatches.Patch(
                color=colors[i], label=state_names[int(values_on_final_pred[i])]
            )
            for i in range(1, len(values_on_final_pred))
        ]
        plt.legend(handles=patches, prop={"size": 13})

        f_ax4.set_title("Prediction final", fontsize=15)
        f_ax4.axis("off")

    for classe_i in range(num_classes):
        f_ax2 = f.add_subplot(spec[1, classe_i])
        divider = make_axes_locatable(f_ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        a = f_ax2.imshow(prediction[0, :, :, classe_i], vmin=0, vmax=1, cmap=cmap)
        f.colorbar(a, cax=cax, orientation="vertical")
        size_cell = np.sum(mask_cell)
        if mode_prediction == "summed_proba":
            if classe_i == 0:
                # background
                if compute_weiht:
                    weight_background = (
                        np.sum(prediction[0, :, :, classe_i]) / size_cell
                    )
                    weight_state = np.sum(prediction[0, :, :, classe_i]) / size_cell
                else:
                    weight_state = np.max(prediction[0, :, :, classe_i])
                f_ax2.set_title(
                    "proba : "
                    + state_names[classe_i]
                    + "\nsummed:"
                    + str(np.round(weight_state, 5)),
                    fontsize=15,
                )
                f_ax2.axis("off")
            else:
                if compute_weiht:
                    # weight_state = np.sum(prediction[0,:,:,classe_i])/size_cell
                    weight_state = np.sum(prediction[0, :, :, classe_i]) / (
                        size_cell * (1 - weight_background + 0.0000001)
                    )
                else:
                    weight_state = np.max(prediction[0, :, :, classe_i])

                f_ax2.set_title(
                    "proba : "
                    + state_names[classe_i]
                    + "\nsummed:"
                    + str(np.round(weight_state, 5)),
                    fontsize=15,
                )
                f_ax2.axis("off")
        if mode_prediction == "max_proba":
            max_state = np.max(prediction[0, :, :, classe_i])
            f_ax2.set_title(
                "proba : " + state_names[classe_i] + "\nmax:" + str(max_state),
                fontsize=15,
            )
            f_ax2.axis("off")
    return f


def display_prediction(
    rgb, prediction, ground_true=None, final_pred=None, class_name=""
):
    """
    Visualise les prediction du model avec les classe
    """
    eosin = rgb_to_eosin(rgb)

    num_classes = prediction.shape[-1]
    nb_subplots = num_classes + 1
    if ground_true is not None:
        nb_subplots += 1
    if final_pred is not None:
        nb_subplots += 1

    f = plt.figure(figsize=(25, 8), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=num_classes, nrows=2, figure=f)

    f_ax1 = f.add_subplot(spec[0, 0])
    a = f_ax1.imshow(rgb)
    f_ax1.set_title("rgb", fontsize=12)
    f_ax1.axis("off")

    f_ax2 = f.add_subplot(spec[0, 1])
    a = f_ax2.imshow(eosin)
    f_ax2.set_title("eosin", fontsize=15)
    f_ax2.axis("off")

    if ground_true is not None:
        f_ax3 = f.add_subplot(spec[0, 2])
        divider = make_axes_locatable(f_ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        a = f_ax3.imshow(ground_true, cmap="viridis")
        f.colorbar(a, cax=cax, orientation="vertical")
        f_ax3.set_title("Ground Truth", fontsize=15)
        f_ax3.axis("off")

    if final_pred is not None:
        f_ax4 = f.add_subplot(spec[0, 3])
        divider = make_axes_locatable(f_ax4)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        a = f_ax4.imshow(final_pred, cmap="viridis")
        f.colorbar(a, cax=cax, orientation="vertical")
        f_ax4.set_title("Prediction final", fontsize=15)
        f_ax4.axis("off")

    for classe_i in range(num_classes):
        f_ax2 = f.add_subplot(spec[1, classe_i])
        divider = make_axes_locatable(f_ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        a = f_ax2.imshow(prediction[0, :, :, classe_i], cmap="viridis")
        f.colorbar(a, cax=cax, orientation="vertical")
        f_ax2.set_title("proba : " + class_name[classe_i], fontsize=15)
        f_ax2.axis("off")

    plt.show()



def display_img(
    np_img,
    text=None,
    font_path=HtmlGenerationConfig.font_path,
    size=48,
    color=(255, 0, 0),
    background=(255, 255, 255),
    border=(0, 0, 0),
    bg=False,
):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.

    Args:
      np_img: Image as a NumPy array.
      text: The text to add to the image.
      font_path: The path to the font to use.
      size: The font size
      color: The font color
      background: The background color
      border: The border color
      bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == "L":
        result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)

        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()


def display_cellpose_on_roi(roi, save_fig=False, figsize=(24, 12), display=False):
    roi.img_rgb_borders, roi.mask_cellpose_extended

    img_pil = np_to_pil(roi.img_rgb_borders)
    img_draw = ImageDraw.Draw(img_pil, "RGBA")
    # img_rgb = draw_roi_delimiter_CELLPOSE(img_rgb)
    img_draw, nb_cells = add_mask_nuclei_to_img_CELLPOSE(
        img_draw, roi.mask_cellpose_extended
    )

    f = plt.figure(figsize=figsize)
    plt.imshow(img_pil)

    if save_fig:
        path_figure = os.path.join(roi.path_classification, "0_Images_ROI")
        last_fig_number = get_last_fig_number(path_figure)
        name_fig = (
            str(last_fig_number + 1).zfill(3) + "_Cellpose_nuclei_segmentation.png"
        )
        mkdir_if_nexist(path_figure)
        path = os.path.join(path_figure, name_fig)
        id_img = 1
        while os.path.exists(path):
            path = os.path.join(path_figure, str(id_img).zfill(2) + "_" + name_fig)
            id_img += 1
        f.savefig(
            path, facecolor="white", dpi="figure", bbox_inches="tight", pad_inches=0.1
        )
    if display:
        plt.show()
    else:
        plt.close("All")
    return nb_cells

def plot_several_mask_1_rgb(
    rgb,
    liste_mask,
    liste_mask_title,
    path_save=None,
    display=False,
    figsize=(7, 3),
    square_crop=True,
):
    """
    Display function
    -------
    Display several masks on the same rgb image
    """
    img_pil = np_to_pil(rgb)
    fig, axes = plt.subplots(
        1, len(liste_mask), figsize=figsize, sharex=True, sharey=True
    )
    ax = axes.ravel()
    for i, mask in enumerate(liste_mask):
        img_pil_copy = img_pil.copy()
        img = ImageDraw.Draw(img_pil_copy, "RGBA")
        im = add_mask_to_img(img, mask)
        im = add_crop_line_on_img(img, mask) if square_crop else None
        ax[i].imshow(img_pil_copy)
        ax[i].set_title(liste_mask_title[i])
        ax[i].set_axis_off()
    if display:
        plt.show()
    if path_save is not None:
        fig.savefig(path_save, facecolor="white", dpi="figure")
    plt.close("All")


def plot_several_mask_several_rgb(
    list_rgb,
    liste_mask,
    liste_mask_title,
    path_save=None,
    display=False,
    figsize=(7, 3),
    square_crop=True,
):
    """
    Display function
    -------
    Display several masks on the same rgb image
    """
    if len(liste_mask) > 1:
        fig, axes = plt.subplots(
            1, len(liste_mask), figsize=figsize, sharex=True, sharey=True
        )
        ax = axes.ravel()
        for i, mask in enumerate(liste_mask):
            img_pil = np_to_pil(list_rgb[i])
            img = ImageDraw.Draw(img_pil, "RGBA")
            im = add_mask_to_img(img, mask)
            im = add_crop_line_on_img(img, mask) if square_crop else None
            ax[i].imshow(img_pil)
            ax[i].set_title(liste_mask_title[i])
            ax[i].set_axis_off()
        # plt.autoscale(tight=True)
        if display:
            plt.show()
        if path_save is not None:
            fig.savefig(
                path_save,
                facecolor="white",
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.1,
            )
        plt.close("All")
    else:
        img_pil = np_to_pil(list_rgb[0])
        img = ImageDraw.Draw(img_pil, "RGBA")
        im = add_mask_to_img(img, liste_mask[0])
        plt.imshow(img_pil)
        plt.axis("off")
        if display:
            plt.show()
        if path_save is not None:
            fig.savefig(
                path_save,
                facecolor="white",
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.1,
            )
        plt.close("All")

def display_roi_in_tissu(
    image1,
    image_background=None,
    title="Mask",
    dict_text=dict(),
    liste_center_of_mass=[],
    add_squares=False,
    add_numbers=False,
    add_center_of_mass=False,
    figsize=(8, 8),
    vmax=None,
    pathsave=None,
    cmap="Purples",
):
    fig1 = plt.figure(figsize=figsize, constrained_layout=True)
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    f1_ax1 = fig1.add_subplot(spec1[0, 0])

    # divider = make_axes_locatable(f1_ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    if vmax is not None:
        f1 = f1_ax1.imshow(image1, cmap=cmap, vmin=np.min(image1), vmax=vmax)
    else:
        f1 = f1_ax1.imshow(image1, cmap=cmap, vmin=np.min(image1), vmax=np.max(image1))

    # fig1.colorbar(f1, cax=cax, orientation='vertical')

    if image_background is not None:
        f1_ax1.imshow(image_background, alpha=0.5)

    if len(liste_center_of_mass) > 0:
        for coord in range(len(liste_center_of_mass)):
            x, y = (
                liste_center_of_mass["x_coord"][coord],
                liste_center_of_mass["y_coord"][coord],
            )
            if add_numbers:
                f1_ax1.text(y, x, str(coord), color="black", fontsize=12)
            if add_center_of_mass:
                f1_ax1.plot(y, x, "+", color="darkgreen", markersize=10)
            if add_squares:
                f1_ax1.add_patch(
                    patches.Rectangle(
                        (int(y - int(SIZE_CROP / 2)), int(x - int(SIZE_CROP / 2))),
                        SIZE_CROP,
                        SIZE_CROP,
                        edgecolor="blue",
                        facecolor="red",
                        fill=False,
                    )
                )

    if len(dict_text) > 0:
        text_kwargs = dict(fontsize=20, color=(0, 77 / 255, 128 / 255, 1))
        for key, item in dict_text.items():
            f1_ax1.text(
                item["y"], item["x"], item["s"], **text_kwargs
            )  # , bbox=dict(fill=False, edgecolor='white', linewidth = 3)

    f1_ax1.set_title(title, fontsize=20)

    if pathsave is not None:
        fig1.savefig(pathsave)

    else:
        plt.show()
