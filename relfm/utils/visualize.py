"""Helpers for visualization"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2


def show_single_image(image: np.ndarray, figsize: tuple = (8, 8), title: str = None, cmap: str = None, ticks=False):
    """Show a single image."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title)
    ax.imshow(image, cmap=cmap)
    
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def show_grid_of_images(
        images: np.ndarray, n_cols: int = 4, figsize: tuple = (8, 8),
        cmap=None, subtitles=None, title=None,
    ):
    """Show a grid of images."""
    
    if subtitles is None:
        subtitles = [None] * len(images)

    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if len(images[i].shape) == 2:
                cmap="gray"
            ax.imshow(images[i], cmap=cmap)
            ax.set_title(subtitles[i])
            ax.axis('off')
    plt.suptitle(title, y=0.8)
    plt.show()


def show_keypoint_matches(
        img1, kp1, img2, kp2, matches,
        K=10, figsize=(10, 5), drawMatches_args=dict(matchesThickness=3, singlePointColor=(0, 0, 0)),
        choose_matches="random",
    ):
    """Displays matches found in the pair of images"""
    if choose_matches == "random":
        selected_matches = np.random.choice(matches, K)
    elif choose_matches == "all":
        K = len(matches)
        selected_matches = matches
    elif choose_matches == "topk":
        selected_matches = matches[:K]
    else:
        raise ValueError(f"Unknown value for choose_matches: {choose_matches}")

    # color each match with a different color
    cmap = matplotlib.cm.get_cmap('gist_rainbow', K)
    colors = [[int(x*255) for x in cmap(i)[:3]] for i in np.arange(0,K)]
    drawMatches_args.update({"matchColor": -1, "singlePointColor": (100, 100, 100)})
    
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, outImg=None, **drawMatches_args)
    show_single_image(
        img3,
        figsize=figsize,
        title=f"[{choose_matches.upper()}] Selected K = {K} matches between the pair of images.",
    )
    return img3