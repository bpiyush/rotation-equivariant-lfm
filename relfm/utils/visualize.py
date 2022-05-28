"""Helpers for visualization"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# define predominanat colors
COLORS = {
    "pink": (242, 116, 223),
    "cyan": (46, 242, 203),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def show_single_image(image: np.ndarray, figsize: tuple = (8, 8), title: str = None, titlesize=18, cmap: str = None, ticks=False, save=False, save_path=None):
    """Show a single image."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(image, Image.Image):
        image = np.asarray(image)

    ax.set_title(title, fontsize=titlesize)
    ax.imshow(image, cmap=cmap)
    
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if save:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()


def show_grid_of_images(
        images: np.ndarray, n_cols: int = 4, figsize: tuple = (8, 8),
        cmap=None, subtitles=None, title=None, subtitlesize=18,
    ):
    """Show a grid of images."""
    n_cols = min(n_cols, len(images))

    copy_of_images = images.copy()
    for i, image in enumerate(copy_of_images):
        if isinstance(image, Image.Image):
            image = np.asarray(image)
            images[i] = image
    
    if subtitles is None:
        subtitles = [None] * len(images)

    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if len(images[i].shape) == 2:
                cmap="gray"
            ax.imshow(images[i], cmap=cmap)
            ax.set_title(subtitles[i], fontsize=subtitlesize)
            ax.axis('off')
    fig.set_tight_layout(True)
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


def draw_kps_on_image(image: np.ndarray, kps: np.ndarray, color=COLORS["red"], radius=3, thickness=-1, return_as="numpy"):
    """
    Draw keypoints on image.

    Args:
        image: Image to draw keypoints on.
        kps: Keypoints to draw. Note these should be in (x, y) format.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image)

    for kp in kps:
        image = cv2.circle(
            image, (int(kp[0]), int(kp[1])), radius=radius, color=color, thickness=thickness)
    
    if return_as == "PIL":
        return Image.fromarray(image)

    return image


def get_concat_h(im1, im2):
    """Concatenate two images horizontally"""
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    """Concatenate two images vertically"""
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def show_images_with_keypoints(images: list, kps: list, radius=15, color=(0, 220, 220), figsize=(10, 8), return_images=False):
    assert len(images) == len(kps)

    # generate
    images_with_kps = []
    for i in range(len(images)):
        img_with_kps = draw_kps_on_image(images[i], kps[i], radius=radius, color=color, return_as="PIL")
        images_with_kps.append(img_with_kps)
    
    # show
    show_grid_of_images(images_with_kps, n_cols=len(images), figsize=figsize)
    
    if return_images:
        return images_with_kps


def set_latex_fonts(usetex=True, fontsize=14, show_sample=False, **kwargs):
    try:
        plt.rcParams.update({
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": fontsize,
            **kwargs,
        })
        if show_sample:
            plt.figure()
            plt.title("Sample $y = x^2$")
            plt.plot(np.arange(0, 10), np.arange(0, 10)**2, "--o")
            plt.grid()
            plt.show()
    except:
        print("Failed to setup LaTeX fonts. Proceeding without.")
        pass


def get_colors(num_colors, palette="jet"):
    cmap = plt.get_cmap(palette)
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors