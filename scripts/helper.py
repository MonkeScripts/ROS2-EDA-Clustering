from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_centroids(data:np.ndarray, labels: np.ndarray):
    centroids = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = data[labels == label]
        centroids[label] = np.mean(cluster_points, axis=0)
    return centroids


@dataclass
class VisualizationConfig:
    """data class for all plotting configs"""

    figsize: tuple[int, int] = (12, 8)
    view_angles: Optional[dict[str, float]] = field(
        default_factory=lambda: {"azim": -66, "elev": 12}
    )
    plot_kwds: Optional[dict[str, float]] = field(
        default_factory=lambda: {
            "alpha": 0.80,
            "s": 80,
        }
    )
    title_fontsize: int = 16
    show_axes: bool = True
    axes_labels: Optional[dict[str, str]] = field(
        default_factory=lambda: {
            "x": "x-axis",
            "y": "y-axis",
            "z": "z-axis",
        }
    )
    centroid_color: str = "red"
    centroid_plot_kwds: Optional[dict[str, float]] = field(
        default_factory=lambda: {
            "s": 200,
        }
    )
    show_centroid_values: bool = False


def plot2D(
    data: np.ndarray,
    labels: np.ndarray,
    title: str,
    colors: Optional[np.ndarray],
    config: VisualizationConfig,
):
    """
    Plots 2D data by taking in the data, title, colors and plot configs
    Args:
    data: data to be visualized
    labels from clustering results
    title: title of the plot
    colors: optional colours
    config: visualization configs
    """
    if data.ndim != 2 and data.shape[1] != 2:
        print("data is not the correct shape, should be (*, 2)")
        return
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot()
    fig.add_axes(ax)
    ax.set_title(title, fontsize=config.title_fontsize)
    ax.set_xlabel(config.axes_labels["x"])
    ax.set_ylabel(config.axes_labels["y"])
    ax.scatter(data[:, 0], data[:, 1], c=colors, **config.plot_kwds)
    centroids = get_centroids(data=data, labels=labels)
    centroid_coords = np.array(list(centroids.values()))
    ax.scatter(
        centroid_coords[:, 0],
        centroid_coords[:, 1],
        c=config.centroid_color,
        marker="x",
        **config.centroid_plot_kwds,
    )
    if config.show_centroid_values:
        for label, centroid in centroids.items():
            ax.annotate(
                f"C{label}: ({centroid[0]:.2f}, {centroid[1]:.2f})",
                (centroid[0], centroid[1]),
                xytext=(10, 10),  # offset text by 10 points
                textcoords="offset points",
                fontsize=9,
                color="red",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
            )
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(config.show_axes)
    frame.axes.get_yaxis().set_visible(config.show_axes)
    return frame


def plot3D(
    data: np.ndarray,
    labels: np.ndarray,
    title: str,
    colors: Optional[np.ndarray],
    config: VisualizationConfig,
):
    """
    Plots 3D data by taking in the data, title, colors and plot configs
    Args:
    data: data to be visualized
    labels:np.ndarray,
    title: title of the plot
    colors: optional colours
    config: visualization configs
    """
    if data.ndim != 2 and data.shape[1] != 3:
        print("data is not the correct shape, should be (*, 2)")
        return
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.set_title(title, fontsize=config.title_fontsize)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, **config.plot_kwds)
    # change view
    ax.view_init(**config.view_angles)
    ax.set_xlabel(config.axes_labels["x"])
    ax.set_ylabel(config.axes_labels["y"])
    ax.set_zlabel(config.axes_labels["z"])
    centroids = get_centroids(data=data, labels=labels)
    centroid_coords = np.array(list(centroids.values()))
    ax.scatter(
        centroid_coords[:, 0],
        centroid_coords[:, 1],
        centroid_coords[:, 2],
        c=config.centroid_color,
        marker="x",
        **config.centroid_plot_kwds,
    )
    if config.show_centroid_values:
        z_min, z_max = data[:, 2].min(), data[:, 2].max()
        z_offset = (z_max - z_min) * 0.1  # 10% of z range
        for label, centroid in centroids.items():
            x, y, z = centroid
            annotation = f"C{label}: ({x:.2f}, {y:.2f}, {z:.2f})"
            ax.text3D(
                x,  # Slight offset in x
                y,  # Slight offset in y
                z,  # Slight offset in z
                annotation,
                fontsize=9,
                color="red",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=1.0),
            )
    # add bottom text
    # _ = ax.text2D(0.8, 0.05, s="n_samples=2000", transform=ax.transAxes)
    # https://snyk.io/advisor/python/matplotlib/functions/matplotlib.pyplot.gca
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(config.show_axes)
    frame.axes.get_yaxis().set_visible(config.show_axes)
    frame.axes.get_zaxis().set_visible(config.show_axes)
    return frame


def plot_clusters(data: np.ndarray, labels: np.ndarray, plot_title: str, config: VisualizationConfig):
    """
    Visualises the clustering result
    Takes in the data, clustering algorithm used. Outputs the visualised plot
    Args:
        data: data to be clustered
        algorithm: name of clustering algorithm
        args, kwds: arguments for specific algorithm
    """
    if data.ndim != 2:
        print("data is not in the correct dimension to be plotted")
    if data.shape[1] not in [2, 3]:
        print("data is not the correct shape, shld be (*, 2) or (*, 3)")
    palette = sns.color_palette(
        palette="pastel", n_colors=np.unique(labels).max() + 1
    )
    plot_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    if data.shape[1] == 2:
        return plot2D(
            data=data,
            labels=labels,
            title=plot_title,
            colors=plot_colors,
            config=config,
        )
    if data.shape[1] == 3:
        return plot3D(
            data=data,
            labels=labels,
            title=plot_title,
            colors=plot_colors,
            config=config,
        )
