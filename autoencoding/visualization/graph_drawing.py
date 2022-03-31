import matplotlib.pyplot as plt
import numpy as np


def codebook_embedding_graph(
    codebook, A, images, embedding_dimensions, node_scale=0.02
):
    """
    codebook: [S, F]
    A: [S, S]
    images: [S, W, H, C]
    embedding_dimensions: [2]
    """
    x_mi, y_mi = codebook.min(dim=0)[0].detach().cpu().numpy()[embedding_dimensions]
    x_ma, y_ma = codebook.max(dim=0)[0].detach().cpu().numpy()[embedding_dimensions]

    ax = plt.axes([x_mi, y_mi, x_ma - x_mi, y_ma - y_mi])

    args = np.argwhere(A > 0)

    lines = []
    for (i1, i2) in args:
        lines.extend(
            [
                (
                    codebook[i1, embedding_dimensions[0]],
                    codebook[i2, embedding_dimensions[0]],
                ),
                (
                    codebook[i1, embedding_dimensions[1]],
                    codebook[i2, embedding_dimensions[1]],
                ),
                "#D3D3D3",
            ]
        )

    ax.plot(*lines)

    ax.set_xlim([x_mi, x_ma])
    ax.set_ylim([y_mi, y_ma])
    ax.set_axis_off()

    for i in range(codebook.shape[0]):
        xa = codebook[i, embedding_dimensions[0]].detach().cpu().numpy()
        ya = codebook[i, embedding_dimensions[1]].detach().cpu().numpy()

        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * node_scale
        icon_center = icon_size / 2.0

        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(images[i], cmap="gray")
        a.set_yticks([])
        a.set_xticks([])
