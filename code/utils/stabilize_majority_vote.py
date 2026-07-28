import numpy as np
from scipy.signal import convolve2d

def stabilize_majority_vote(coords, labels):

    # ---- Map labels to 0/1 ----
    # 'normal'/'cancer' are the trained classifier's fixed output vocabulary
    # (non-malignant/malignant), not renamed here since these strings must
    # match what the classifier actually predicts.
    label_to_id = {'normal': 0, 'cancer': 1}
    id_to_label = {v: k for k, v in label_to_id.items()}
    labels = np.vectorize(label_to_id.get)(labels)

    # ---- Build sorted coordinate grid ----
    xs = sorted(np.unique(coords[:, 0]))
    ys = sorted(np.unique(coords[:, 1]))

    W = len(xs)
    H = len(ys)

    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}

    # ---- Initialize grids ----
    grid = np.zeros((H, W), dtype=np.uint8)     # malignant / non-malignant
    mask = np.zeros((H, W), dtype=np.uint8)     # whether a patch exists

    for (x, y), v in zip(coords, labels):
        j = y_to_j[y]
        i = x_to_i[x]
        grid[j, i] = v
        mask[j, i] = 1

    # ---- Convolution kernel (3×3) ----
    kernel = np.ones((3, 3), dtype=np.uint8)

    # ---- Iterate to stability ----
    for _ in range(1000):
        # Sum malignant counts in neighbors
        malignant_sum = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)

        # Count how many neighbors exist
        neighbor_count = convolve2d(mask, kernel, mode='same', boundary='fill', fillvalue=0)

        # Majority threshold = half of existing patches
        majority = (malignant_sum >= (neighbor_count / 2)).astype(np.uint8)

        # New grid only applies changes where mask=1
        new_grid = grid.copy()
        new_grid[mask == 1] = majority[mask == 1]

        if np.array_equal(new_grid, grid):
            break

        grid = new_grid
    else:
        print("Warning: max iterations reached")

    # ---- Convert back to original label list ----
    out_labels = []
    for x, y in coords:
        out_labels.append(id_to_label[grid[y_to_j[y], x_to_i[x]].item()])

    return np.array(out_labels)
