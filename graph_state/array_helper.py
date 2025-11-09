import numpy as np

def probabilistic_select_rows(arrays, probabilities):
    """
    Selects entire rows from a list of 2D NumPy arrays based on probabilities.

    For each row index, one of the four input arrays is chosen based on the
    probabilities, and the entire row from that chosen array is used for the
    output.

    Args:
        arrays (list of np.ndarray): A list of 4 2D NumPy arrays
                                    of the same shape (M, N).
        probabilities (list of float): A list of 4 probabilities (p1, p2, p3, p4)
                                       that sum to 1.

    Returns:
        np.ndarray: A new 2D NumPy array of shape (M, N).
    """
    # --- Input Validation ---
    if not np.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    if len(arrays) != 4 or len(probabilities) != 4:
        raise ValueError("Exactly 4 arrays and 4 probabilities are required.")

    # Check if all arrays are 2D and have the same shape
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError("All input arrays must be 2-dimensional.")
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        raise ValueError("All arrays must have the same shape.")

    M, N = shape

    # --- Core Logic ---
    # 1. Stack the arrays along a new first axis.
    # If input shape is (M, N), stacked shape becomes (4, M, N).
    stacked_arrays = np.stack(arrays)

    # 2. For each of the M rows, randomly choose an index from [0, 1, 2, 3]
    # based on the provided probabilities. This gives a 1D array of M choices.
    choice_indices = np.random.choice(4, size=M, p=probabilities)

    # 3. Use advanced NumPy indexing to select the rows.
    # We provide the chosen array index (from choice_indices) for each row index.
    # np.arange(M) provides the row indices [0, 1, ..., M-1].
    # For each i in arange(M), this selects the slice:
    #   stacked_arrays[choice_indices[i], i, :]
    # The result is a 2D array of shape (M, N).
    result = stacked_arrays[choice_indices, np.arange(M)]

    return result