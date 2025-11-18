import numpy as np


def probabilistic_select_rows(arrays, probabilities, seed: int = None):
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
    rng = np.random.default_rng(seed=seed)
    choice_indices = rng.choice(4, size=M, p=probabilities)

    # 3. Use NumPy indexing to select the rows.
    # We provide the chosen array index (from choice_indices) for each row index.
    # np.arange(M) provides the row indices [0, 1, ..., M-1].
    # For each i in arange(M), this selects the slice:
    #   stacked_arrays[choice_indices[i], i, :]
    # The result is a 2D array of shape (M, N).
    result = stacked_arrays[choice_indices, np.arange(M)]

    return result


def _greedy_group_disjoint(even_stabilizer_indices):
    """
    Groups even stabilizers into the minimum number of disjoint groups.
    This implements a greedy graph coloring algorithm.

    Args:
        even_stabilizer_indices (np.array): 1D array of stabilizer integers.

    Returns:
        list[list[int]]: A list where each inner list is a group of
                         bitwise-disjoint stabilizer indices.
    """
    # Convert to a standard list to make dynamic grouping easier
    nodes = list(even_stabilizer_indices)
    groups_indices = []  # This will store our list of groups

    for node in nodes:
        placed = False
        # Try to place the node in an existing group
        for group in groups_indices:
            is_disjoint_from_all = True
            for member in group:
                if (node & member) != 0:
                    is_disjoint_from_all = False
                    break  # This group won't work

            if is_disjoint_from_all:
                group.append(node)
                placed = True
                break  # Node has been placed

        # If no existing group worked, create a new one
        if not placed:
            groups_indices.append([node])

    return groups_indices


def constructive_disjoint_complete_graph_stabilizer_grouping(even_stabilizer_indices, n):
    """
    Split all non-trivial stabilizer elements (N-1; not including trivial I^n; N = 2^n)
    of "complete graph state" into groups with the following constraints.
    
    1. Each stabilizer element is assigned to a 'single' group
    2. Every pair of Pauli string in the same group, P and Q, must have non-overlapped support
        i.e., weight(P) + weight(Q) = weight(P.Q) 
        
    In this specific case of K_n graph, all odd product of generators have full support on the qubits,
    thus they each form each own group (N/2) group. We then left with finding the grouping method for 
    even product of generators. For K_n graph, any even product of generators will only have I or Y term.

    Using a binary string "b" to represent a stabilizer element, where "b_j" = 1 if the generator from qubit-j
    is selected, we can have a complete ordering for the stabilizer; e.g., b = 1001 = Y1 Y4
    Specifically, the string b is representative of the Pauli string where 1 is Y and 0 is I.

    We provide two strategies for when the number of qubits (n) is even, and when (n) is odd.
    The following strategy is used to group the even product stabilizer elements:
    1. Even n: Uses the optimal (k, n-k) complement pairing --- this always guarantee full support (optimal).
    2. Odd n:  Uses the k_a + k_b <= n rule to create "super-groups"
               of compatible k-values, then runs the greedy algorithm
               on those smaller, simpler sub-problems.
    """
    # validation we assume that the input even_stabilizer_indices must be unique and have length N/2 - 1
    N = 2 ** n
    if len(even_stabilizer_indices) != (N//2 - 1) or len(set(even_stabilizer_indices)) != len(even_stabilizer_indices):
        raise ValueError("Expected full even products of complete graph stabilizer elements. Partial grouping not supported.")

    # 1. Partition stabilizers by their bit count (k)
    k_map = {}
    for idx in even_stabilizer_indices:
        k = np.bitwise_count(idx)
        if k not in k_map:
            k_map[k] = []
        k_map[k].append(idx)

    groups = []

    if n % 2 == 0:
        all_ones = (1 << n) - 1
        canonical_pairs = {tuple(sorted((a, all_ones ^ a))) for a in even_stabilizer_indices}

        special_pair = (0, all_ones)
        canonical_pairs.remove(special_pair)
        canonical_pairs.add((all_ones,))

        groups = sorted([list(pair) for pair in canonical_pairs])
    else:
        # --- Strategy for odd n: k_a + k_b <= n "super-groups" ---

        # 1. Find which k-values can be grouped together.
        super_groups_k = []  # list of lists of k-values
        remaining_k = sorted(k_map.keys())

        while remaining_k:
            k = remaining_k.pop(0)
            current_super_group = [k]
            temp_remaining = []

            for other_k in remaining_k:
                can_add = True
                for member_k in current_super_group:
                    if member_k + other_k > n:
                        can_add = False
                        break

                if can_add:
                    current_super_group.append(other_k)
                else:
                    temp_remaining.append(other_k)

            super_groups_k.append(current_super_group)
            remaining_k = temp_remaining

        # 2. Run the simple greedy algorithm on each "super-group"
        for k_list in super_groups_k:
            nodes_to_group = []
            for k in k_list:
                nodes_to_group.extend(k_map[k])

            # Run greedy algorithm *only* on this separable subset
            if nodes_to_group:
                sub_groups = _greedy_group_disjoint(np.array(nodes_to_group))
                groups.extend(sub_groups)

    return groups
