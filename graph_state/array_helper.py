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
    groups_indices = [] # This will store our list of groups
    
    for node in nodes:
        placed = False
        # Try to place the node in an existing group
        for group in groups_indices:
            is_disjoint_from_all = True
            for member in group:
                if (node & member) != 0:
                    is_disjoint_from_all = False
                    break # This group won't work
            
            if is_disjoint_from_all:
                group.append(node)
                placed = True
                break # Node has been placed
        
        # If no existing group worked, create a new one
        if not placed:
            groups_indices.append([node])
            
    return groups_indices

def constructive_grouping(even_stabilizer_indices, n):
    """
    Implements a much smarter grouping strategy based on the
    combinatorial structure of the stabilizers, as suggested by the user.

    1. Even n: Uses the optimal (k, n-k) complement pairing.
    2. Odd n:  Uses the k_a + k_b <= n rule to create "super-groups"
               of compatible k-values, then runs the greedy algorithm
               on those smaller, simpler sub-problems.
    """

    # 1. Partition stabilizers by their bit count (k)
    k_map = {}
    for idx in even_stabilizer_indices:
        k = np.bitwise_count(idx)
        if k not in k_map:
            k_map[k] = []
        k_map[k].append(idx)

    groups = []

    if (n % 2 == 0):
        # --- Strategy for even n: (k, n-k) pairing ---
        processed_k = set()
        all_ones = (1 << n) - 1

        for k in sorted(k_map.keys()):
            if k in processed_k:
                continue

            complement_k = n - k

            if k == complement_k:
                # Special case: k = n/2. Pair them with each other.
                paired_nodes = set()
                for a in k_map[k]:
                    if a in paired_nodes:
                        continue
                    b = a ^ all_ones
                    # b must also be in k_map[k] and not yet paired
                    if b in k_map[k] and b not in paired_nodes:
                        groups.append([a, b])
                        paired_nodes.add(a)
                        paired_nodes.add(b)
                    else:
                        # This should not happen if n/2 is even
                        groups.append([a]) 
                processed_k.add(k)

            elif complement_k in k_map:
                # Standard case: k and n-k are distinct.
                # All complements are in the other k-group.
                for a in k_map[k]:
                    b = a ^ all_ones
                    groups.append([a, b])
                processed_k.add(k)
                processed_k.add(complement_k)

            else:
                # Unpaired k-group (e.g., k=n)
                for a in k_map[k]:
                    groups.append([a])
                processed_k.add(k)

    else:
        # --- Strategy for odd n: k_a + k_b <= n "super-groups" ---

        # 1. Find which k-values can be grouped together.
        super_groups_k = [] # list of lists of k-values
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
