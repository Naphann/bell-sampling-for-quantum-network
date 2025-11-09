import math
from functools import partial  # For fixing arguments to worker functions

import igraph as ig
import multiprocess as mp
import numpy as np
import stim
from sympy import fwht

from graph_state.array_helper import probabilistic_select_rows


class GraphState:

    def __init__(self, n: int, graph_type="complete", edges: list[tuple[int, int]] = None):
        """unless graph type is specified as 'manual' the edges parameters are ignored
        all the graphs are guaranteed to be single connected component with the exception of manual graph_type
        """
        if graph_type not in [
            "manual",
            "complete",
            "random-tree",
            "star",
            "path",
            "ring",
        ]:
            raise ValueError(f"unknown graph_type (given {graph_type})")

        if graph_type == "manual" and edges is None:
            raise ValueError("edge list must be given for manual graph type")

        # attribute template
        self.n = n
        self.type = graph_type
        self.graph = ig.Graph()
        self.adj_mat = np.array([[]])
        self.stab_generators: list[str] = []
        self.int_stab_generators = np.array([])

        # attribute assignment
        if graph_type == "complete":
            self.graph = ig.Graph.Full(n)
        elif graph_type == "random-tree":
            self.graph = ig.Graph.Tree_Game(n)
        elif graph_type == "star":
            self.graph = ig.Graph.Star(n)
        elif graph_type == "path":
            self.graph = ig.Graph.Ring(n, circular=False)
        elif graph_type == "ring":
            self.graph = ig.Graph.Ring(n, circular=True)
        else:  # manual
            # TODO: add some checks
            self.graph = ig.Graph(n=n, edges=edges)

        self.adj_mat = self.graph.get_adjacency()
        self._populate_generators()
        self._powers_of_2_for_selection = 2 ** np.arange(self.n)

    def _populate_generators(self):
        """enumerate all stabilizer generators as 
            1. strings and store them to .stab_generators.
            2. int and store them to .int_stab_generators (1 for Z, 2 for X, and 0 for I).
        """
        for u, vs in enumerate(self.adj_mat):
            paulis = ["Z" if v == 1 else "_" for v in vs]
            paulis[u] = "X"
            self.stab_generators.append("".join(paulis))

        # populate the integer version for easy sampling
        self.int_stab_generators = np.array(self.graph.get_adjacency())
        for u in range(self.n):
            self.int_stab_generators[u][u] = 2

    def generate_all_int_staiblizers(self):
        """
        Generates all non-identity stabilizer elements from the base generators.

        The return format for each stabilizer is a Boolean numpy array of
        shape (3 * n_qubits), where the first n_qubits elements denote
        Pauli X locations, the next n_qubits for Z locations, and the
        final n_qubits for Y locations.

        Assumes Pauli encoding: I=0, Z=1, X=2, Y=3.
        """
        for i in range(1, 1 << self.n):
            # The j-th generator is selected if the j-th bit of 'i' is set.
            selection_mask = (i & self._powers_of_2_for_selection) != 0
            selected_generators = self.int_stab_generators[selection_mask]

            # Compute the Pauli product for this combination.
            # This is a column-wise bitwise XOR sum of the selected generators.
            # The result, 'pauli_product_vector', is a 1D array of length self.n_qubits,
            # where each element is the integer representation of the resulting
            # Pauli operator on that qubit (e.g., X, Y, Z, or I).
            pauli_product_vector = np.bitwise_xor.reduce(selected_generators, axis=0)

            # Based on the integer encoding (e.g., X=2, Z=1, Y=3):
            # Create boolean arrays indicating the presence of X, Z, or Y Paulis
            # at each qubit position. Each is of shape (self.n_qubits,).
            is_X = pauli_product_vector == 2
            is_Z = pauli_product_vector == 1
            is_Y = pauli_product_vector == 3

            # Concatenate these boolean arrays to form the specified extended format:
            # [X_q0, X_q1,..., X_qn-1, Z_q0,..., Z_qn-1, Y_q0,..., Y_qn-1]
            yield np.concatenate((is_X, is_Z, is_Y))

    def get_stabilizers_extended_format(self):
        """
        The return format is a Boolean numpy array of shape (3n)
        where the first n elements denoting the marked Pauli X, and then Z, and then Y.
        """
        for i in range(1, 2**self.n):
            si = np.array([c == "1" for c in f"{i:0{self.n}b}"[::-1]])
            yield np.array(
                [
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 2,
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 1,
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 3,
                ]
            ).reshape(self.n * 3)

    def sample_int_stabilizers(self, shots: int):
        """
        xx = int_paulis == 2
        zz = int_paulis == 1
        yy = int_paulis == 3
        """
        stabilizers = np.random.rand(shots, self.n) > 0.5
        return np.array(
            [
                [
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 2,
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 1,
                    (np.bitwise_xor.reduce(self.int_stab_generators[si][:])) == 3,
                ]
                for si in stabilizers
            ]
        ).reshape(shots, self.n * 3)

    def get_graph_state_circuit(self, offset: int) -> stim.Circuit:
        """return stim.Circuit representing the graph state initialization with qubit index (offset, offset + graph.n)"""
        circuit = stim.Circuit(f"""H {' '.join(map(str, range(offset, offset + self.n)))}""")
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if self.adj_mat[u][v]:
                    circuit.append("CZ", [offset + u, offset + v])

        return circuit

    def get_noise_circuit(self, fidelity: float, error_model: str, offset: int) -> stim.Circuit:
        if error_model not in [
            "no-error",
            "single-qubit-dephasing",
            "bimodal",
            "fully-dephased",
        ]:
            raise ValueError(f"unknown error_model (given {error_model})")
        if fidelity < 0 or fidelity > 1:
            raise ValueError(f"Fidelity must be between 0 and 1 (given {fidelity})")
        circuit = stim.Circuit()
        
        if fidelity == 1 or error_model == "no-error":
            return circuit
        
        if error_model == "single-qubit-dephasing":
            p = 1 - (fidelity ** (1.0 / self.n))
            return stim.Circuit(f'Z_ERROR({p}) {" ".join(map(str, range(offset, offset + self.n)))}')
        
        if error_model == 'fully-dephased':
            return stim.Circuit(f'Z_ERROR({0.5}) {" ".join(map(str, range(offset, offset + self.n)))}')
        
        if error_model == "bimodal":
            p = 1 - fidelity
            return stim.Circuit(f"Z_ERROR({p}) {offset}")
        
        raise RuntimeError("reaching end of noise circuit generation without getting valid noise circuit.")
    
    def get_bell_sampling_circuit(self) -> stim.Circuit:
        sampling_circuit = stim.Circuit(
            f"""
        CX {' '.join([f'{i} {self.n + i}' for i in range(self.n)])}
        H {' '.join(map(str, range(self.n)))}
        CX {' '.join([f'{i} {2 * self.n + i}' for i in range(self.n)])}
        CX {' '.join([f'{self.n + i} {2 * self.n + i}' for i in range(self.n)])}
        X {' '.join(map(str,range(2 * self.n, 3 * self.n)))}
        MZ {' '.join(map(str, range(3 * self.n)))}
        """
        )
        return sampling_circuit
    
    def get_partial_tomo_measurement_circuit(self) -> stim.Circuit:
        stim_pauli_prods = []
        for st in self.stab_generators:
            words = [f'{p}{i}' for i, p in enumerate(st) if p != '_']
            words = '*'.join(words)
            stim_pauli_prods.append(words)

        return stim.Circuit(f'MPP {' '.join(stim_pauli_prods)}')


def get_true_diagonals(num_qubits: int, fidelity: float, error_model: str):
    """Returns the true diagonal vector in the graph-state basis.
    
    Recall that the vector is independent to the underlying graph and only depends on the noise model.
    """
    n = num_qubits
    N = 2 ** n
    us = np.array([0.0] * N, dtype=np.float64)
    if error_model == "no-error":
        us[0] = 1
        return us
    
    if error_model == "depolarizing":
        us[0] = fidelity
        us[1:] = (1 - fidelity) / (N - 1)
        return us
    
    if error_model == "single-qubit-dephasing":
        p = 1 - (fidelity ** (1.0 / n))
        us = np.array(
            [
                p ** i.bit_count() * (1 - p) ** (n - i.bit_count())
                for i in range(N)
            ]
        )
        return us
    
    if error_model == "bimodal":
        us[0] = fidelity
        us[1] = 1 - fidelity
        return us

    raise ValueError(f"unknown error_model (given {error_model})")


def expectation_value_of_observables(int_paulis: np.ndarray, measurement_results: np.ndarray):
    # tables for eigenvalues
    #      | Phi+ | Phi- | Psi+ | Psi- |
    #  XX  |   1  |  -1  |   1  |  -1  |  check Z
    #  YY  |  -1  |   1  |   1  |  -1  |  xor 00
    #  ZZ  |   1  |   1  |  -1  |  -1  |  check X
    fn = lambda x: np.bitwise_xor.reduce(x & int_paulis, axis=1) * -2 + 1
    return sum(fn(measurement_results)) / len(measurement_results)


def expectation_value_of_observables_bitpacked(
    int_paulis: np.ndarray, measurement_results: np.ndarray
):
    gn = lambda x: np.bitwise_count(np.bitwise_xor.reduce(x & int_paulis, axis=1)) % 2
    n = len(measurement_results)
    return max((n - 2.0 * np.sum(gn(measurement_results))) / n, 0)


def bell_sampling(g: GraphState, error_model: str, fidelity: float, shots: int):
    """steps to perform Bell samping.
        1. create the circuit of graph state.
        2. add noise according to the given noise model and fidelity
        3. generate samples and return.
    """
    # TODO: write the output format of the samples
    if error_model not in [
            "no-error",
            "depolarizing",
            "single-qubit-dephasing",
            "bimodal",
        ]:
        raise ValueError(f"unknown error_model (given {error_model})")
    if fidelity < 0 or fidelity > 1:
        raise ValueError(f"Fidelity must be between 0 and 1 (given {fidelity})")
    
    if error_model != "depolarizing":
        circuit = g.get_graph_state_circuit(0) + g.get_graph_state_circuit(g.n)
        circuit += g.get_noise_circuit(fidelity, error_model, 0) + g.get_noise_circuit(fidelity, error_model, g.n)
        circuit += g.get_bell_sampling_circuit()
        # return circuit
        return circuit.compile_sampler().sample_bit_packed(shots)
    
    """error model must be depolarizing, we need to build 4 circuits:
    1. no error/no error
    2. no error/fully dephased
    3. fully dephased/no error
    4. fully dephased/fully dephased
    and sample this based on the given fidelity
    """
    base_circ = g.get_graph_state_circuit(0) + g.get_graph_state_circuit(g.n)
    bell_circ = g.get_bell_sampling_circuit()

    circ_1 = base_circ + bell_circ
    circ_2 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', 0) + bell_circ
    circ_3 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', g.n) + bell_circ
    circ_4 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', 0) + g.get_noise_circuit(fidelity, 'fully-dephased', g.n) + bell_circ

    samples_1 = circ_1.compile_sampler().sample_bit_packed(shots)
    samples_2 = circ_2.compile_sampler().sample_bit_packed(shots)
    samples_3 = circ_3.compile_sampler().sample_bit_packed(shots)
    samples_4 = circ_4.compile_sampler().sample_bit_packed(shots)

    N = 2 ** g.n
    p = fidelity - (1 - fidelity) / (N - 1)

    samples = probabilistic_select_rows([samples_1, samples_2, samples_3, samples_4], [p**2, p * (1 - p), p * (1 - p), (1 - p)**2])
    return samples    

def expectation_value_of_observables_from_bell_bitpacked(
    int_paulis: np.ndarray,
    measurement_results: np.ndarray,
) -> float:
    """
    Performs expectation values calculation of a given observable defined by Pauli string
    into custom binary format (but bitpacked into integers),
    where the format is constructed by expanding length n Pauli string into length 3n.
    The 3n bits are separated into 3 blocks of length n each.
    The 1s in the first block indicates whether or not the Pauli string has X on that position.
    Similarly for 2nd and 3rd block for Z and Y.

    For example, _X_YZ would translate to
    01000 00001 00010 -> bitpack into 01000000 0100010_ = 128 66
    (I could get the endian wrong)

    The expected measurement results are to be bitpacked and each entry has length 3n,
    where each block of n represents XX, ZZ, and YY parities (recall BSM).
    The YY is added by xoring the results of first two blocks.

    The calculation goes as follows.
    1. For each measurement result, perform bitwise and (&) with the Pauli string.
       (this tells the sign contribution from each bit)
    2. We see if the total is an odd or even parity (total sign of + or -)
    3. We then sum all of them and divided by the total number of measurements
    """
    gn = lambda x: np.bitwise_count(np.bitwise_xor.reduce(x & int_paulis, axis=1)) % 2
    n = len(measurement_results)
    return (n - 2.0 * np.sum(gn(measurement_results))) / n

def _worker_calculate_exp_value_packed(packed_s_item, meas_samples_fixed):
    """
    Worker function for multiprocessing. Calls the original expectation value function.
    'packed_s_item' is one row from the Packed_S_matrix.
    'meas_samples_fixed' is the constant meas_samples array.
    """
    return expectation_value_of_observables_from_bell_bitpacked(packed_s_item, meas_samples_fixed)

def expectation_value_of_observables_from_bell_bitpacked_parallelized(g: GraphState, bell_samples):
    stabilizers_unpacked_list = list(g.generate_all_int_staiblizers())
    S_matrix = np.array(stabilizers_unpacked_list)
    Packed_S_matrix = np.packbits(S_matrix, axis=1, bitorder='little')
    task_function = partial(_worker_calculate_exp_value_packed, meas_samples_fixed=bell_samples)
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores for other tasks
    packed_s_tasks = [row for row in Packed_S_matrix]
    with mp.Pool(processes=num_processes) as pool:
        exps_list_results = pool.map(task_function, packed_s_tasks)
        exps = np.array(exps_list_results)
    return exps

def fidelity_estimation_via_random_sampling_bitpacked(g: GraphState, num_stabilizers: int, bell_samples):
    # bitpacked in little endian, the same way as stim does
    fn = lambda x: np.sqrt(np.maximum(expectation_value_of_observables_bitpacked(x, bell_samples), 0))
    stabilizers = np.packbits(g.sample_int_stabilizers(num_stabilizers), axis=1, bitorder='little')
    sum_sqrt_cp = np.sum([fn(p) for p in stabilizers])
    return sum_sqrt_cp / len(stabilizers)

def fidelity_estimation_via_random_sampling_parallelized(g: GraphState, num_obs: int, bell_samples):
    ps = np.packbits(g.sample_int_stabilizers(num_obs), axis=1, bitorder='little')
    task_function = partial(_worker_calculate_exp_value_packed, meas_samples_fixed=bell_samples)
    num_processes = max(1, mp.cpu_count() - 2) # Leave some cores for other tasks
    packed_s_tasks = [row for row in ps]
    with mp.Pool(processes=num_processes) as pool:
        exps_list_results = pool.map(task_function, packed_s_tasks)
        exps = np.array(exps_list_results)
    return np.mean(np.sqrt(np.maximum(0, exps)))

def _post_process_partial_tomo_generator_samples_to_all_stabilizers(g: GraphState, samples):
    num_shots, num_chunks = samples.shape
    N = 2 ** g.n - 1 # we have 2^n - 1 non-trivial stabilizer

    num_shots_to_use = (num_shots // N) * N
    sgr_trimmed = samples[:num_shots_to_use]
    group_size = num_shots_to_use // N

    # 1. Generate all N stabilizer masks at once.
    # Each integer `i` from 1 to n represents a unique product of stabilizer
    # generators. We treat `i` as a bitmask and unpack it into a byte-wise
    # mask that aligns with the byte-packed measurement data.
    i_vals = np.arange(1, N + 1, dtype=np.uint64)[:, np.newaxis]
    exponents = np.arange(num_chunks, dtype=np.uint64)
    divisors = 256**exponents
    masks = ((i_vals // divisors) % 256).astype(sgr_trimmed.dtype)
    # print(masks)

    # 2. Reshape the measurement data into N distinct groups.
    # Shape changes from (num_shots_to_use, num_chunks) to (n, group_size, num_chunks).
    sgr_grouped = sgr_trimmed.reshape(N, group_size, num_chunks)

    # 3. Apply the i-th mask to the i-th group using broadcasting.
    # This `bitwise_and` operation effectively selects the measurement outcomes
    # of the generators that are part of the i-th stabilizer product.
    masks_reshaped = masks[:, np.newaxis, :]
    and_results = np.bitwise_and(sgr_grouped, masks_reshaped)

    # 4. Reduce the results for each shot.
    # The `bitwise_xor.reduce` sums the selected generator outcomes (modulo 2),
    # which gives the measurement outcome for the corresponding stabilizer product.
    xor_results = np.bitwise_xor.reduce(and_results, axis=2)

    # 5. Calculate the parity of the outcome for each shot.
    # The parity of the number of set bits in the result corresponds to the
    # eigenvalue: 0 for +1 (even) and 1 for -1 (odd).
    bit_counter = np.frompyfunc(int.bit_count, 1, 1)
    parities = bit_counter(xor_results).astype(np.int8) % 2
    parities = parities * -2 + 1

    # 6. Calculate the probability of a +1 eigenvalue for each group.
    # This is (1 - average_parity), since average_parity is the probability of -1.
    prob_plus_one = np.mean(parities, axis=1)

    return prob_plus_one

def partial_tomo(g: GraphState, error_model: str, fidelity: float, shots: int):
    """steps to perform partial tomo (to get expectation values over all obsevables defined from stabilizer elements).
        1. create the circuit of graph state.
        2. add noise according to the given noise model and fidelity
        3. add stabilizer generator measurement parts (we get n measurements each run)
        4. transform these n measurements to expectation values over all observables defined from stabilizer elements.
        5. return results.
    """
    # TODO: write the output format of the samples
    if error_model not in [
            "no-error",
            "depolarizing",
            "single-qubit-dephasing",
            "bimodal",
        ]:
        raise ValueError(f"unknown error_model (given {error_model})")
    if fidelity < 0 or fidelity > 1:
        raise ValueError(f"Fidelity must be between 0 and 1 (given {fidelity})")

    if error_model != "depolarizing":
        circuit = g.get_graph_state_circuit(0)
        circuit += g.get_noise_circuit(fidelity, error_model, 0)
        circuit += g.get_partial_tomo_measurement_circuit()

        samples = circuit.compile_sampler().sample_bit_packed(shots)
        return _post_process_partial_tomo_generator_samples_to_all_stabilizers(g, samples)
    
    """error model must be depolarizing, we need to build 4 circuits:
    1. no error/no error
    2. no error/fully dephased
    3. fully dephased/no error
    4. fully dephased/fully dephased
    and sample this based on the given fidelity
    """
    base_circ = g.get_graph_state_circuit(0)
    meas_circ = g.get_partial_tomo_measurement_circuit()

    circ_1 = base_circ + meas_circ
    circ_2 = base_circ + g.get_noise_circuit(fidelity, 'fully-dephased', 0) + meas_circ

    samples_1 = circ_1.compile_sampler().sample_bit_packed(shots)
    samples_2 = circ_2.compile_sampler().sample_bit_packed(shots)

    N = 2 ** g.n
    p = fidelity - (1 - fidelity) / (N - 1)

    mask = np.random.rand(shots) < p
    mask_reshaped = mask[:, np.newaxis]
    samples = np.where(mask_reshaped, samples_1, samples_2)

    return _post_process_partial_tomo_generator_samples_to_all_stabilizers(g, samples)

def get_diagonals_from_all_stabilizer_observables(g: GraphState, expvals):
    N_fwhm = 1 << g.n # 2**n_qubits, size of the FWHT vector
        
    # Create the input vector for FWHT of size N_fwhm
    fwht_input = np.zeros(N_fwhm, dtype=float)

    # Get the integer indices corresponding to the stabilizers in 'exps'
    # These indices are assumed to be 1, 2, ..., N_fwhm-1 if exps covers all non-identity Paulis
    stabilizer_integer_indices = np.arange(1, 2 ** g.n)
    # sqrt_exps_safe = np.sqrt(np.maximum(0, expvals))

    # Populate fwht_input:
    # fwht_input[0] remains 0 (for identity Pauli, if not included in exps)
    # fwht_input[s] = sqrt(expectation_value_of_stabilizer_s)
    fwht_input[stabilizer_integer_indices] = expvals
    fwht_input[0] = 1.0
    
    # Calculate the Fast Walsh-Hadamard Transform
    # The result 'transformed_coeffs[i]' = sum_s (fwht_input[s] * (-1)**<i,s>)
    # where <i,s> is the bitwise dot product (popcount(i&s) % 2)
    transformed_coeffs = np.array(fwht(fwht_input), dtype=float)
    
    # Calculate the final diagonal values
    # diagonals = (1.0 + transformed_coeffs) / N_fwhm
    diagonals = transformed_coeffs / N_fwhm
    return diagonals