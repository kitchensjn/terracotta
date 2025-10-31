

def _up_and_down(
        parent_list,
        branch_above_list,
        time_bin_widths,
        ids_asc_time,
        roots,
        sample_ids,
        sample_locations_array,
        branch_lengths,
        precomputed_transitions,
        log_stationary
    ):
    """Calculates the log-likelihood of a tree

    Parameters
    ----------
    parent_list
    branch_above_list
    roots
    sample_ids
    sample_locations_array
    branch_lengths
    precomputed_transitions
    precomputed_log

    Returns
    -------
    tree_likelihood
    root_log_likes
    """
    
    num_nodes = len(branch_above_list[0])
    num_demes = len(sample_locations_array[0])

    node_pos = np.zeros((num_nodes, num_demes), dtype="float64")

    for i in ids_asc_time:
        if i in sample_ids:
            node_pos[i] = sample_locations_array[np.where(sample_ids==i)[0][0]]
        else:
            children_of_i = np.where(parent_list==i)[0]
            if len(children_of_i) > 0:
                incoming_messages = np.zeros((len(children_of_i), num_demes), dtype="float64")
                counter = 0
                for child in children_of_i:
                    bl = branch_above_list[:, child]
                    incoming_messages[counter] = node_pos[child]
                    counter += 1


            children_of_i = np.where(parent_list==i)[0]
            if len(children_of_i) > 0:
                incoming_log_messages = np.zeros((len(children_of_i), num_demes), dtype="float64")
                counter = 0
                for child in children_of_i:
                    incoming_log_messages[counter] = log_messages[child]
                    counter += 1
                summed_log_messages = np.sum(incoming_log_messages, axis=0)
                bl = branch_above_list[:, i]
                included_epochs = np.where(bl > 0)[0]
                if (len(included_epochs) > 0):
                    transition_prob = np.eye(num_demes)
                    for epoch in included_epochs:
                        bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                        transition_prob = np.dot(transition_prob, precomputed_transitions[epoch][bl_index])
                    transition_prob_log = np.log(transition_prob)
                    combined = summed_log_messages + transition_prob_log
                    c = np.max(combined)
                    log_sum_exp = c + np.log(np.sum(np.exp(combined - c), axis=1))
                    outgoing_log_message = log_sum_exp + np.log(time_bin_widths[i])
                else:
                    outgoing_log_message = summed_log_messages
                log_messages[i] = outgoing_log_message



def _parallel_process_trees_for_node_locs(
        parents,
        branch_above,
        time_bin_widths,
        ids_asc_time,
        roots,
        sample_ids,
        sample_locations_array,
        branch_lengths,
        precomputed_transitions,
        log_stationary
    ):
    """Calculates and combines the log-likelihood of every tree

    Parameters
    ----------
    children
    branch_above
    roots
    sample_ids
    sample_locations_array
    branch_lengths
    precomputed_transitions
    precomputed_log

    Returns
    -------
    mr_log_like
    log_likelihoods
    """

    positions = []
    for i in range(len(branch_above)):
        _up_and_down(
            parent_list=parents[i],
            branch_above_list=branch_above[i],
            time_bin_widths=time_bin_widths[i],
            ids_asc_time=ids_asc_time[i],
            roots=roots[i],
            sample_ids=sample_ids,
            sample_locations_array=sample_locations_array,
            branch_lengths=branch_lengths,
            precomputed_transitions=precomputed_transitions,
            log_stationary=log_stationary
        )
    

def _calc_node_locations(migration_rates, world_map, parents, branch_above, time_bin_widths, ids_asc_time, roots, branch_lengths, output_file=None):
    """Calculates the composite log-likelihood of the specified migration rates across trees
    
    Loops through all trees and calculates the log-likelihood for each, before summing together.

    Parameters
    ----------
    migration_rates
    world_map
    parents
    branch_above
    roots
    branch_lengths

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """

    transition_matrices = world_map.build_transition_matrices(migration_rates=migration_rates)

    # https://people.duke.edu/~ccc14/sta-663-2016/homework/Homework02_Solutions.html
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrices[-1].T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    stationary = w/np.sum(w)
    stationary[stationary<1e-99] = 1e-99
    log_stationary = np.log(stationary)
    
    precomputed_transitions = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrices=transition_matrices
    )

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    _parallel_process_trees_for_node_locs(
        parents=parents,
        branch_above=branch_above,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        roots=roots,
        sample_ids=sample_ids,
        sample_locations_array=sample_locations_array,
        branch_lengths=branch_lengths,
        precomputed_transitions=precomputed_transitions,
        log_stationary=log_stationary
    )

def reconstruct_node_locations(
        migration_rates,
        demes_path,
        samples_path,
        trees_dir_path,
        time_bins=None,
        asymmetric=False,
        output_file=None
    ):

    if output_file is not None:
        sys.stdout = open(output_file, "w")
    
    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, samples, asymmetric)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts)
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())

    pl, bal, tbw, iat, r, ubl = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)  # needed to use numba

    _calc_node_locations(
        migration_rates=migration_rates,
        world_map=world_map,
        parents=pl,
        branch_above=bal,
        time_bin_widths=tbw,
        ids_asc_time=iat,
        roots=r,
        branch_lengths=ubl,
        output_file=output_file
    )


