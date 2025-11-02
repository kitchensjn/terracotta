
@njit()
def _calc_tree_log_likelihood(
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

    log_messages = np.zeros((num_nodes, num_demes), dtype="float64")

    for counter in range(len(sample_ids)):
        i = sample_ids[counter]
        bl = branch_above_list[:, i]
        included_epochs = np.where(bl > 0)[0]
        if (len(included_epochs) > 0):
            transition_prob = np.eye(num_demes)
            for epoch in included_epochs:
                bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                transition_prob = np.dot(transition_prob, precomputed_transitions[epoch][bl_index])
            log_messages[i] = np.log(np.dot(transition_prob, sample_locations_array[counter]))
    
    for i in ids_asc_time:
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
    root_log_likes = np.zeros((len(roots)), dtype="float64")
    counter = 0
    for r in roots:
        if r not in sample_ids:
            root_pos = log_messages[r] + log_stationary
            c = np.max(root_pos)
            root_log_likes[counter] = c + np.log(np.sum(np.exp(root_pos - c), axis=0))
        counter += 1
    tree_likelihood = sum(root_log_likes)
    return tree_likelihood, root_log_likes

@njit(parallel=True)
def _parallel_process_trees(
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

    log_likelihoods = np.zeros(len(branch_above), dtype="float64")
    for i in prange(len(branch_above)):
        log_likelihoods[i] = _calc_tree_log_likelihood(
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
        )[0]
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods


def calc_log_migration_rate_log_likelihood(log_migration_rates, world_map, parents, branch_above, time_bin_widths, ids_asc_time, roots, branch_lengths, output_file=None):
    migration_rates = np.exp(log_migration_rates)
    return calc_migration_rate_log_likelihood(migration_rates, world_map, parents, branch_above, time_bin_widths, ids_asc_time, roots, branch_lengths, output_file=None)

def calc_migration_rate_log_likelihood(migration_rates, world_map, parents, branch_above, time_bin_widths, ids_asc_time, roots, branch_lengths, output_file=None):
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

    stationary = np.full(len(world_map.demes), 1/len(world_map.demes))

    log_stationary = np.log(stationary)
    
    precomputed_transitions = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrices=transition_matrices
    )

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    like, like_list = _parallel_process_trees(
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
    if output_file is not None:
        print(migration_rates, abs(like), file=f)
    else:
        print(migration_rates, abs(like), flush=True)
    return abs(like)

def _deconstruct_tree(tree, epochs, time_bins=None):
    num_nodes = len(tree.postorder())
    parents = np.full(num_nodes, -1, dtype="int64")
    branch_above = np.zeros((len(epochs), num_nodes), dtype="float64")
    time_bin_widths = np.full(num_nodes, -1, dtype="int64")
    ids_asc_time = np.full(num_nodes, -1, dtype="int64")
    for i,node in enumerate(tree.nodes(order="timeasc")):
        node_time = tree.time(node)
        parent = tree.parent(node)
        if parent != -1:
            parent_time = tree.time(parent)
            starting_epoch = np.digitize(node_time, epochs)-1
            ending_epoch = np.digitize(parent_time, epochs)-1
            if starting_epoch == ending_epoch:
                branch_above[starting_epoch, node] = parent_time - node_time
            else:
                branch_above[starting_epoch, node] = epochs[starting_epoch+1] - node_time
                for e in range(starting_epoch+1, ending_epoch):
                    branch_above[e, node] = epochs[e+1] - epochs[e]
                branch_above[ending_epoch, node] = parent_time - epochs[ending_epoch]
        ids_asc_time[i] = node
        parents[node] = parent
        if time_bins is not None:
            i = next(j for j, e in enumerate(time_bins) if e >= node_time)
            width = max(1, time_bins[i] - time_bins[i - 1])
            time_bin_widths[node] = width
        else:
            time_bin_widths[node] = 1
    return parents, branch_above, time_bin_widths, ids_asc_time

def _deconstruct_trees(trees, epochs, time_bins=None):
    """

    Note: It would be great if pl and bal were numpy.ndarray, but that would force
    the trees to have the same number of nodes, which is unrealistic.
    """
    
    pl = []
    bal = []
    tbw = []
    iat = []
    roots = []
    all_branch_lengths = [[] for e in epochs]
    for tree in trees:
        parents, branch_above, time_bin_widths, ids_asc_time = _deconstruct_tree(tree, epochs, time_bins=time_bins)
        pl.append(parents)
        bal.append(branch_above)
        tbw.append(time_bin_widths)
        iat.append(ids_asc_time)
        roots.append(np.where(parents==-1)[0])
        for e in range(len(epochs)):
            all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))
    return pl, bal, tbw, iat, roots, unique_branch_lengths




def run_for_single(
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

    likelihood = calc_migration_rate_log_likelihood(
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

    return likelihood


def run(
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
        tree = tskit.load(ts).simplify()
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())

    pl, bal, tbw, iat, r, ubl = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)  # needed to use numba

    bounds = [(-10, 0) for rate in world_map.existing_connection_types]

    res = shgo(
        calc_log_migration_rate_log_likelihood,
        bounds=bounds,
        n=100,
        iters=max(5, len(bounds)),
        sampling_method="sobol",
        args=(world_map, pl, bal, tbw, iat, r, ubl, output_file)
    )

    if output_file is not None:
        sys.stdout.close()

    return res

def ancs(tree, u):
    """Find all of the ancestors above a node for a tree

    Taken directly from https://github.com/tskit-dev/tskit/issues/2706

    Parameters
    ----------
    tree : tskit.Tree
    u : int
        The ID for the node of interest

    Returns
    -------
    An iterator over the ancestors of u in this tree
    """

    u = tree.parent(u)
    while u != -1:
         yield u
         u = tree.parent(u)

def _get_messages(
        parents,
        branch_above,
        branch_lengths,
        backwards,
        sample_locations_array,
        sample_ids,
        ids_asc_time,
        roots,
        stationary
    ):


    num_demes = len(sample_locations_array[0])

    messages = {}
    for child in ids_asc_time:
        parent = int(parents[child])
        bl = branch_above[:, child]
        included_epochs = np.where(bl > 0)[0]
        if (len(included_epochs) > 0):
            transition_prob = np.eye(num_demes)
            for epoch in included_epochs:
                bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                transition_prob = np.dot(transition_prob, backwards[epoch][bl_index])
            if child in sample_ids:
                loc_vec = sample_locations_array[np.where(sample_ids==child)[0][0]]
            else:
                incoming_keys_child = [key for key in messages.keys() if key[1] == child]
                incoming_messages = [messages[income] for income in incoming_keys_child]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                if len(incoming_messages) > 0:
                    loc_vec = incoming_messages[0]
                else:
                    loc_vec = np.ones(num_demes)
            messages[(child, parent)] = np.dot(transition_prob, loc_vec)
        elif parent != -1:
            messages[(child, parent)] = loc_vec

    #for root in roots:
    #    messages[(-1, root)] = stationary

    for child in ids_asc_time[-1::-1]:#range(len(parents)-1, -1, -1):
        if child not in sample_ids:
            parent = int(parents[child])
            bl = branch_above[:, child]
            included_epochs = np.where(bl > 0)[0]
            if (len(included_epochs) > 0):
                transition_prob = np.eye(num_demes)
                for epoch in included_epochs:
                    bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                    transition_prob = np.dot(transition_prob, backwards[epoch][bl_index])
                incoming_keys_parent = [key for key in messages.keys() if key[1] == parent]
                incoming_messages = [messages[income] for income in incoming_keys_parent if income[0] != child]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                if len(incoming_messages) > 0:
                    loc_vec = incoming_messages[0]
                else:
                    loc_vec = np.ones(num_demes)
                messages[(parent, child)] = np.dot(loc_vec, transition_prob)
    return messages
        

def track_lineage_over_time(
        sample,
        times,
        tree,
        world_map,
        migration_rates
    ):
    """Estimates the location probability distribution for an ancestral lineage

    Note: this function is very slow and could benefit from revisiting.

    Parameters
    ----------
        sample : int
            sample ID 
        times : list
            List of times (in generations before past)
        tree : tskit.Tree
            Input tree for a specified location
        world_map : WorldMap
            Map including the demes and sample locations
        migration_rates : np.array
            Rates of different connection types

    Returns
    -------
    positions : dict
        All of the positions for 
    """

    ancestors = [sample] + list(ancs(tree=tree, u=sample))

    node_times = []
    for a in ancestors:
        node_times.append(int(tree.time(a)))

    pc_combos = []
    for t in times:
        for i,v in enumerate(node_times):
            if v > t:
                child = ancestors[i-1]
                parent = ancestors[i]
                break
            elif v == t:
                child = ancestors[i]
                parent = ancestors[i]
                break
        pc_combos.append((child, parent))

    parents, branch_above, time_bin_widths, ids_asc_time = _deconstruct_tree(tree, world_map.epochs)
    
    all_branch_lengths = [[] for e in world_map.epochs]
    for e in range(len(world_map.epochs)):
        all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(world_map.epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))

    roots = np.where(parents==-1)[0]

    transition_matrices = world_map.build_transition_matrices(migration_rates=migration_rates)

    # https://people.duke.edu/~ccc14/sta-663-2016/homework/Homework02_Solutions.html
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrices[-1].T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    stationary = w/np.sum(w)

    precomputed_transitions = precalculate_transitions(
        branch_lengths=unique_branch_lengths,
        transition_matrices=transition_matrices
    )

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    messages = _get_messages(
        parents,
        branch_above,
        unique_branch_lengths,
        precomputed_transitions,
        sample_locations_array,
        sample_ids,
        ids_asc_time,
        roots,
        stationary
    )

    positions = {}
    for element,node_combo in enumerate(pc_combos):
        if node_combo[0] == node_combo[1]:
            if node_combo[0] in sample_ids:
                node_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_keys = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages = [messages[income] for income in incoming_keys]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                node_pos = incoming_messages[0]
        else:
            if node_combo[0] in sample_ids:
                child_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_keys_child = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages_child = [messages[income] for income in incoming_keys_child if income[0] != node_combo[1]]
                if len(incoming_keys_child) > 0:
                    for i in range(1,len(incoming_messages_child)):
                        incoming_messages_child[0] = np.multiply(incoming_messages_child[0], incoming_messages_child[i])
                        incoming_messages_child[0] = incoming_messages_child[0] / np.sum(incoming_messages_child[0])
                    child_pos = incoming_messages_child[0]
                else:
                    child_pos = np.ones((1,len(world_map.demes)))[0]
            if node_combo[1] in sample_ids:
                parent_pos = sample_locations_array[np.where(sample_ids==node_combo[1])[0][0]]
            else:
                incoming_keys_parent = [key for key in messages.keys() if key[1] == node_combo[1]]
                incoming_messages_parent = []
                for income in incoming_keys_parent:
                    if income[0] != node_combo[0]:
                        incoming_messages_parent.append(messages[income])
                if len(incoming_messages_parent) > 0:
                    for i in range(1,len(incoming_messages_parent)):
                        incoming_messages_parent[0] = np.multiply(incoming_messages_parent[0], incoming_messages_parent[i])
                        incoming_messages_parent[0] = incoming_messages_parent[0] / np.sum(incoming_messages_parent[0])
                    parent_pos = incoming_messages_parent[0]
                else:
                    parent_pos = np.ones((1,len(world_map.demes)))[0]

            branch_length_to_child = int(times[element] - tree.time(node_combo[0]))
            bl_child = branch_above[:, node_combo[0]].copy()
            bl_parent = bl_child.copy()
            whats_left = branch_length_to_child
            for e in range(len(world_map.epochs)):
                current = bl_parent[e].copy()
                if current >= whats_left:
                    bl_parent[e] -= whats_left
                else:
                    bl_parent[e] = 0
                whats_left -= current
            bl_child -= bl_parent

            included_epochs = np.where(bl_child > 0)[0]
            transition_prob_backwards = np.eye(len(world_map.demes))
            for e in included_epochs:
                transition_prob_backwards = np.dot(transition_prob_backwards, np.linalg.matrix_power(linalg.expm(transition_matrices[e]), bl_child[e]))

            included_epochs = np.where(bl_parent > 0)[0]
            transition_prob_forwards = np.eye(len(world_map.demes))
            for e in included_epochs:
                transition_prob_forwards = np.dot(transition_prob_forwards, np.linalg.matrix_power(linalg.expm(transition_matrices[e]), bl_parent[e]))
            
            outgoing_child_message = np.dot(transition_prob_backwards, child_pos)
            outgoing_parent_message = np.dot(parent_pos, transition_prob_forwards)
            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
            node_pos = node_pos / np.sum(node_pos)
        positions[times[element]] = node_pos
    return positions


def locate_nodes(
        tree,
        world_map,
        migration_rates
    ):

    parents, branch_above, time_bin_widths, ids_asc_time = _deconstruct_tree(tree, world_map.epochs)
    
    all_branch_lengths = [[] for e in world_map.epochs]
    for e in range(len(world_map.epochs)):
        all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(world_map.epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))

    roots = np.where(parents==-1)[0]

    transition_matrices = world_map.build_transition_matrices(migration_rates=migration_rates)

    # https://people.duke.edu/~ccc14/sta-663-2016/homework/Homework02_Solutions.html
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrices[-1].T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    stationary = w/np.sum(w)
    
    stationary = np.full(len(world_map.demes), math.pi)

    precomputed_transitions = precalculate_transitions(
        branch_lengths=unique_branch_lengths,
        transition_matrices=transition_matrices
    )

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    messages = _get_messages(
        parents,
        branch_above,
        unique_branch_lengths,
        precomputed_transitions,
        sample_locations_array,
        sample_ids,
        ids_asc_time,
        roots,
        stationary
    )

    locs = np.zeros((len(parents), len(world_map.demes)), dtype="float64")
    for node in tree.nodes():
        if node in sample_ids:
            locs[node] = sample_locations_array[np.where(sample_ids==node)[0][0]]
        else:
            incoming_keys = [key for key in messages.keys() if key[1] == node]
            incoming_messages = [messages[income] for income in incoming_keys]
            if len(incoming_messages) > 0:
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                locs[node] = incoming_messages[0]
            else:
                raise RuntimeError(f"No incoming messages for Node {node}. Check the connectedness of your tree to ensure all non-sample nodes are connected.")

    return locs



def locate_nodes_revell(
        ts,
        world_map,
        migration_rates
    ):


    transition_matrices = world_map.build_transition_matrices(migration_rates=migration_rates)

    L = np.zeros((ts.num_nodes, len(world_map.demes)))

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    for i,id in enumerate(sample_ids):
        L[id] = sample_locations_array[i]

    tables = ts.tables
    edges = tables.edges
    parents = pd.unique(edges.parent)

    for parent in parents:
        parent_time = ts.node(parent).time
        children = edges.child[np.where(edges.parent==parent)[0]]
        PP = np.zeros((len(children), len(world_map.demes)))
        for j,child in enumerate(children):
            child_time = ts.node(child).time
            edge_length = parent_time - child_time
            P = linalg.expm(transition_matrices[0]*edge_length)
            PP[j] = np.dot(P, L[child])
        L[parent] = np.prod(PP, axis=0)
    
    prob = np.log(np.sum((1/len(world_map.demes))*L[parent]))

    for parent in parents[-1::-1]:
        parent_time = ts.node(parent).time
        children = edges.child[np.where(edges.parent==parent)[0]]
        for j,child in enumerate(children):
            child_time = ts.node(child).time
            edge_length = parent_time - child_time
            P = linalg.expm(transition_matrices[0]*edge_length)
            pp = L[parent] / (np.dot(P, L[child]))
            L[child] = np.multiply(np.dot(pp, P), L[child])
    
    L = L/L.sum(axis=1, keepdims=True)
    
    return L