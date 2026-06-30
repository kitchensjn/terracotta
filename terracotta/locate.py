import numpy as np
from .main import deconstruct_tree
from scipy.linalg import expm


def _calc_branch_message(
        current_pos,
        branch_above,
        transition_matrices,
        direction="backward"
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    direction : string


    Returns
    -------
    current_pos : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """


    included_epochs = np.where(branch_above > 0)[0]
    for epoch in included_epochs:
        trans_prob = expm(transition_matrices[epoch]*branch_above[epoch])
        if direction == "backward":
            current_pos = np.matmul(trans_prob, current_pos)
        else:
            current_pos = np.matmul(current_pos, trans_prob)
    return current_pos


def _calc_all_messages(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    ):

    num_demes = len(sample_locations_array[0])
    messages = np.ones((len(parents)*2, num_demes), dtype="float64")
    for id in ids_asc_time:
        # Within this loop, repeatedly implement Equation 2 from the manuscript

        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            current_pos = np.prod(messages[np.where(parents==id)[0]], axis=0)

        current_pos = current_pos / np.sum(current_pos)
        
        messages[id] = _calc_branch_message(
            current_pos,
            branch_above[:,id],
            transition_matrices,
            direction="backward"
        )
    
    for id in ids_asc_time[::-1]:
        parent_of = np.where(parents==id)[0]
        for c in range(len(parent_of)):
            alt_children = np.delete(parent_of, c)
            current_pos = np.prod(np.concatenate((messages[[id+len(parents)]], messages[alt_children])), axis=0)

            current_pos = current_pos / np.sum(current_pos)
            
            messages[parent_of[c]+len(parents)] = _calc_branch_message(
                current_pos,
                branch_above[:,id],
                transition_matrices,
                direction="forward"
            )

    return messages

def ancs(tree, u):
    """Find all of the ancestors above a node for a tree

    Taken directly from https://github.com/tskit-dev/tskit/issues/2706

    Parameters
    ----------
    tree : tskit.Tree
        Tree to be traversed
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

def track_lineage_over_time(
        sample,
        times,
        tree,
        world_map,
        parameters
    ):
    """
    Parameters
    ----------

    Returns
    -------
    positions
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

    if "alpha" in world_map.parameters:
        alpha = parameters[world_map.parameters.index("alpha")]
    else:
        alpha = 1

    sample_locations_array, sample_ids = world_map.build_sample_locations_array()
    parents, branch_above, time_bin_widths, ids_asc_time = deconstruct_tree(tree, world_map.epochs)
    
    transition_matrices = world_map.build_transition_matrices(parameters=parameters)
    pop_sizes = world_map.suitabilities ** alpha

    messages = _calc_all_messages(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    )

    positions = np.zeros((len(pc_combos), len(world_map.demes)))
    for element, node_combo in enumerate(pc_combos):
        if node_combo[0] == node_combo[1]:
            if node_combo[0] in sample_ids:
                node_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                combined = np.prod(np.concatenate((messages[[node_combo[0]+len(parents)]], messages[np.where(parents==node_combo[0])[0]])), axis=0)
                node_pos = combined / sum(combined)
        else:
            if node_combo[0] in sample_ids:
                child_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_child_messages = messages[np.where(parents==node_combo[0])[0]]
                if len(incoming_child_messages) > 0:
                    combined = np.prod(incoming_child_messages, axis=0)
                    child_pos = combined / sum(combined)
                else:
                    child_pos = np.ones((1,len(world_map.demes)))[0]
            if node_combo[1] in sample_ids:
                parent_pos = sample_locations_array[np.where(sample_ids==node_combo[1])[0][0]]
            else:
                backward_messages = np.where(parents==node_combo[1])[0]
                backward_messages = backward_messages[backward_messages != node_combo[0]]
                incoming_parent_messages = np.concatenate((messages[[node_combo[1]+len(parents)]], messages[backward_messages]))
                if len(incoming_parent_messages) > 0:
                    combined = np.prod(incoming_parent_messages, axis=0)
                    parent_pos = combined / sum(combined)
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

            outgoing_child_message = _calc_branch_message(
                child_pos,
                bl_child,
                transition_matrices,
                direction="backward"
            )
            
            outgoing_parent_message = _calc_branch_message(
                parent_pos,
                bl_parent,
                transition_matrices,
                direction="forward"
            )

            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
            node_pos = node_pos / np.sum(node_pos)
        positions[element] = node_pos

    return positions