import numpy as np
from scipy import linalg
from scipy.special import logsumexp

def _propagate_up(nodes, arg, children, parents, log_messages, transition_matrix, roots):           
    for i,node in enumerate(nodes):
        node_children = np.unique(children[np.where(parents == node.id)[0]])
        if len(node_children) > 0:
            incoming_log_messages = []
            for child in node_children:
                incoming_log_messages.append(log_messages[f"{child}_{node.id}"])
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            node_parents = np.unique(parents[np.where(children == node.id)[0]])
            if len(node_parents) == 0:
                log_messages[f"{node.id}_"] = summed_log_messages
            elif len(node_parents) == 1:
                bl = arg.node(node_parents[0]).time - node.time
                if bl > 0:
                    outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*bl)).T + summed_log_messages, axis=1)])
                    log_messages[f"{node.id}_{node_parents[0]}"] = outgoing_log_message
                else:
                    log_messages[f"{node.id}_{node_parents[0]}"] = summed_log_messages
            else:
                num_positions = len(summed_log_messages[0])
                pos_root_likelihoods = []
                for pos in range(num_positions):
                    set_pos = np.zeros((1,num_positions))
                    set_pos[0,pos] = 1
                    for p in node_parents:
                        bl = arg.node(p).time - node.time
                        if bl > 0:
                            #outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*bl)).T + set_pos, axis=1)]) #### THIS IS A BUG
                            outgoing_log_message = np.log(np.matmul(set_pos, linalg.expm(transition_matrix*bl)))
                            log_messages[f"{node.id}_{p}"] = outgoing_log_message
                        else:
                            raise RuntimeError(f"Edge for node {node.id} to {p} has length {bl}, which is not allowed.")
                    pos_root_likelihoods.append(_propagate_up(nodes[i+1:], arg, children, parents, log_messages, transition_matrix, roots) + summed_log_messages[0][pos])
                arg_likelihood = logsumexp(pos_root_likelihoods)
                return arg_likelihood
    root_log_likes = [logsumexp(log_messages[f"{r}_"]) for r in roots]
    arg_likelihood = sum(root_log_likes)
    return arg_likelihood

def _calc_generalized_log_likelihood(arg, sample_location_vectors, transition_matrix):
    """Calculates the log_likelihood of the tree using Felsenstein's Pruning Algorithm.

    NOTE: Assumes that samples are always tips on the tree.
    NOTE: Ignores samples that are completely detached from the tree(s).
    NOTE: Parent of sample cannot have the same time as sample.
    NOTE: Currently, assumes that you know the sample location with certainty... which I think is fair.

    Parameters
    ----------
    tree : tskit.Tree
        This is a tree taken from the tskit.TreeSequence.
    sample_location_vectors : dict
        Contains all of the location vectors for the samples
    transition_matrix : np.matrix
        Instantaneous migration rate matrix between demes

    Returns
    -------
    tree_likelihood : float
        likelihood of the tree (product of the root likelihoods)
    root_log_likes : list
        List of root likelihoods (sum of the root locations vector)
    """

    children = arg.edges_child
    parents = arg.edges_parent

    log_messages = {}
    for l in sample_location_vectors:
        sample_parents = np.unique(parents[np.where(children == l)[0]])
        for p in sample_parents:
            bl = arg.node(p).time - arg.node(l).time
            if bl > 0:
                log_messages[f"{l}_{p}"] = np.log(np.matmul(sample_location_vectors[l], linalg.expm(transition_matrix*bl)))
            else:
                raise RuntimeError(f"Edge for sample {l} to {p} has length {bl}, which is not allowed.")
    
    nodes = list(arg.nodes(order="timeasc"))
    roots = [int(r) for r in set(parents).difference(set(children))]
    arg_likelihood = _propagate_up(nodes, arg, children, parents, log_messages, transition_matrix, roots)
    return arg_likelihood

def calc_generalized_migration_rate_log_likelihood(world_map, trees, migration_rates):
    """Calculates the composite log-likelihood of the specified migration rates across trees
    
    Loops through all trees and calculates the log-likelihood for each, before summing together.

    Parameters
    ----------
    world_map : terracotta.WorldMap

    trees : list
        List of tskit.Tree objects
    migration_rates : dict
        Keys are the connection type and values are the instantaneous migration
        rate along that connection

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """

    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
    log_likelihoods = []
    for arg in trees:
        log_likelihoods.append(_calc_generalized_log_likelihood(arg, world_map.sample_location_vectors, transition_matrix))
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods