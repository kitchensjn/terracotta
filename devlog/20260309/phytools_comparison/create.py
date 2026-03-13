import tskit
import newick
import pandas as pd
import os


def from_newick(
    string, *, min_edge_length=0, span=1, time_units=None, node_name_key=None
) -> tskit.TreeSequence:
    """
    Create a tree sequence representation of the specified newick string.

    The tree sequence will contain a single tree, as specified by the newick. All
    leaf nodes will be marked as samples (``tskit.NODE_IS_SAMPLE``). Newick names and
    comments will be written to the node metadata. This can be accessed using e.g.
    ``ts.node(0).metadata["name"]``.

    :param string string: Newick string
    :param float min_edge_length: Replace any edge length shorter than this value by this
        value. Unlike newick, tskit doesn't support zero or negative edge lengths, so
        setting this argument to a small value is necessary when importing trees with
        zero or negative lengths.
    :param float span: The span of the tree, and therefore the
        :attr:`~TreeSequence.sequence_length` of the returned tree sequence.
    :param str time_units: The value assigned to the :attr:`~TreeSequence.time_units`
        property of the resulting tree sequence. Default: ``None`` resulting in the
        time units taking the default of :attr:`tskit.TIME_UNITS_UNKNOWN`.
    :param str node_name_key: The metadata key used for the node names. If ``None``
        use the string ``"name"``, as in the example of accessing node metadata above.
        Default ``None``.
    :return: A tree sequence consisting of a single tree.
    """
    
    trees = newick.loads(string)
    if len(trees) > 1:
        raise ValueError("Only one tree can be imported from a newick string")
    if len(trees) == 0:
        raise ValueError("Newick string was empty")
    tree = trees[0]
    tables = tskit.TableCollection(span)
    if time_units is not None:
        tables.time_units = time_units
    if node_name_key is None:
        node_name_key = "name"
    nodes = tables.nodes
    nodes.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                node_name_key: {
                    "type": ["string"],
                    "description": "Name from newick file",
                },
                "comment": {
                    "type": ["string"],
                    "description": "Comment from newick file",
                },
            },
        }
    )

    id_map = {}

    def get_or_add_node(newick_node, time):
        if newick_node not in id_map:
            flags = tskit.NODE_IS_SAMPLE if len(newick_node.descendants) == 0 else 0
            metadata = {}
            if newick_node.name:
                metadata[node_name_key] = newick_node.name
            if newick_node.comment:
                metadata["comment"] = newick_node.comment
            id_map[newick_node] = tables.nodes.add_row(
                flags=flags, time=time, metadata=metadata
            )
        return id_map[newick_node]

    root = next(tree.walk())
    get_or_add_node(root, 0)
    for newick_node in tree.walk():
        node_id = id_map[newick_node]
        for child in newick_node.descendants:
            length = max(child.length, min_edge_length)
            if length <= 0:
                raise ValueError(
                    "tskit tree sequences cannot contain edges with lengths"
                    " <= 0. Set min_edge_length to force lengths to a"
                    " minimum size"
                )
            child_node_id = get_or_add_node(child, nodes[node_id].time - length)
            tables.edges.add_row(0, span, node_id, child_node_id)
    # Rewrite node times to fit the tskit convention of zero at the youngest leaf
    nodes = tables.nodes.copy()
    youngest = min(tables.nodes.time)
    tables.nodes.clear()
    for node in nodes:
        tables.nodes.append(node.replace(time=node.time - youngest + root.length))
    tables.sort()
    return tables.tree_sequence()

newick_tree = open("data/tutorial_tree.newick", "r").read()[:-1]
tree = from_newick(newick_tree)
tree.dump("data/trees/0.trees")

feeding_mode = pd.read_csv("data/feeding_mode.csv")
feeding_mode.columns = ["species","feeding.mode","gape.width","buccal.length"]
for sample in tree.samples():
    feeding_mode.loc[feeding_mode["species"]==tree.node(sample).metadata["name"], "id"] = sample
feeding_mode["id"] = feeding_mode["id"].astype(int)
feeding_mode["deme"] = 0
feeding_mode.loc[feeding_mode["feeding.mode"]=="non", "deme"] = 1
feeding_mode.loc[feeding_mode["feeding.mode"]=="pisc", "deme"] = 2

samples = feeding_mode.loc[:, ["id", "deme", "species"]]
samples.to_csv("data/samples.tsv", sep="\t", index=False)

demes = pd.DataFrame({"id":[0,1,2], "xcoord":[0,1,0], "ycoord":[0,1,1], "suitability":[1,1,1]})
connections = pd.DataFrame({"id":[0,1,2,3,4,5,], "deme_0":[0,0,1,1,2,2], "deme_1":[1,2,0,2,0,1], "migration_modifier":["a","b","c","d","e","f"]})
demes.to_csv("data/demes.tsv", sep="\t", index=False)
connections.to_csv("data/connections.tsv", sep="\t", index=False)