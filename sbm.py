"""
Methods to estimate the parameters of a stochastic block model for a given set of policy positions;
Also includes methods to get parameters from an SBM object and other useful methods
"""

# Check whether graph-tool is installed
import os
from datetime import datetime

try:
    import graph_tool
except ImportError:
    print(
        "graph-tool is not installed. Install and try again."
        "See https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions"
    )
    exit(1)

import graph_tool.all as gt
from collections import defaultdict
import numpy as np
import pandas as pd
import tqdm
import networkx as nx
import pickle


def get_bipartite_adjacency_matrix_kcore(positions, k_core=(5, 2)):
    """
    Construct the adjacency matrix A from the positions dataframe; A is a pandas dataframe
    :param positions (pandas dataframe): the positions dataframe
    :param k_core (tuple): the minimum number of clients and bills for the k-core
    :return A (pandas dataframe): the adjacency matrix
    """

    # Keep only rows with a client_uuid and a bill_identifier and a position_numeric in [-1, 1]
    selection = positions[positions.client_uuid.notnull()].copy()
    selection = selection[selection.bill_identifier.notnull()].copy()
    selection = selection[selection.position_numeric.isin([-1, 1])].copy()

    # Calculate number of positions per client and bill
    n_client_positions = selection.client_uuid.value_counts()
    n_bill_positions = selection.bill_identifier.value_counts()

    # Keep only clients and bills with at least k_core[0] and k_core[1] positions, respectively
    selection = selection[selection.client_uuid.map(n_client_positions) >= shape[0]]
    selection = selection[selection.bill_identifier.map(n_bill_positions) >= shape[1]]

    # Calculate the adjacency matrix
    A = selection.groupby(['client_uuid', 'bill_identifier']).position_numeric.sum().unstack()

    # Because clients can record multiple positions on the same bill, some values in A can be > 1 or < -1
    # We set these values to 1 or -1, respectively
    A = np.sign(A)

    # Delete the selection of the positions dataframe to save memory
    del selection

    # Create the k-core of the adjacency matrix by iteratively removing nodes with degree < k_core[0]
    # and nodes with degree < k_core[1], for clients and bills, respectively, until no such nodes remain
    while any(abs(A).sum(1) < shape[0]) or any(abs(A).sum(0) < shape[1]):
        A = A.loc[abs(A).sum(1) >= shape[0], :]
        A = A.loc[:, abs(A).sum(0) >= shape[1]]

    G = nx.from_edgelist(A.stack().dropna().reset_index().values[:, :2].tolist())
    components = sorted(nx.connected_components(G), key=len)

    # If there are no connected components, return None
    if len(components) == 0:
        return None

    # Remove any connected components that are not connected to the largest connected component
    c = components[-1]
    A = A.reindex(index=sorted(c & {*A.index.values}), columns=sorted(c & {*A.columns.values}))

    return A


def get_bipartite_graph(A, allowed_positions=[1, -1]):
    """
    Construct a bipartite graph from the adjacency matrix A; A should be a pandas dataframe
    in which the rows are interest groups and the columns are bills; the values in A should
    be the positions of the interest groups on the bills; the allowed_positions parameter
    specifies which positions are allowed in the graph (e.g. 1 and -1 for a binary model)

    This code is adapted from the sbmtm model code:
     https://github.com/martingerlach/hSBM_Topicmodel
    :param A (pandas dataframe): the adjacency matrix
    :param allowed_positions (optional): the positions that are allowed in the graph
    :return g (graph-tool graph): the bipartite graph
    """

    # Construct edgelist; for now, keep only positive and negative positions
    E_combo = A[~A.isna()].stack()
    E_combo = E_combo.reset_index(drop=False)
    E_combo.columns = ['source', 'target', 'weight']
    E_combo = E_combo[E_combo.weight.isin(allowed_positions)]

    g = gt.Graph(directed=False)

    # define node properties
    # name: clients - client_uuid, bills - bill_identifier
    # kind: clients - 0, bills - 1
    name = g.vp["name"] = g.new_vp("string")
    kind = g.vp["kind"] = g.new_vp("int")
    etype = g.ep["weight"] = g.new_ep("int")

    bills_add = defaultdict(lambda: g.add_vertex())
    clients_add = defaultdict(lambda: g.add_vertex())

    igs = E_combo.source.unique()
    bls = E_combo.target.unique()

    I = len(igs)
    # add all interest groups first
    for i in range(I):
        ig = igs[i]
        d = clients_add[ig]
        name[d] = ig
        kind[d] = 0

    # add all bills
    for i in range(len(bls)):
        bill = bls[i]
        b = bills_add[bill]
        name[b] = bill
        kind[b] = 1

    # add all edges and assign their type = to the numeric position
    for i in tqdm.tqdm(range(len(E_combo))):
        i_client = np.where(igs == E_combo.iloc[i]['source'])[0][0]
        i_bill = np.where(bls == E_combo.iloc[i]['target'])[0][0]
        e = g.add_edge(i_client, I + i_bill)
        etype[e] = E_combo.iloc[i]['weight']

    return g


def remove_redundant_levels(state):
    """
    Remove redundant levels from a blockstate
    Taken from sbmtm model code: https://github.com/martingerlach/hSBM_Topicmodel
    :param state (graph-tool blockstate): the blockstate to remove redundant levels from
    :return state (graph-tool blockstate): the blockstate with redundant levels removed
    """

    state_tmp = state.copy()
    mdl = state_tmp.entropy()

    L = 0
    for s in state_tmp.levels:
        L += 1
        if s.get_nonempty_B() == 2:
            break
    state_tmp = state_tmp.copy(bs=state_tmp.get_bs()[:L] + [np.zeros(1)])

    mdl_tmp = state_tmp.entropy()
    if mdl_tmp < mdl:
        mdl = 1.0 * mdl_tmp
        state = state_tmp.copy()

    return state


def run_blockmodel(positions: pd.DataFrame,
                   deg_corr: bool = True,
                   layers: bool = False,
                   overlap: bool = False,
                   k_core: tuple = (2, 2),
                   load_model: str = None,
                   save_model: str = None,
                   verbose: bool = True,
                   ):
    """
    Run the blockmodel on the positions dataframe and save the results to saveloc
    Note that the positions dataframe should have the following columns:
    client_uuid, bill_identifier, position_numeric

    :param verbose: the verbosity level (0, 1, or 2)
    :param save_model: the location to save the model
    :param load_model: the location to load the model from (if None, run the model from scratch)
    :param positions: the positions dataframe
    :param deg_corr: whether to use degree correction
    :param layers: whether to use a layered or categorical model
    :param overlap: whether to allow overlapping blocks
    :param saveloc: where to save the results
    :param k_core: the k-core to use for the blockmodel

    :type deg_corr: bool (default: True)
    :type layers: bool (default: False)
    :type overlap: bool (default: False)
    :type k_core: tuple (default: (2,2))
    :type load_model: str (default: None)
    :type save_model: str (default: None)
    :type verbose: bool (default: True)
    :type positions: pd.DataFrame

    :return graph-tool blockstate: the blockstate
    """

    df = positions.copy()

    adj_matrix = get_bipartite_adjacency_matrix_kcore(df, k_core=k_core)
    if adj_matrix is None:
        print("No connected components; exiting")
        return None

    g = get_bipartite_graph(adj_matrix)

    if not overlap:
        clabel = g.vp['kind']
    else:
        clabel = None

    def save_state(state):
        # Save the model
        # if save_model was specified, save the model there
        if save_model is not None:
            assert os.path.exists(save_model), "Save location does not exist"
            saveloc = save_model

        # otherwise, save the model in a folder with the date
        else:
            # Save the model in a folder with the date
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            saveloc = f"models/blockmodel/{date}"
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)

        # Save the parameters
        params = {'deg_corr': deg_corr,
                  'layers': layers,
                  'overlap': overlap,
                  'k_core': k_core}
        with open(f"{saveloc}/params.pkl", 'wb') as f:
            pickle.dump(params, f)

        # Save the blockstate
        with open(f"{saveloc}/blockstate.pkl", 'wb') as f:
            pickle.dump(state, f)

    # If a load_model is specified, load the model from that location
    if os.path.exists(load_model):
        state = pickle.load(open(load_model, 'rb'))
        print("Loaded model from ", load_model)

    elif os.path.exists('models/blockmodel/' + load_model):
        state = pickle.load(open('models/blockmodel/' + load_model, 'rb'))
        print("Loaded model from ", 'models/blockmodel/' + load_model)

    else:
        print("Estimating 20 blockstates and choosing the best one...")

        min_E = np.inf
        state = None
        for iteration in range(20):
            new_state = gt.minimize_nested_blockmodel_dl(
                g,
                state_args=dict(
                    base_type=gt.LayeredBlockState,
                    clabel=clabel, pclabel=clabel,  # impose hard bipartite constraint
                    state_args=dict(ec=g.ep.weight,
                                    layers=layers,
                                    deg_corr=deg_corr,
                                    overlap=overlap,
                                    )),
                multilevel_mcmc_args=dict(verbose=False)
            )
            E = new_state.entropy()
            print(f"Entropy iteration {iteration}: {E}")
            if E < min_E:
                state = new_state
            min_E = E
            print("Done.")

    # Remove redundant levels
    state = remove_redundant_levels(state)

    # Save the state to the saveloc
    save_state(state)

    if not overlap:

        print("Annealing state...")
        gt.mcmc_anneal(state,
                       beta_range=(1, 10),
                       niter=1000,
                       mcmc_equilibrate_args=dict(force_niter=10))
        print("Done.")

    else:

        print("Sweeping state")
        pbar = tqdm.tqdm(total=1000)
        for i in range(1000):  # this should be sufficiently large
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
            pbar.update(1)

    # Save the state to the saveloc
    save_state(state)

    return state
