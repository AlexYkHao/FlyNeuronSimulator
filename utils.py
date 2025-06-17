import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import navis
import navis.interfaces.neuprint as neu
from neuprint import Client, fetch_neurons, fetch_skeleton, fetch_synapses, attach_synapses_to_skeleton, fetch_simple_connections, fetch_synapse_connections
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC



def skeleton_df_to_swc(df, rescaling_factor=0.008, export_path=None):
    """
    Create an SWC file from a skeleton DataFrame.

    Args:
        df:
            DataFrame, as returned by :py:meth:`.Client.fetch_skeleton()
        rescaling_factor:
            Optional. Rescale the coordinates so that the unit is in um.
        export_path:
            Optional. Write the SWC file to disk a the given location.

    Returns:
        ``str``
    """
    df = df.copy()
    # rescale coordinates and radius
    df['x'] *= rescaling_factor 
    df['y'] *= rescaling_factor
    df['z'] *= rescaling_factor
    df['radius'] *= rescaling_factor
    if 'structure' not in df.columns:
        df['structure'] = 'neurite'
    # Add a new column for node type based on the structure, if structure is neurite, set node_type to 4, if synapse, set node_type to 5
    df['node_type'] = np.where(df['structure'] == 'neurite', 4, 5)
    # write the file to swc format
    df = df[['rowId', 'node_type', 'x', 'y', 'z', 'radius', 'link']]
    swc = "# "
    swc += df.to_csv(sep=' ', header=True, index=False)

    if export_path:
        with open(export_path, 'w') as f:
            f.write(swc)

    return swc