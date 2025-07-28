import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from typing import Union, Optional, List
import plotly.graph_objects as go

import navis
import navis.interfaces.neuprint as neu
from neuprint import Client, fetch_neurons, fetch_skeleton, fetch_mean_synapses, fetch_synapses
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


def get_most_important_column_coords(nid, roi_info:dict):
    # for a given neuron, figure out a column with the most input
    per_column_input = {}
    for k, v in roi_info.items():
        if '_col_' in k:
            if 'post' in v.keys():
                per_column_input[k] = v['post']

    # sort the columns by the number of inputs
    sorted_columns = {k: v for k, v in sorted(per_column_input.items(), key=lambda item: item[1], reverse=True)}
    most_input_column = list(sorted_columns.keys())[0]

    # fetch the mean synapses in this column
    mean_synapses_df = fetch_mean_synapses(nid, SC(type='post', rois=most_input_column, primary_only=False))
    return mean_synapses_df[['x', 'y', 'z']].mean()


def get_decay_delay(simulation_pickle: str):
    result_dict = {}
    with open(simulation_pickle, 'rb') as f:
        data = pickle.load(f)
        v_traces = data['v_traces']
        t_vec = data['t']
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        z_coords = data['z_coords']
        p_coords = np.array([x_coords, y_coords, z_coords]).T
        # now calculate v-trace weighted centroid time for each v-trace
        centroid_times = []
        AOC = []
        time_variances = []
        for v_trace in v_traces:
            v_trace = v_trace - v_trace[0]
            A = np.sum(v_trace) + 0.0001
            B = np.sum(v_trace * t_vec)
            C = np.sum(v_trace * t_vec**2)
            AOC.append(A)
            centroid_times.append(B/A)
            time_variances.append(C/A - (B/A)**2)
        result_dict['max_decay_ratio'] = max(AOC)/min(AOC)
        result_dict['peak_time'] = max(centroid_times) - min(centroid_times)
        result_dict['width_change'] = max(time_variances) - min(time_variances)
        max_AOC_idx = np.argmax(AOC)
        min_AOC_idx = np.argmin(AOC)
        result_dict['min_max_distance'] = np.linalg.norm(p_coords[max_AOC_idx] - p_coords[min_AOC_idx])
        # calculate the max distance between two points
        hull = ConvexHull(p_coords)
        hull_points = p_coords[hull.vertices]
        distances = cdist(hull_points, hull_points)
        result_dict['max_distance'] = np.max(distances)
    return result_dict


def get_masked_decay_delay(simulation_pickle: str, 
                           sec_mapping_path:str, 
                           region_mask_path:str):
    with open(region_mask_path, 'rb') as f:
        region_mask = pickle.load(f)
    with open(sec_mapping_path, 'rb') as f:
        seg2seg_mapping = pickle.load(f)

    region_mask_dict = {sec: reg for reg in region_mask.keys() for sec in list(region_mask[reg]) }
    grouped_traces = {reg:[] for reg in region_mask.keys()}
    grouped_points = {reg:[] for reg in region_mask.keys()}
   
    with open(simulation_pickle, 'rb') as f:
        data = pickle.load(f)
        t_vec = data['t']
        i = 0 
        for x, y, z, v_trace in zip(data['x_coords'], data['y_coords'], data['z_coords'], data['v_traces']):
            which_sec = seg2seg_mapping[i]
            i += 1
            if which_sec not in region_mask_dict.keys():
                continue
            which_group = region_mask_dict[which_sec]
            grouped_traces[which_group].append(v_trace - v_trace[0])
            grouped_points[which_group].append([x, y, z])

    result_dict = {'AOC':{}, 'centroid_time':{}, 'time_variance':{}, 'group_centroids':{}}
    for reg in region_mask.keys():
        v_trace = grouped_traces[reg]
        v_trace = np.array(v_trace)
        p_coords = grouped_points[reg]
        p_coords = np.array(p_coords)
        mean_v_trace = v_trace.mean(axis=0)
        A = np.sum(mean_v_trace) + 0.0001
        B = np.sum(mean_v_trace * t_vec)
        C = np.sum(mean_v_trace * t_vec**2)
        result_dict['AOC'][reg] = A
        result_dict['centroid_time'][reg] = B/A
        result_dict['time_variance'][reg] = C/A - (B/A)**2
        result_dict['group_centroids'][reg] = p_coords.mean(axis=0)
    return result_dict


def sample_v_traces(simulation_pickle: str, 
                    max_trace_overlay: bool = False, 
                    trace_idx: float = 0.5, xlim: tuple = (0, 1000)):
    with open(simulation_pickle, 'rb') as f:
        data = pickle.load(f)
        v_traces = data['v_traces']
        t_vec = data['t']
        peak_amplitudes = data['peak_amplitudes']
    max_trace_idx = np.argmax(peak_amplitudes)
    plot_trace_idx = int(trace_idx * len(v_traces))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_vec, v_traces[plot_trace_idx], label='Trace {}'.format(plot_trace_idx))
    if max_trace_overlay:
        ax.plot(t_vec, v_traces[max_trace_idx], label='Max Trace')
    ax.legend()
    ax.set_xlim(xlim)
    plt.show()


def sample_grouped_v_traces(simulation_pickle: str, 
                           sec_mapping_path:str, 
                           region_mask_path:str,
                           max_trace_overlay: bool = False):
    with open(region_mask_path, 'rb') as f:
        region_mask = pickle.load(f)
    with open(sec_mapping_path, 'rb') as f:
        seg2seg_mapping = pickle.load(f)

    region_mask_dict = {sec: reg for reg in region_mask.keys() for sec in list(region_mask[reg]) }
    grouped_traces = {reg:[] for reg in region_mask.keys()}
   
    with open(simulation_pickle, 'rb') as f:
        data = pickle.load(f)
    t_vec = data['t']
    peak_amplitudes = data['peak_amplitudes']
    max_trace_idx = np.argmax(peak_amplitudes)
    i = 0 
    for v_trace in data['v_traces']:
        which_sec = seg2seg_mapping[i]
        i += 1
        if which_sec not in region_mask_dict.keys():
            continue
        which_group = region_mask_dict[which_sec]
        grouped_traces[which_group].append(v_trace - v_trace[0])
    fig, ax = plt.subplots(figsize=(10, 6))
    for reg in region_mask.keys():
        ax.plot(t_vec, np.mean(grouped_traces[reg], axis=0), label=reg)
    if max_trace_overlay:
        max_trace = data['v_traces'][max_trace_idx]
        max_trace = max_trace - max_trace[0]
        ax.plot(t_vec, max_trace, label='Max Trace')
    ax.legend()
    plt.show()


def plot_bars_with_scatter(df: pd.DataFrame, 
                          x_col: str, 
                          y_col: str,
                          figsize: tuple = (10, 6),
                          bar_alpha: float = 0.7,
                          scatter_alpha: float = 0.6,
                          scatter_size: float = 30,
                          jitter_width: float = 0.3,
                          bar_color: str = 'skyblue',
                          scatter_color: str = 'red',
                          error_bars: bool = True,
                          title: Optional[str] = None,
                          xlabel: Optional[str] = None,
                          ylabel: Optional[str] = None) -> plt.Figure:
    """
    Create a bar plot with overlaid scatter points.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis (categorical labels)
        y_col: Column name for y-axis (lists of numbers)
        figsize: Figure size as (width, height)
        bar_alpha: Transparency of bars (0-1)
        scatter_alpha: Transparency of scatter points (0-1)
        scatter_size: Size of scatter points
        jitter_width: Width of horizontal jitter for scatter points
        bar_color: Color of the bars
        scatter_color: Color of scatter points
        error_bars: Whether to show error bars (standard error)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    
    Returns:
        matplotlib Figure object
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories
    categories = df[x_col].unique()
    
    # Calculate means and standard errors for each category
    means = []
    std_errors = []
    
    for cat in categories:
        # Get all values for this category
        cat_data = df[df[x_col] == cat][y_col]
        
        # Flatten all lists for this category
        all_values = []
        for item in cat_data:
            if isinstance(item, (list, np.ndarray)):
                all_values.extend(item)
            else:
                all_values.append(item)
        
        all_values = np.array(all_values)
        means.append(np.mean(all_values))
        std_errors.append(np.std(all_values) / np.sqrt(len(all_values)))
    
    # Create bar plot
    x_positions = np.arange(len(categories))
    bars = ax.bar(x_positions, means, alpha=bar_alpha, color=bar_color, 
                  label='Mean', width=0.6)
    
    # Add error bars if requested
    if error_bars:
        ax.errorbar(x_positions, means, yerr=std_errors, 
                   fmt='none', color='black', capsize=5, capthick=1)
    
    # Add scatter points
    for i, cat in enumerate(categories):
        cat_data = df[df[x_col] == cat][y_col]
        
        # Collect all individual points
        y_points = []
        for item in cat_data:
            if isinstance(item, (list, np.ndarray)):
                y_points.extend(item)
            else:
                y_points.append(item)
        
        # Create jittered x positions
        n_points = len(y_points)
        x_jitter = np.random.uniform(-jitter_width/2, jitter_width/2, n_points)
        x_points = np.full(n_points, i) + x_jitter
        
        # Plot scatter points
        ax.scatter(x_points, y_points, alpha=scatter_alpha, 
                  s=scatter_size, color=scatter_color, 
                  label='Data points' if i == 0 else "")
    
    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f'{y_col} by {x_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()


def show_regional_masks_3d(simulation_pickle: str, 
                           sec_mapping_path:str, 
                           region_mask_path:str, 
                           save_path=None):
    colors = ['gray', 'red', 'blue', 'green', 'yellow', 'purple', 'orange']
    n_colors = len(colors)
    with open(region_mask_path, 'rb') as f:
        region_mask = pickle.load(f)
    with open(sec_mapping_path, 'rb') as f:
        seg2seg_mapping = pickle.load(f)
    with open(simulation_pickle, 'rb') as f:
        data = pickle.load(f)

    region_mask_dict = {sec: reg for reg in region_mask.keys() for sec in list(region_mask[reg]) }

    unlabeled_set = set(seg2seg_mapping.values()) - set(region_mask_dict.keys())
    region_mask_dict.update({lb:'others' for lb in list(unlabeled_set)})

    colored_skeleton = {}
    colored_skeleton['others'] = []
    colored_skeleton.update({reg:[] for reg in region_mask.keys()})
    colored_skeleton
    i = 0
    for p in zip(data['x_coords'], data['y_coords'], data['z_coords']):
        which_sec = seg2seg_mapping[i]
        which_group = region_mask_dict[which_sec]
        colored_skeleton[which_group].append(p)
        i += 1

    fig = go.Figure()
    for i, (k, v) in enumerate(colored_skeleton.items()):
        if len(v) == 0:
            continue
        p_array = np.array(v).T
        fig.add_trace(go.Scatter3d(
            x=p_array[0], y=p_array[1], z=p_array[2],
            mode='markers',
            marker=dict(
                color=colors[i%n_colors], # Sets all markers to blue
                size=2,
                opacity=0.5
            ),
            name=k,
            showlegend=False  # Turn on if you want to see section names
        ))
    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    if save_path:
        fig.write_html(save_path)
    else:
        return fig
    
    
def find_column_structure(nid, column_name:str, max_item=20, rescaling_factor=0.008):
    # download the synapses projecting on the neuron, and aggregate them by roi
    synapses_df = fetch_synapses(NC(bodyId=nid), SC(type='post', primary_only=False))
    synapses_df_reduced = synapses_df.drop(columns=['type'])
    synapses_df_reduced = synapses_df_reduced.groupby(['roi']).agg({
                'x': list, # list,
                'y': list, 
                'z': list,
                'confidence': 'mean',
                'bodyId': 'count'  # Count number of synapses
            }).rename(columns={'bodyId': 'weight'})
    synapses_df_reduced = synapses_df_reduced.reset_index()

    # find the rows with roi containing the column_name, and sort by synapse count (weight)
    synapses_df_layered = synapses_df_reduced.loc[synapses_df_reduced['roi'].str.contains(column_name)]
    synapses_df_layered = synapses_df_layered.sort_values(by='weight', ascending=False).reset_index(drop=True)
    
    synapses_df_layered = synapses_df_layered.iloc[:max_item]
    # aggregate the x, y, z columns to get the mean of each list
    synapses_df_layered['cx'] = synapses_df_layered['x'].apply(lambda x: rescaling_factor*np.mean(x))
    synapses_df_layered['cy'] = synapses_df_layered['y'].apply(lambda x: rescaling_factor*np.mean(x))
    synapses_df_layered['cz'] = synapses_df_layered['z'].apply(lambda x: rescaling_factor*np.mean(x))

    # merge [x, y, z] into a single column
    synapses_df_layered['p'] = synapses_df_layered[['cx', 'cy', 'cz']].apply(lambda x: np.array(x), axis=1)
    return synapses_df_layered

def construct_convex_hull(synapses_df_layered, p, distance_threshold=5, rescaling_factor=0.008):
    from scipy.spatial import ConvexHull, Delaunay
    # find the row in synapses_df_layered, whose x, y, z are closest to the injection site p
    distance_df = synapses_df_layered.apply(lambda row: np.linalg.norm(row['p'] - p), axis=1)
    # find the row with the minimum distance
    closest_roi = distance_df.idxmin()
    distance = distance_df.min()
    if distance > distance_threshold:
        print(f'The distance between the injection site and the closest synapse is less than the threshold ({distance_threshold} um)')
        return None

    chosen_column = synapses_df_layered.loc[closest_roi]
    points = np.array(chosen_column[['x', 'y', 'z']].tolist()).T * rescaling_factor
    hull = ConvexHull(points)
    # Get the vertices of the convex hull
    hull_vertices = points[hull.vertices]
    hull_delaunay = Delaunay(points[hull.vertices])
    return hull_delaunay

def construct_convex_hull_all(synapses_df_layered, rescaling_factor=0.008):
    from scipy.spatial import ConvexHull, Delaunay
    roi_hull_dict = {}
    for i, row in synapses_df_layered.iterrows():
        points = np.array(row[['x', 'y', 'z']].tolist()).T * rescaling_factor
        if points.shape[0] <= 4:
            continue  # not enough points to construct a hull
        hull = ConvexHull(points)
        # Get the vertices of the convex hull
        hull_delaunay = Delaunay(points[hull.vertices])
        roi_hull_dict[row['roi']] = hull_delaunay
    return roi_hull_dict