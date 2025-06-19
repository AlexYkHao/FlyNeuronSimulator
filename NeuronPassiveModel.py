import os
import pickle
from typing import Optional, Tuple, List, Dict, Any
import numpy.typing as npt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuprint import Client, fetch_neurons, fetch_mean_synapses, fetch_synapse_connections, fetch_synapses,fetch_simple_connections
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from neuron import h, gui

class NeuronPassiveModel(object):
    """A class for simulating passive electrical properties of neurons from SWC files.
    
    Args:
        swc_path: Path to the SWC morphology file
        simu_config: Dictionary containing simulation parameters
    """
    # Class constants
    DEFAULT_SEGMENT_POSITION = 0.5
    DEFAULT_DPI = 300

    def __init__(self, swc_path: str, simu_config: Dict[str, Any]) -> None:
        self.swc_path = swc_path
        self.simu_config = simu_config
        self.injection_site = None  # should be a tuple of (sec, seg)
        self._load_swc()
        self._set_passive_properties()

    def _load_swc(self):
        h.load_file('import3d.hoc')
        dt = self.simu_config['dt']  # used to be 0.25  # Set the time step for the simulation
        h.dt = dt
        h.steps_per_ms = 1.0/dt
        cell = h.Import3d_SWC_read()
        if not os.path.exists(self.swc_path):
            raise FileNotFoundError(f"SWC file {self.swc_path} not found")
        try:
            cell.input(self.swc_path)
        except Exception as e:
            print(f"Error loading {self.swc_path}: {e}")
        self.imported_cell = h.Import3d_GUI(cell, 0)
        self.imported_cell.instantiate(None)
    
    def _set_passive_properties(self):
        for sec in h.allsec():
            sec.Ra = self.simu_config['Ra']
            sec.cm = self.simu_config['cm']
            sec.insert('pas')
            sec.g_pas = self.simu_config['g_pas']
            sec.e_pas = self.simu_config['e_pas']   
    
    def find_closest_sec(self, injection_point: npt.NDArray[np.float64]) -> None:
        """Find the closest section and segment to a given 3D point."""
        closest_sec = None
        closest_seg = None
        closest_dist = float('inf')
        for sec in h.allsec():
            # Get all 3D points for this section at once
            n_points = sec.n3d()
            if n_points == 0:
                continue
            
            # Vectorize distance calculation
            coords = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] 
                              for i in range(n_points)])
            distances = np.linalg.norm(coords - injection_point, axis=1)
            min_idx = np.argmin(distances)
            
            if distances[min_idx] < closest_dist:
                closest_dist = distances[min_idx]
                closest_sec = sec
                closest_seg = (min_idx/float(n_points-1) if n_points > 1 
                              else self.DEFAULT_SEGMENT_POSITION)
        
        self.injection_site = (closest_sec, closest_seg)
    
    def current_injection_process(self):
        # Define a stimulation protocol (inject a current)
        stim = h.IClamp(self.injection_site[0](self.injection_site[1])) 
        stim.amp = self.simu_config['injected_current']  
        stim.dur = self.simu_config['stim_duration']  
        stim.delay = self.simu_config['stim_start']  
        t = h.Vector().record(h._ref_t)  
        self.all_voltage = []
        for sec in h.allsec():
            voltage_of_sec = []
            for j in range(sec.n3d()):
                # record voltage at all segments
                voltage_of_sec.append(h.Vector().record(sec(j/float(sec.n3d()-1))._ref_v))
            self.all_voltage.append(voltage_of_sec)

        h.tstop = self.simu_config['simulation_length']
        h.run()  # Run the simulation
        self.t_vec = np.array(t.to_python())
    
    def post_processing(self):
        self.x_coords = []
        self.y_coords = []
        self.z_coords = []
        self.peak_amplitudes = []
        self.v_traces = []
        for i, sec in enumerate(h.allsec()):
            for j in range(sec.n3d()):
                x, y, z = sec.x3d(j), sec.y3d(j), sec.z3d(j)
                v = self.all_voltage[i][j]
                v = np.array(v.to_python())
                self.v_traces.append(v)
                self.x_coords.append(x)
                self.y_coords.append(y)
                self.z_coords.append(z)
                self.peak_amplitudes.append(max(v) - self.simu_config['e_pas'])

    def save_results(self, save_path: str) -> None:
        """Save simulation results to pickle file."""
        results_data = {
            'simu_config': self.simu_config,
            'x_coords': self.x_coords,
            'y_coords': self.y_coords, 
            'z_coords': self.z_coords,
            't': self.t_vec,
            'v_traces': self.v_traces,
            'peak_amplitudes': self.peak_amplitudes
        }        
        with open(save_path, 'wb') as f:
            pickle.dump(results_data, f)

    def plot_results(self, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.x_coords, self.y_coords, self.z_coords, c=self.peak_amplitudes, cmap='viridis')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        plt.colorbar(sc, label='Peak Amplitude (mV)')
        if save_path is not None:
            plt.savefig(save_path, dpi=self.DEFAULT_DPI)
            plt.close()
        else:
            plt.show()   



class NeuronInputSampler(object):
    def __init__(self, neuron_id, source_type=None, min_weight=5, scaling_factor=0.008):
        self.nid = neuron_id
        self.min_weight = min_weight
        self.injection_sites = None
        self.injection_site_coords = None
        self.scaling_factor = scaling_factor   # turn the voxel into micrometer
        self._get_neuron_df()
        if source_type is not None:
            self.source_type = source_type
        else:
            self._find_dominant_source_type()
    
    def _get_neuron_df(self):
        self.neuron_df, self.roi_conn_df = fetch_neurons(NC(bodyId=self.nid))   
        self.neuron_type = self.neuron_df.iloc[0]['type']
    
    def _find_dominant_source_type(self):
        conn_df = fetch_simple_connections(
            upstream_criteria=None,  # Any upstream neuron
            downstream_criteria= NC(type=self.neuron_type),  # Use the target neuron's type
            min_weight=self.min_weight   # Minimum number of synapses
        )
        # Group by upstream neuron type and sum synapses
        synapse_counts = conn_df.groupby('type_pre')['weight'].sum().sort_values(ascending=False)
        # Get the pre-synaptic type with most synapses to type A
        self.source_type = synapse_counts.index[0]

    def sample_input_site(self, roi=None, n_samples=5, sample_type='random'):
        conn_df = fetch_simple_connections(
            upstream_criteria=NC(type=self.source_type),  # Any upstream neuron
            downstream_criteria=self.nid,  # Type A as downstream
            min_weight=self.min_weight   # Minimum number of synapses
        )
        if len(conn_df) == 0:
            print(f"No synapses found for {self.nid}")
            return
        syn_df = fetch_synapse_connections(source_criteria=conn_df['bodyId_pre'].to_list(), target_criteria=self.nid)
        # ditch the bodyId_pre, bodyId_post, and roi_pre columns
        syn_df = syn_df.drop(columns=['bodyId_post', 'roi_pre'])
        syn_df['weight'] = 1.0
        # groupn syn_df by bodyId_post and roi_post, and average the x, y, z coordinates
        syn_df_grouped = syn_df.groupby(['bodyId_pre','roi_post']).mean()
        syn_df_grouped = syn_df_grouped.reset_index()
        if roi is not None:
            syn_df_grouped = syn_df_grouped[syn_df_grouped['roi_post']==roi]
        # now draw n_samples from the syn_df_grouped and save the x, y, z coordinates in the injection_sites
        n_samples = min(n_samples, len(syn_df_grouped))
        if sample_type == 'random':
            self.injection_sites = syn_df_grouped.sample(n_samples)
        elif sample_type == 'top':
            self.injection_sites = syn_df_grouped.sort_values(by='weight', ascending=False).head(n_samples)
        else:
            raise ValueError(f"Invalid sample_type: {sample_type}")
        # turn the coordinates into micrometer
        self.injection_site_coords = self.injection_sites[['x_post', 'y_post', 'z_post']].to_numpy() * self.scaling_factor
    
    def save_injection_sites(self, save_path: str) -> None:
        if self.injection_sites is None:
            return
        self.injection_sites.to_pickle(save_path)

    def load_injection_sites(self, load_path: str) -> None:
        self.injection_sites = pd.read_pickle(load_path)
        self.injection_site_coords = self.injection_sites[['x_post', 'y_post', 'z_post']].to_numpy() * self.scaling_factor

