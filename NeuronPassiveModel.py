import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from neuron import h, gui

class NeuronPassiveModel(object):
    def __init__(self, swc_path: str, simu_config: dict):
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
    
    def find_closest_sec(self, injection_point: np.ndarray):
        closest_sec = None
        closest_seg = None
        closest_dist = float('inf')
        for sec in h.allsec():
            for i in range(sec.n3d()):
                x, y, z = sec.x3d(i), sec.y3d(i), sec.z3d(i)
                dist = np.sqrt((x - injection_point[0])**2 + (y - injection_point[1])**2 + (z - injection_point[2])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_sec = sec
                    closest_seg = i/float(sec.n3d()-0.9)  # the point_process object on the closest section, 0.9 instead of 1 to avoid zero division
        self.injection_site = (closest_sec, closest_seg)
    
    def current_injection_process(self, injected_current: float, stim_duration: float, stim_start: float):
        # Define a stimulation protocol (inject a current)
        stim = h.IClamp(self.injection_site[0](self.injection_site[1])) 
        stim.amp = injected_current  
        stim.dur = stim_duration  
        stim.delay = stim_start  
        t = h.Vector().record(h._ref_t)  
        self.all_volrage = []
        for sec in h.allsec():
            voltage_of_sec = []
            for j in range(sec.n3d()):
                # record voltage at all segments
                voltage_of_sec.append(h.Vector().record(sec(j/float(sec.n3d()-1))._ref_v))
            self.all_volrage.append(voltage_of_sec)

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
                v = self.all_volrage[i][j]
                v = np.array(v.to_python())
                self.v_traces.append(v)
                self.x_coords.append(x)
                self.y_coords.append(y)
                self.z_coords.append(z)
                self.peak_amplitudes.append(max(v) - self.simu_config['e_pas'])

    def save_results(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump({'x_coords': self.x_coords, 'y_coords': self.y_coords, 'z_coords': self.z_coords, 
                         't': self.t_vec, 'v_traces': self.v_traces,
                         'peak_amplitudes': self.peak_amplitudes}, f)

    def plot_results(self, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.x_coords, self.y_coords, self.z_coords, c=self.peak_amplitudes, cmap='viridis')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        plt.colorbar(sc, label='Peak Amplitude (mV)')
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()    