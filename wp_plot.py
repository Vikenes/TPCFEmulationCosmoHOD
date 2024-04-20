import numpy as np
from pathlib import Path
from datetime import datetime 
import pandas as pd
import h5py
import os 
import yaml 
from typing import List, Optional, Union
from collections.abc import Iterable
import time 
import sys 
sys.path.append("../emul_utils")
from _nn_config import DataConfig #, TrainingConfig, ModelConfig
from _predict import Predictor

import matplotlib.pyplot as plt
from matplotlib import gridspec
from _plot import set_matplotlib_settings, get_CustomCycler
set_matplotlib_settings()
custom_cycler = get_CustomCycler()

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline as ius
from scipy.integrate import simpson 

import warnings 
warnings.filterwarnings("ignore", category=UserWarning, message="Input line")

"""

WARNING!!!

Fix proj_corrfunc! 
There are factors of 10**xi hiding 




"""

global SAVEFIG 
global PUSH
global PRESENTATION

SAVEFIG         = False
PUSH            = False
PRESENTATION    = False

dataset_names = {"train": "Training", "val": "Validation", "test": "Test"}

class cm_emulator_class:
    def __init__(
            self, 
            LIGHTING_LOGS_PATH,
            version=0,
            ):
        self.predictor = Predictor.from_path(f"{LIGHTING_LOGS_PATH}/version_{version}")

    def __call__(
        self,
        params,
        return_tensors: bool = False,
    ):
        inputs = np.array(params)
        return self.predictor(inputs).reshape(-1)


class emulator_test:
    def __init__(self,
                root_dir:  str = "./emulator_data",
                dataset:   str = "vary_r",
                emul_dir:  str = "compare_scaling",
                flag:      str = "val",
                print_config_param:     List[str] = None,
                ):
        self.dataset    = dataset
        self.data_dir   = Path(f"{root_dir}/{dataset}")
        self.emul_dir   = Path(self.data_dir / "emulators" / emul_dir) # name of emulator logs directory
        self.fig_dir    = Path(f"./plots/{dataset}/{emul_dir}") # name of emulator plots directory 
        if not self.emul_dir.exists():
            # Check if the emulator logs directory exists
            raise FileNotFoundError(f"Path {self.emul_dir} does not exist.")

        self.flag           = flag # data set to be plotted 
        self.N_versions     = len(sorted(self.emul_dir.glob("version_*"))) # number of versions of the emulator
        """
        Get the name of all features and labels from the config file.
        For each emulator version, they are identical, so we get it from version_0.
        This config file should not be used for anything else!
        Elsewhere, the config file for each version should be used. 
        """
        self.config         = yaml.safe_load(open(self.emul_dir/ "version_0/config.yaml", "r"))
        data_config         = DataConfig(**self.config["data"])
       
        self.param_names    = data_config.feature_columns[0:13]  # parameter names in feature columns
        self.r_key          = data_config.feature_columns[13]    # r key in feature columns, in case r is scaled differently than other features. e.g. "r", "log10r"
        self.xi_key         = data_config.label_columns[0]      # xi key in label columns

        with h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r') as fff:
            self.simulation_keys = [key for key in fff.keys() if key.startswith("AbacusSummit")]
            self.N_simulations   = len(self.simulation_keys)

        # The keys of the config file to be printed during plotting
        # Makes comparison of different versions easier by seeing which parameters correspond to which errors
        # For e.g. "learning_rate", "patience", the values of these parameters are printed during plotting for each version 
        self.config_param_names     = [print_config_param] if type(print_config_param) != list and print_config_param is not None else print_config_param


    def get_config_parameter(
            self, 
            version: int,
            ) -> dict:
        """
        Make dictionary of config.yaml parameters in self.config_param_names
            key: parameter name
            value: parameter value
        """
        vv_config           = yaml.safe_load(open(f"{self.emul_dir}/version_{version}/config.yaml", "r"))
        vv_config_flattened = pd.json_normalize(vv_config).to_dict(orient="records")[0]
        vv_config_all       = {k.split(".")[1]: v for k, v in vv_config_flattened.items()}
        vv_config_output    = {k: v for k, v in vv_config_all.items() if k in self.config_param_names}
        return vv_config_output

            

    def print_config_parameters(self, version=None, verbose=True):
        assert not isinstance(version, str), "Argument 'version' can not be a string"

        # Ensure version to be an iterable
        if version is None:
            version_list = range(self.N_versions)
        elif not isinstance(version, Iterable):
            # Version is not an iterable, convert to list 
            version_list = [version] 
        else:
            # Version is already an iterable
            version_list = version

        for ver in version_list:
            vv_config_output = self.get_config_parameter(ver)
            if verbose:
                # Don't want to print version number when we're printing errors
                print()
                print(f"Version {ver}:")
        
            for k, v in vv_config_output.items():
                print(f" - {k}={v}")



    def save_tpcf_errors(
            self, 
            plot_versions:          Union[List[int], range, str] = "all",
            r_error_mask:           bool            = True,
            max_r_error:            float           = 60.0,
            min_r_error:            Optional[float] = None, # Not implemented yet,
            ):
        
        flag = self.flag 
        
        if type(plot_versions) == list or type(plot_versions) == range:
            version_list = plot_versions
        else:
            version_list = range(self.N_versions)

        fff         = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')
        t0_tot      = time.time()
        dur_vv_list = []

        # Compute errors for each version, save to file
        print(f"Saving errors for {self.emul_dir.name}")
        for vv in version_list:

            versionpath = Path(f'{self.emul_dir}/version_{vv}')
            file = Path(versionpath / f'{flag}_errors.txt')
            if file.exists():
                print(f"File ../{'/'.join(file.parts[-2:])} already exists. Skipping...")
                continue

            print(f'Saving errors for version {vv}')

            t0_vv = time.time()

            # Load emulator for this version
            _emulator       = cm_emulator_class(version=vv, LIGHTING_LOGS_PATH=self.emul_dir)

            # Lists to store errors for each cosmology
            _err_lst_version = []
            _sim_lst_version = []
            for simulation_key in self.simulation_keys:
                # Use "c___ph___" as row name for the errors 
                s_ = simulation_key.split("_")
                _sim_lst_version.append(f"{s_[2]}_{s_[3]}")

                fff_cosmo = fff[simulation_key]
                _err_lst_cosmo = []
                for params in fff_cosmo.keys():

                    fff_cosmo_HOD = fff_cosmo[params]

                    r_data = fff_cosmo_HOD[self.r_key][...]
                    r_mask          = r_data < max_r_error if r_error_mask else np.ones_like(r_data, dtype=bool)
                    r_data          = r_data[r_mask]

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data))
                            , r_data
                            ))
                    xi_data         = fff_cosmo_HOD[self.xi_key][...][r_mask]
                    xi_emul         = _emulator(params_batch)

                    rel_err         = np.abs(xi_emul / xi_data - 1)
                    _err_lst_cosmo.append(rel_err)
                    

                _err_lst_version.append(_err_lst_cosmo)

            # Compute mean, median, stddev for each individual cosmology            
            err_all          = np.array(_err_lst_version)
            err_mean_cosmo   = np.mean(err_all, axis=(1,2))
            err_median_cosmo = np.median(err_all, axis=(1,2))
            err_stddev_cosmo = np.std(err_all, axis=(1,2))

            # Compute mean, median, stddev across all cosmologies
            err_mean    = np.mean(_err_lst_version)
            err_median  = np.median(_err_lst_version)
            err_stddev  = np.std(_err_lst_version)

            dur_vv = time.time() - t0_vv
            dur_vv_list.append(dur_vv)
            print(f"Done. Took {dur_vv:.2f} sec")

            # Make list of config parameters used for this version 
            config_params_dict = self.get_config_parameter(vv)
            config_params = []
            for k, v in config_params_dict.items():
                config_params.append(f"{k}={v}")

            # Save errors to file
            print(f'creating file: {file}')
            with open(file, 'w') as f:
                f.write(f"#RESULTS INDIVIDUAL COSMOLOGIES:\n")
                f.write("#Simulation | MeanError | MedianError | ErrorStdDev\n")
                for i in range(len(err_mean_cosmo)):
                    f.write(f" {_sim_lst_version[i]:10} , ")
                    f.write(f"{err_mean_cosmo[i]:9.4f} , ")
                    f.write(f"{err_median_cosmo[i]:11.4f} , ")
                    f.write(f"{err_stddev_cosmo[i]:11.4f}\n")

                f.write('\n')
                f.write("#####################################")
                f.write('\n\n')
                f.write("#RESULTS ALL VERSIONS\n")
                f.write(f'#MeanError | MedianError | ErrorStdDev | ConfigParam: \n')
                f.write(f'{err_mean:9.5f} , {err_median:11.5f} , {err_stddev:11.5f} , {config_params} \n')
                f.close()
            
            print()
           

        fff.close()
        print("All versions done.")

        if len(dur_vv_list) > 1:
            dur_tot     = time.time() - t0_tot
            dur_vv_avg  = np.mean(dur_vv_list)
            
            print(f"TOT duration: {dur_tot:.2f} s")
            print(f"Version avg.: {dur_vv_avg:.2f} s")


    def print_tpcf_errors(
            self, 
            plot_versions:          Union[List[int], range, str] = "all",
            print_individual:       bool = False,
            errors_only:            bool = False,
            ):
        
        flag = self.flag 
        

        if type(plot_versions) == list or type(plot_versions) == range:
            version_list = plot_versions
        else:
            version_list = range(self.N_versions)

        print(50*"=")
        for vv in version_list:
            version_path    = Path(self.emul_dir / f'version_{vv}')
            error_file      = Path(version_path / f'{flag}_errors.txt')
            if not error_file.exists():
                error_file      = Path(version_path / f'errors.txt')
                if not error_file.exists():
                    raise FileNotFoundError(
                        f'Error: File {error_file} does not exist. Run save_tpcf_errors to save errors first.'
                        )

            tot_errors = np.loadtxt(error_file, delimiter=',', usecols=[0,1,2], skiprows=self.N_simulations+2, max_rows=1)
            """ 
            Display errors and relevant config parameters for each version
            """
            print(f"VERSION {vv}:")
            print()
            if print_individual:
                sim_versions = np.loadtxt(error_file, delimiter=',', usecols=0, max_rows=self.N_simulations, dtype=str)
                sim_errors   = np.loadtxt(error_file, delimiter=',', usecols=[1,2,3], max_rows=self.N_simulations)
                print(f"INDIVIDUAL SIMULATION ERRORS:")
                print("Simulation   | MeanError | MedianError | ErrorStdDev")
                for i in range(len(sim_errors)):
                    print(f"{sim_versions[i]:10}", end=" | ")
                    print(f"{sim_errors[i,0]:9.4f}", end=" | ")
                    print(f"{sim_errors[i,1]:11.4f}", end=" | ")
                    print(f"{sim_errors[i,2]:11.4f}")
            training_dur = np.loadtxt(f"{version_path}/training_duration.txt", dtype=str, delimiter="_")
            print(f"TOTAL ERRORS:")
            print(f" - MEAN:   {tot_errors[0]:.4f}")
            print(f" - MEDIAN: {tot_errors[1]:.4f}")
            print(f" - STD:    {tot_errors[2]:.4f}")
            if not errors_only:
                print("PARAMS:")
                self.print_config_parameters(version=vv, verbose=False)
                print(f"Training duration: - {training_dur}")

            # print()
            print(50*"=")
            print()


    def compute_wp(
            self,
            xi_data:        np.ndarray,
            r_data:         np.ndarray,
            r_perp_min:     float   = 0.5,
            r_perp_max:     float   = 40.0,
            N_perp:         int     = 40,
            r_para_max:     float   = 100.0,
            N_para:         int     = int(1e4),
            lin:            bool    = False,
            linlog:         bool    = False,

    ):
        """
        Computes projected correlation function wp(r_perp) from xi(r)
        given by w_p(r_perp) = 2 * int_0^r_para_max xi(r) dr_para
        with r = sqrt(r_perp^2 + r_para^2). 
        """

        if lin:
            r_perp_binedge  = np.linspace(r_perp_min, r_perp_max, N_perp)
        elif linlog:
            N_perp_frac = r_data[r_data < 5].shape[0] / r_data[r_data > 5].shape[0]
            N_perp_log = int(N_perp * N_perp_frac)
            N_perp_lin = N_perp - N_perp_log
            r_perp_log = np.geomspace(0.5, 5, N_perp_log, endpoint=False)
            r_perp_lin = np.linspace(5, r_perp_max, N_perp_lin)
            r_perp_binedge = np.concatenate((r_perp_log, r_perp_lin))

        else:
            r_perp_binedge  = np.geomspace(r_perp_min, r_perp_max, N_perp)

        r_perp_bins     = (r_perp_binedge[1:] + r_perp_binedge[:-1]) / 2
        pi_upper_lim    = np.sqrt(np.max(r_data.reshape(-1,1)**2 - r_perp_bins.reshape(1,-1)**2))
        pi_max          = np.min([pi_upper_lim, r_para_max]) #- 10
        r_para          = np.linspace(0, pi_max, N_para)

        # Callable func to interpolate xi(r) 
        xiR_func        = ius(
            r_data, 
            xi_data,
            )

        wp = 2.0 * simpson(
            xiR_func(np.sqrt(r_perp_bins.reshape(-1, 1)**2 + r_para.reshape(1, -1)**2)), 
            r_para, 
            axis=-1,
            )
      
        wp[wp < 0] = 0
        return r_perp_bins, wp



    def plot_proj_corrfunc(
            self, 
            plot_versions:          Union[List[int], range, str] = "all",
            max_r_error:            float   = 60.0,
            nodes_per_simulation:   int     = 1,
            masked_r:               bool    = False,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
     
        flag = self.flag 
        np.random.seed(42)
        

        # if masked_r:
        #     r_mask  = self.r_common < max_r_error
        # else:
        #     r_mask  = np.ones_like(self.r_common, dtype=bool)

        # r_common    = self.r_common[r_mask]
        # xi_fiducial = self.xi_fiducial[r_mask]
        # r_len       = len(r_common)

        fff         = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')

        if type(plot_versions) == list or type(plot_versions) == range:
            version_list = plot_versions
        elif type(plot_versions) == int:
            version_list = [plot_versions]
        else:
            version_list = range(self.N_versions)

        for vv in version_list:
            print(f"Plotting version {vv}")
            fig, ax = plt.subplots(figsize=(10, 9))
            plt.rc('axes', prop_cycle=custom_cycler)
            ax.set_prop_cycle(custom_cycler)

            # Load emulator for this version
            _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.emul_dir)


            for simulation_key in self.simulation_keys:
                
                fff_cosmo = fff[simulation_key]
                nodes_idx = np.random.randint(0, len(fff_cosmo.keys()), nodes_per_simulation)


                for jj in nodes_idx:
                    fff_cosmo_HOD = fff_cosmo[f"node{jj}"]

                    r_data = fff_cosmo_HOD[self.r_key][...]
                    r_mask = r_data < max_r_error if masked_r else np.ones_like(r_data, dtype=bool)
                    r_data = r_data[r_mask]

                    xi_data = fff_cosmo_HOD[self.xi_key][...][r_mask] 

                    ppp = [fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]
                    print(ppp)
                    exit()

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data)
                            )
                            , r_data
                            ))
                    
                    print(params_batch.shape)
                    exit()

                    xi_emul         = _emulator(params_batch) 
                    rp_data, wp_data = self.compute_wp(xi_data, r_data, r_perp_min=0.5)
                    rp_emul, wp_emul = self.compute_wp(xi_emul, r_data, r_perp_min=0.5)

                    ax.plot(rp_data, rp_data * wp_data,     linewidth=0, marker="o", ls="solid",  markersize=2, alpha=1)
                    ax.plot(rp_emul, rp_emul * wp_emul, linewidth=1, alpha=1, label=f"{simulation_key.split('_')[2]}_node{jj}")



            ax.set_xscale("log")
            ax.set_yscale("log")

            ylabel =  r"$r_\bot w_p(r_\bot)\:[h^{-2}\,\mathrm{Mpc}^{2}]$"

            ax.set_xlabel(r'$\displaystyle  r_\bot \: [h^{-1} \mathrm{Mpc}]$',fontsize=18)
            ax.set_ylabel(ylabel,fontsize=22)

            plot_title = f"Version {vv}. {dataset_names[flag]} data \n"
            plot_title += rf"Showing {nodes_per_simulation} sets of $\vec{{\mathcal{{G}}_i}}$ for all {self.N_simulations} sets of $\vec{{\mathcal{{C}}_j}}$"
            ax.set_title(plot_title)

            ax.plot([], linewidth=0, marker='o', color='k', markersize=2, alpha=0.5, label="data")
            ax.plot([], linewidth=1, color='k', alpha=1, label="emulator")
            ax.legend(loc="upper right", fontsize=12)

            if not SAVEFIG:
                plt.show()
            
            else:

                if PRESENTATION:
                    fig_dir_list = list(self.fig_dir.parts)
                    fig_dir_list.insert(1, "presentation")
                    figdir = Path("").joinpath(*fig_dir_list)
                else:
                    figdir = self.fig_dir
            
                figdir.mkdir(parents=True, exist_ok=True)
                
                figtitle = f"version{vv}_wp"
                if masked_r:
                    figtitle += f"_r_max{max_r_error:.0f}"
                if PRESENTATION:
                    week_number = datetime.now().strftime("%U")
                    figtitle = f"week{int(week_number)}_{figtitle}"


                figtitle += ".png"
                figname = Path(figdir / figtitle)
                

                plt.savefig(
                    figname,
                    dpi=200 if figtitle.endswith(".png") else None,
                    bbox_inches="tight",
                    pad_inches=0.05,        
                )
                print(f'save plot to {figname}')
                plt.close(fig)
                if PUSH:
                    os.system(f'git add {figname}')
                    os.system(f'git commit -m "add plot {figname}"')
                    os.system('git push')

        fff.close()


    def plot_tpcf(
            self, 
            plot_versions:          Union[List[int], range, str] = "all",
            max_r_error:            float   = 60.0,
            nodes_per_simulation:   int     = 1,
            masked_r:               bool    = True,
            legend:                 bool    = False,
            r_power:                float   = 0.0,
            setaxinfo:              bool    = True,
            plot_title:                  str     = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
        flag = self.flag 
        np.random.seed(42)
        
        fff   = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')

        if type(plot_versions) == list or type(plot_versions) == range:
            version_list = plot_versions
        elif type(plot_versions) == int:
            version_list = [plot_versions]
        else:
            version_list = range(self.N_versions)

        for vv in version_list:
            print(f"Plotting version {vv}")
            fig = plt.figure(figsize=(10, 9))
            gs = gridspec.GridSpec(2, 1, hspace=0, height_ratios=[1.5, 1])
            plt.rc('axes', prop_cycle=custom_cycler)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax0.set_prop_cycle(custom_cycler)

            # Load emulator for this version
            _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.emul_dir)

            for simulation_key in self.simulation_keys:
                
                fff_cosmo = fff[simulation_key]
                nodes_idx = np.random.randint(0, len(fff_cosmo.keys()), nodes_per_simulation)


                for jj in nodes_idx:
                    fff_cosmo_HOD = fff_cosmo[f"node{jj}"]

                    r_data  = fff_cosmo_HOD[self.r_key][...]
                    r_mask = r_data < max_r_error if masked_r else np.ones_like(r_data, dtype=bool)

                    r_data  = r_data[r_mask]
                    xi_data = fff_cosmo_HOD[self.xi_key][...][r_mask]

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data))
                            , r_data
                            ))

                    xi_emul         = _emulator(params_batch)
                    

                    rel_err         = np.abs(xi_emul / xi_data - 1)

                    y_data = xi_data * r_data**r_power
                    y_emul = xi_emul * r_data**r_power


                    ax0.plot(r_data, y_data, linewidth=0, marker='o', markersize=1, alpha=1)
                    ax0.plot(r_data, y_emul, linewidth=1, alpha=1, label=f"{simulation_key.split('_')[2]}_node{jj}")
                    ax1.plot(r_data, rel_err, color="gray", linewidth=0.7, alpha=0.5)
                
            
            for i in range(1, 4):
                ax1.plot(
                    r_data,
                    np.ones_like(r_data) * 10**(-i),
                    linewidth=0.8,
                    linestyle="--",
                    color='gray',
                    zorder=100,
                )

            if not setaxinfo:
                ax0.set_xscale("log")
                ax0.set_yscale("log")

                ax1.set_xscale("log")
                ax1.set_yscale("log")

                plt.show()
            
            else:        
                if r_power <= 1:
                    if masked_r:
                        ax0.set_ylim([1e-2, 1e4])
                    else:
                        ax0.set_ylim([1e-3, 1e4])
                elif r_power==1.5:
                    ax0.set_ylim([3e0, 2.5e3])
                elif r_power==2:
                    ax0.set_ylim([1e1, 3e3])

                ax1.set_ylim([1e-4, 1e0])

                if r_power == 0:
                    ax0.set_ylabel(r"$\xi_{gg}(r)$",fontsize=22)
                elif r_power == 1:
                    ax0.set_ylabel(r"$r \xi_{gg}(r)$",fontsize=22)

                else:
                    ax0.set_ylabel(rf"$r^{{{r_power}}}\xi_{{gg}}(r)$",fontsize=22)
                ax1.set_xlabel(r'$\displaystyle  r \:  [h^{-1} \mathrm{Mpc}]$',fontsize=18)
                ax1.set_ylabel(r'$\displaystyle \left|\frac{\xi_{gg}^\mathrm{pred} - \xi_{gg}^\mathrm{N-body}}{\xi_{gg}^\mathrm{pred}}\right|$',fontsize=15)


                ax0.xaxis.set_ticklabels([])
                ax0.set_xscale("log")
                ax0.set_yscale("log")
                ax1.set_yscale("log")
                ax1.set_xscale("log")
                if plot_title is None:
                    plot_title = f"Version {vv}. {dataset_names[flag]} data \n"
                    plot_title += rf"Showing {nodes_per_simulation} sets of $\vec{{\mathcal{{G}}_i}}$ for each of the {self.N_simulations} sets of $\vec{{\mathcal{{C}}_j}}$"
                ax0.set_title(plot_title)

                ax0.plot([], linewidth=0, marker='o', color='k', markersize=2, alpha=0.5, label="data")
                ax0.plot([], linewidth=1, color='k', alpha=1, label="emulator")
                if legend:
                    ax0.legend(loc="upper right", fontsize=12)

                if not SAVEFIG:
                    if masked_r:
                        ax0.plot(r_data, np.ones_like(r_data), lw=0)
                    plt.show()
                
                else:
                

                    if PRESENTATION:
                        fig_dir_list = list(self.fig_dir.parts)
                        fig_dir_list.insert(1, "presentation")
                        figdir = Path("").joinpath(*fig_dir_list)
                    else:
                        figdir = self.fig_dir
                
                    figdir.mkdir(parents=True, exist_ok=True)
                    
                    figtitle = f'version{vv}_xi'
                    if masked_r:
                        figtitle += f"_r_max{max_r_error:.0f}"

                    figtitle += f"_{nodes_per_simulation}nodes"

                    if PRESENTATION:
                        week_number = datetime.now().strftime("%U")
                        figtitle = f"week{int(week_number)}_{figtitle}"


                    figtitle += ".png"
                    figname = Path(figdir / figtitle)

                    plt.savefig(
                        figname,
                        dpi=200 if figtitle.endswith(".png") else None,
                        bbox_inches="tight",
                        pad_inches=0.05,        
                    )
                    print(f'save plot to {figname}')
                    plt.close(fig)
                    if PUSH:
                        os.system(f'git add {figname}')
                        os.system(f'git commit -m "add plot {figname}"')
                        os.system('git push')

        fff.close()



S = emulator_test(
    root_dir="./emulator_data",
    dataset="vary_r",
    emul_dir="compare_scaling",
    flag="val",
    print_config_param=["feature_scaler", "label_scaler"],
)
"""
compare_scaling. versions worth looking further into:
v1: id  | log10 , 98 min
v3: id  | stdlog, 575 min 
v4: std | log10 , 306 min 
v6: std | stdlog, 174 min
v7: std | id    , 166 min

Bad:
v0: id  | id, 861 min (bad)
v2: id  | std, 127 min (worse)
v5: std | std, 176 min (worst) 
"""
# S.plot_tpcf()
# S.save_tpcf_errors()
# S.print_tpcf_errors()

S.plot_proj_corrfunc(plot_versions=6)