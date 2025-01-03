import numpy as np
from pathlib import Path
from datetime import datetime 
import pandas as pd
import h5py
import os 
from typing import List, Optional, Union
from collections.abc import Iterable
import sys 
sys.path.append("Path/to/EmulationUtilities")
from _predict import Predictor

import matplotlib.pyplot as plt
from matplotlib import gridspec
from _plot import set_matplotlib_settings, get_CustomCycler
set_matplotlib_settings()
custom_cycler = get_CustomCycler()

import warnings 
warnings.filterwarnings("ignore", category=UserWarning, message="Input line")

global SAVEFIG 
SAVEFIG         = False

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
    ):
        return self.predictor(np.array(params)).reshape(-1)


class TPCF_emulator:
    def __init__(self,
                root_dir:  str = "./emulator_data",
                dataset:   str = None,
                emul_dir:  str = "first_test",
                flag:      str = "val",
                print_config_param:     List[str] = None,
                ):
        if dataset is None:
            self.dataset    = "full"
            self.data_dir   = Path(f"{root_dir}")
        else:
            self.dataset    = dataset
            self.data_dir   = Path(f"{root_dir}/{dataset}")
        self.emul_dir   = Path(self.data_dir / "emulators" / emul_dir) # name of emulator logs directory
        self.fig_dir    = Path(f"./plots/{dataset}/{emul_dir}") # name of emulator plots directory 
        if not self.emul_dir.exists():
            raise FileNotFoundError(f"Path {self.emul_dir} does not exist.")

        self.flag           = flag # data set to be plotted 
        self.N_versions     = len(sorted(self.emul_dir.glob("version_*"))) # number of versions of the emulator
        """
        Get the name of all features and labels from the config file.
        For each emulator version, they are identical, so we get it from version_0.
        This config file should not be used for anything else!
        Elsewhere, the config file for each version should be used. 
        """
        # self.config         = yaml.safe_load(open(self.emul_dir/ "version_0/config.yaml", "r"))
        config_file         = Predictor.load_config(self.emul_dir / "version_0")
        config_keys         = [subkey for key in config_file.keys() for subkey in config_file[key].keys()]

        feature_columns    = config_file["data"]["feature_columns"]  # parameter names in feature columns
        self.param_names    = [param for param in feature_columns if param != "r"]
        self.r_key          = "r"
        self.xi_key         = "xi"      # xi key in label columns

        with h5py.File(self.data_dir / f"TPCF_{flag}.hdf5", 'r') as fff:
            self.simulation_keys = [key for key in fff.keys() if key.startswith("AbacusSummit")]
            self.N_simulations   = len(self.simulation_keys)
            self.N_bins_per_node = fff[self.simulation_keys[0]]["node0"][self.r_key].shape[0]
        self.N_nodes_per_simulation = {
            "test": 100,
            "val": 100,
            "train": 500,
        }
        

        # The keys of the config file to be printed during plotting
        # Makes comparison of different versions easier by seeing which parameters correspond to which errors
        # For e.g. "learning_rate", "patience", the values of these parameters are printed during plotting for each version 
        self.config_param_names     = [print_config_param] if type(print_config_param) != list and print_config_param is not None else print_config_param
        for k in self.config_param_names:
            if k not in config_keys:
                raise KeyError(f"Key {k} not found in config file.")
            

    def get_config_parameter(
            self, 
            version: int,
            ) -> dict:
        """
        Make dictionary of config.yaml parameters in self.config_param_names
            key: parameter name
            value: parameter value
        """
        vv_config = Predictor.load_config(self.emul_dir / f"version_{version}")
        vv_config_flattened = pd.json_normalize(vv_config).to_dict(orient="records")[0]
        vv_config_all       = {k.split(".")[1]: v for k, v in vv_config_flattened.items()}
        vv_config_output    = {k: v for k, v in vv_config_all.items() if k in self.config_param_names}
        return vv_config_output

            

    def print_config_parameters(self, version=None, print_version_number=True):
        """
        Print relevant config parameters for each version
        Simplifies comparison of different versions
        """
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
        if print_version_number:
            print(f"Emulators in {'/'.join(self.emul_dir.parts[-3:])}:")

        for ver in version_list:
            vv_config_output = self.get_config_parameter(ver)
            if print_version_number:
                # Don't want to print version number when we're printing errors
                # print()
                print(f" v{ver}:")
        
            for k, v in vv_config_output.items():
                print(f" - {k}={v}")



    def save_tpcf_errors(
            self, 
            versions:          Union[List[int], range, str] = "all",
            max_r_error:            float           = np.inf,
            min_r_error:            Optional[float] = 0.0, 
            overwrite:              bool            = False,
            ):
        
        flag = self.flag 
        
        if type(versions) == list or type(versions) == range:
            version_list = versions
        elif type(versions) == int:
            version_list = [versions]
        else:
            version_list = range(self.N_versions)

        fff         = h5py.File(self.data_dir / f"TPCF_{flag}.hdf5", 'r')
        # Compute errors for each version, save to file
        for vv in version_list:

            versionpath = Path(f'{self.emul_dir}/version_{vv}')
            # Add suffix to file name with min_r_error and max_r_error information
            if min_r_error == 0.0 and max_r_error == np.inf:
                suffix = "full"
            else:
                suffix = "sliced"
            
            fname = f'{flag}_errors_{suffix}.txt'
            file = Path(versionpath /fname )
            if file.exists() and not overwrite:
                continue
                
            print(f'Saving errors for version {vv} in {self.emul_dir.name}')

            # Load emulator for this version
            _emulator       = cm_emulator_class(version=vv, LIGHTING_LOGS_PATH=self.emul_dir)

            # Lists to store errors for each cosmology
            _err_lst_version = []
            _sim_lst_version = []
            err_mean_cosmo   = []
            err_median_cosmo = []
            err_stddev_cosmo = []
            for simulation_key in self.simulation_keys:
                # Use "c___ph___" as row name for the errors 
                s_ = simulation_key.split("_")
                _sim_lst_version.append(f"{s_[2]}_{s_[3]}")

                fff_cosmo = fff[simulation_key]
                _err_lst_cosmo = []

                for node in range(len(fff_cosmo.keys())):
                    fff_cosmo_HOD = fff_cosmo[f"node{node}"]

                    r_data          = fff_cosmo_HOD[self.r_key][...]
                    r_mask          = (r_data > min_r_error) & (r_data < max_r_error)
                    r_data          = r_data[r_mask]

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data))
                            , r_data
                            ))
                    xi_data         = fff_cosmo_HOD[self.xi_key][...][r_mask]
                    xi_emul         = _emulator(params_batch)

                    rel_err         = np.abs(xi_emul / xi_data - 1)
                    _err_lst_cosmo.extend(rel_err)

                _err_lst_version.extend(_err_lst_cosmo)
                err_mean_cosmo.append(np.mean(_err_lst_cosmo))
                err_median_cosmo.append(np.median(_err_lst_cosmo))
                err_stddev_cosmo.append(np.std(_err_lst_cosmo))
      
            # Compute mean, median, stddev across all cosmologies
            err_mean    = np.mean(_err_lst_version)
            err_median  = np.median(_err_lst_version)
            err_stddev  = np.std(_err_lst_version)
      

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
                f.write("#r_min, r_max: \n")
                f.write(f"{min_r_error}, {max_r_error}\n")
                f.close()
            
        fff.close()

    def print_tpcf_errors(
            self, 
            versions:          Union[List[int], range, str] = "all",
            print_individual:       bool = True,
            print_params:           bool = True,
            min_r_error:            float = 0.0,
            max_r_error:            float = np.inf,
            overwrite:              bool = False,
            ):
        
        flag = self.flag 
        

        if type(versions) == list or type(versions) == range:
            version_list = versions
        elif type(versions) == int:
            version_list = [versions]
        else:
            version_list = range(self.N_versions)

        # print(52*"=")
        if overwrite:
            self.save_tpcf_errors(versions=version_list, max_r_error=max_r_error, min_r_error=min_r_error, overwrite=overwrite)
        print(" ", "#"*48)
        print(f"  # {flag} errors: {self.dataset}/{self.emul_dir.name} ")
        print(" ", "#"*48)

        if min_r_error == 0.0 and max_r_error == np.inf:
            suffix = "full"
        else:
            suffix = "sliced"
        
        fname = f'{flag}_errors_{suffix}.txt'

        for vv in version_list:
            version_path    = Path(self.emul_dir / f'version_{vv}')
            error_file      = Path(version_path / fname)
            if not error_file.exists():
                self.save_tpcf_errors(versions=vv, max_r_error=max_r_error, min_r_error=min_r_error)

            tot_errors   = np.loadtxt(error_file, delimiter=',', usecols=[0,1,2], skiprows=self.N_simulations+2, max_rows=1)
            r_min, r_max = np.loadtxt(error_file, delimiter=',', usecols=[0,1], skiprows=self.N_simulations+2)[-1]#, max_rows=1)
            """ 
            Display errors and relevant config parameters for each version
            """
            print(f"Version{vv}     |           {r_min} < r < {r_max}:")
            print(52*"=")
            print("Simulation   | MeanError | MedianError | ErrorStdDev")
            if print_individual:
                print(52*"-")
                sim_versions = np.loadtxt(error_file, delimiter=',', usecols=0, max_rows=self.N_simulations, dtype=str)
                sim_errors   = np.loadtxt(error_file, delimiter=',', usecols=[1,2,3], max_rows=self.N_simulations)
                for i in range(len(sim_errors)):
                    print(f"{sim_versions[i]:10}", end=" | ")
                    print(f"{sim_errors[i,0]:9.4f}", end=" | ")
                    print(f"{sim_errors[i,1]:11.4f}", end=" | ")
                    print(f"{sim_errors[i,2]:11.4f}")
            training_dur = np.loadtxt(f"{version_path}/training_duration.txt", dtype=str, delimiter="_")
            print(52*"-")

            print(f" ALL         ", end="| ")
            print(f"{tot_errors[0]:9.4f}", end=" | ")
            print(f"{tot_errors[1]:11.4f}", end=" | ")
            print(f"{tot_errors[2]:11.4f}")

            if print_params:
                print(52*"-")
                print("PARAMS:")
                self.print_config_parameters(version=vv, print_version_number=False)
                print(f"Training duration:")
                print(f" - {training_dur}")

            print("\n")

    def get_rel_err_all(
            self, 
            version:                int,
            percentile:             float = 68,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
        flag = self.flag 
        fff   = h5py.File(self.data_dir / f"TPCF_{flag}.hdf5", 'r')

        outfname_stem   = f"./rel_errors/v{version}_{flag}_xi"
        statistics      = ["mean", "median", "stddev", f"{percentile}percentile"]
        fnames          = {
            stat: Path(f"{outfname_stem}_{stat}.npy") for stat in statistics if not Path(f"{outfname_stem}_{stat}.npy").exists()
        }
        # Check if fnames is empty
        if not fnames:
            print("All files exist. Exiting.")
            return
        print(f"Saving {[k for k in fnames.keys()]} for v{version}")

        # Load emulator for this version
        _emulator       = cm_emulator_class(version=version,LIGHTING_LOGS_PATH=self.emul_dir)
        rel_err_arr_ = np.zeros((self.N_simulations, self.N_nodes_per_simulation[flag], self.N_bins_per_node))
        for ii, simulation_key in enumerate(self.simulation_keys):
            fff_cosmo = fff[simulation_key]

            for jj in range(self.N_nodes_per_simulation[flag]):
                fff_cosmo_HOD = fff_cosmo[f"node{jj}"]

                r_data  = fff_cosmo_HOD[self.r_key][...]

                params_batch   = np.column_stack(
                    (np.vstack(
                        [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data))
                        , r_data
                        ))

                xi_data                 = fff_cosmo_HOD[self.xi_key][...]
                xi_emul                 = _emulator(params_batch)
                rel_err_arr_[ii, jj, :] = np.abs(xi_emul / xi_data - 1)
               
        # Compute mean, median, stddev as a function of r
        rel_err_statistics = {
            "mean":                     np.mean(rel_err_arr_, axis=(0,1)),
            "median":                   np.median(rel_err_arr_, axis=(0,1)),
            "stddev":                   np.std(rel_err_arr_, axis=(0,1)),
            f"{percentile}percentile":  np.percentile(rel_err_arr_, percentile, axis=(0,1)),
        }
        for key in fnames.keys():
            print(f"Saving {fnames[key]}")
            np.save(fnames[key], rel_err_statistics[key])

    def plot_tpcf(
            self, 
            versions:          Union[List[int], range, str] = "all",
            max_r_error:            float   = np.inf,
            min_r_error:            float   = 0.0,
            nodes_per_simulation:   int     = 1,
            legend:                 bool    = True,
            r_power:                float   = 0.0,
            setaxinfo:              bool    = True,
            outfig:                 str     = None,
            percentile:             float   = 68,
            rel_err_statistics:     bool    = False,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
        if type(versions) == list or type(versions) == range:
            version_list = versions
        elif type(versions) == int:
            version_list = [versions]
        else:
            version_list = range(self.N_versions)

        if min_r_error == 0.0 and max_r_error == np.inf:
            masked_r = False
        else:
            masked_r = True

        flag = self.flag 
        fff   = h5py.File(self.data_dir / f"TPCF_{flag}.hdf5", 'r')
        np.random.seed(42)
        available_nodes = np.arange(self.N_nodes_per_simulation[flag])
        
        nodes_idx = {
            simulation_key: np.random.choice(available_nodes, nodes_per_simulation, replace=False) for simulation_key in self.simulation_keys
        }

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
            rel_err_lst_ = []
            r_data_lst_  = []
            for simulation_key in self.simulation_keys:
                fff_cosmo = fff[simulation_key]


                for jj in nodes_idx[simulation_key]:
                    fff_cosmo_HOD = fff_cosmo[f"node{jj}"]

                    r_data  = fff_cosmo_HOD[self.r_key][...]
                    r_mask = (r_data > min_r_error) & (r_data < max_r_error)
                    r_data  = r_data[r_mask]



                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data))
                            , r_data
                            ))

                    xi_data = fff_cosmo_HOD[self.xi_key][...][r_mask]
                    xi_emul         = _emulator(params_batch)
                    rel_err         = np.abs(xi_emul / xi_data - 1)
                    rel_err_lst_.append(rel_err)
                    r_data_lst_.append(r_data)

                    y_data = xi_data * r_data**r_power
                    y_emul = xi_emul * r_data**r_power


                    ax0.plot(r_data, y_data , linewidth=0,   alpha=1, marker='o', markersize=1.5)
                    ax0.plot(r_data, y_emul , linewidth=1,   alpha=1)
                    ax1.plot(r_data, rel_err, linewidth=0.7, alpha=0.5, color="gray")
            for i in range(1, 4):
                ax1.plot(
                    r_data,
                    np.ones_like(r_data) * 10**(-i),
                    linewidth=0.8,
                    linestyle="--",
                    color='gray',
                    zorder=100,
                )
            if rel_err_statistics:
                
                # All nodes have the same number of r-values
                rel_err_mean        = np.load(f"./rel_errors/v{vv}_{flag}_xi_mean.npy")[r_mask]
                rel_err_median      = np.load(f"./rel_errors/v{vv}_{flag}_xi_median.npy")[r_mask]
                rel_err_stddev      = np.load(f"./rel_errors/v{vv}_{flag}_xi_stddev.npy")[r_mask]
                rel_err_percentile  = np.load(f"./rel_errors/v{vv}_{flag}_xi_{percentile}percentile.npy")[r_mask]
            
                # Plot shaded region for standard deviation
                # ax1.fill_between(r_data, rel_err_mean - rel_err_stddev, rel_err_mean + rel_err_stddev, alpha=0.1, color='red', zorder=0)
                ax1.plot(r_data, rel_err_mean, linewidth=1, color='green', label="Mean")
                ax1.plot(r_data, rel_err_median, linewidth=1, color='blue', label="Median")
                # ax1.plot(r_data, rel_err_perc, linewidth=1, color='red', label=f"{percentile}th percentile")


            if not setaxinfo and outfig is None:
                ax0.set_xscale("log")
                ax0.set_yscale("log")

                ax1.set_xscale("log")
                ax1.set_yscale("log")

                plt.show()
            
            else:        
                if r_power <= 1:
                    pass
                    # if masked_r:
                    #     ax0.set_ylim([1.5e-3, 1e5])
                    # else:
                    #     ax0.set_ylim([1e-3, 5e4])
                # elif r_power==1.5:
                #     ax0.set_ylim([3e0, 2.5e3])
                # elif r_power==2:
                #     ax0.set_ylim([1e1, 3e3])

                ax1.set_ylim([5e-4, 0.9e0])

                if r_power == 0:
                    ax0.set_ylabel(r"$\xi^R(r)$",fontsize=22)
                elif r_power == 1:
                    ax0.set_ylabel(r"$r \xi^R(r)\quad [h^{-1}\mathrm{Mpc}]$",fontsize=22)
                elif r_power == 2:
                    ax0.set_ylabel(r"$r^2 \xi^R(r)\quad [h^{-2}\mathrm{Mpc}^2]$",fontsize=22)
                else:
                    ax0.set_ylabel(rf"$r^{{{r_power}}}\xi_{{gg}}(r)$",fontsize=22)
                ax1.set_xlabel(r'$\displaystyle  r \quad   [h^{-1} \mathrm{Mpc}]$',fontsize=18)
                ax1.set_ylabel(r'$\displaystyle \left|\frac{\xi^R_\mathrm{pred} - \xi^R_\mathrm{data}}{\xi^R_\mathrm{pred}}\right|$',fontsize=15)


                ax0.xaxis.set_ticklabels([])
                ax0.set_xscale("log")
                ax0.set_yscale("log")
                ax1.set_yscale("log")
                ax1.set_xscale("log")
                # if plot_title is None:
                    # plot_title = f"Version {vv}. {dataset_names[flag]} data \n"
                    # plot_title += rf"Showing {nodes_per_simulation} sets of $\vec{{\mathcal{{G}}_i}}$ for each of the {self.N_simulations} sets of $\vec{{\mathcal{{C}}_j}}$"
                # ax0.set_title(plot_title)

                ax0.plot([], linewidth=0, marker='o', color='k', markersize=2, alpha=0.5, label="Data")
                ax0.plot([], linewidth=1, color='k', alpha=1, label="Emulator")
                if legend:
                    ax0.legend(loc="upper right", fontsize=12)
                    ax1.legend(loc="upper left", fontsize=12)

                if not SAVEFIG and outfig is None:
                    if masked_r:
                        ax0.plot(r_data, np.ones_like(r_data), lw=0)
                    fig.tight_layout()
                    plt.show()
                
                else:
                    if outfig is None:
                        figdir = self.fig_dir 
                        figdir.mkdir(parents=True, exist_ok=True)
                    
                        y_title = "xi" if r_power == 0 else f"r_{r_power}_xi"
                        figtitle = f'version{vv}_{y_title}.png'
                        outfig = f"{figdir}/{figtitle}"
                    plt.savefig(
                        Path(outfig),
                        dpi=150 if outfig.endswith(".png") else None,
                        bbox_inches="tight",
                        pad_inches=0.05,        
                    )
                    print(f'save plot to {outfig}')
                    plt.close(fig)


        fff.close()

TPCF_sliced_3040 = TPCF_emulator(
    root_dir            =   "./emulator_data",
    dataset             =   "sliced_r",
    emul_dir            =   "batch_size_3040",
    flag                =   "test",
    print_config_param  =   ["batch_size", "hidden_dims", "stopping_patience"],
)
def plot_xi():
    r_powers = [2]
    for r_power in r_powers:
        outfig = f"plots/thesis_figures/emulators/r_power_{r_power}_xi_{TPCF_sliced_3040.flag}"
        outfig_pdf = f"{outfig}.pdf"
        outfig_png = f"{outfig}.png"
        TPCF_sliced_3040.plot_tpcf(versions=2, r_power=r_power, rel_err_statistics=True, outfig=outfig_pdf)
        TPCF_sliced_3040.plot_tpcf(versions=2, r_power=r_power, rel_err_statistics=True, outfig=outfig_png)
def print_xi_err():
    TPCF_sliced_3040.print_tpcf_errors(versions=2, print_individual=True, print_params=False, overwrite=False)
    TPCF_sliced_3040.print_tpcf_errors(versions=2, print_individual=True, print_params=False, min_r_error=0.1, max_r_error=60, overwrite=False)



# print_xi_err()
plot_xi()
# TPCF_sliced_3040.get_rel_err_all(version=2)
# TPCF_sliced_3040.plot_tpcf(2, rel_err_statistics=True, r_power=2, outfig="plots/thesis_figures/emulators/test.png")

