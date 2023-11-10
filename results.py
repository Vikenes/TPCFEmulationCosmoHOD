import numpy as np
from pathlib import Path
from datetime import datetime 
import pandas as pd
import h5py
import os 
import yaml 
from typing import List, Optional, Union
import time 
import typing 

import sys 
sys.path.append("../emul_utils")
from _nn_config import DataConfig, TrainingConfig, ModelConfig
from _predict import Predictor

import matplotlib.pyplot as plt
from matplotlib import gridspec
from _plot import set_matplotlib_settings, get_CustomCycler
set_matplotlib_settings()
custom_cycler = get_CustomCycler()


global SAVEFIG 
global PUSH
global SAVEERRORS
global TRANSFORM
global PRESENTATION

SAVEFIG         = False
PUSH            = False
SAVEERRORS      = False
TRANSFORM       = False
PRESENTATION    = False



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
        transform_: bool = False,
    ):
        inputs = np.array(params)
        return self.predictor(inputs, transform=transform_).reshape(-1)


class emulator_test:
    def __init__(self,
                root_dir:  str = "./tpcf_data",
                dataset:   str = "xi_over_xi_fiducial",
                emul_dir:  str = "time_test",
                flag:      str = "val",
                print_config_param:     List[str] = None,
                ):
        self.dataset    = dataset
        self.data_dir   = Path(f"{root_dir}/{dataset}")
        self.emul_dir   = Path(self.data_dir / "emulators" / emul_dir) # name of emulator logs directory
        self.fig_dir    = Path(f"./plots/{dataset}/{emul_dir}") # name of emulator plots directory 
        # self.logs_path  = Path(f"./emulators/{dataset}/{emul_dir}") # emulator logs path
        if not self.emul_dir.exists():
            # Check if the emulator logs directory exists
            raise FileNotFoundError(f"Path {self.emul_dir} does not exist.")

        self.tpcf_data    = Path(self.data_dir / f"TPCF_{flag}_ng_fixed.csv") # name of data set
        
        self.flag           = flag # data set to be plotted 
        self.N_versions     = len(sorted(self.emul_dir.glob("version_*"))) # number of versions of the emulator
        version_dirs        = [f"{self.emul_dir}/version_{i}" for i in range(self.N_versions)]
        self.config         = yaml.safe_load(open("config.yaml", "r"))
        data_config         = DataConfig(**self.config["data"])
       
        self.param_names    = data_config.feature_columns[0:13]  # parameter names in feature columns
        self.r_key          = data_config.feature_columns[13]    # r key in feature columns, e.g. "r", "log10r"
        self.xi_key         = data_config.label_columns[0]      # xi key in label columns

        # The keys of the config file to be printed during plotting
        # Makes comparison of different versions easier by seeing which parameters correspond to which errors
        # For e.g. "learning_rate", "patience", the values of these parameters are printed during plotting for each version 
        self.print_config_param     = [print_config_param] if type(print_config_param) != list and print_config_param is not None else print_config_param


    def print_config(self, version):
        """
        Prints the config parameter values for an emulator version
        corresponding to the keys in self.print_config_param
        """
        vv_config           = yaml.safe_load(open(f"{self.emul_dir}/version_{version}/config.yaml", "r"))
        vv_config_flattened = pd.json_normalize(vv_config).to_dict(orient="records")[0]
        vv_config_all       = {k.split(".")[1]:v for k,v in vv_config_flattened.items()}
        vv_config_output = {k:v for k,v in vv_config_all.items() if k in self.print_config_param}


        if self.print_config_param is not None:
            if SAVEERRORS:
                return vv_config_output
            else:
                
                for k,v in vv_config_output.items():
                    print(" - ", end="")
                    print(f"{k}={v}")


    def save_tpcf_errors(
            self, 
            plot_versions:          Union[List[int], range, str] = "all",
            max_r_error:            float = 60.0,
            ):
        
        flag = self.flag 
        
        fff_common   = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')
        r_common     = fff_common["r"][...]
        r_error_mask = r_common < max_r_error
        r_len        = len(r_common)
        fff_common.close()

        if type(plot_versions) == list or type(plot_versions) == range:
            version_list = plot_versions
        else:
            version_list = range(self.N_versions)

        t0_tot = time.time()
        dur_vv_list = []

        for vv in version_list:
            t0_vv = time.time()

            _err_lst_version = []    
            fff = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')
            for iii, simulation_key in enumerate(fff.keys()):
                if not simulation_key.startswith("AbacusSummit"):
                    # Skip the xi_fiducial and r keys
                    continue
                
                fff_cosmo = fff[simulation_key]
                _err_lst_cosmo = []
                for jj, params in enumerate(fff_cosmo.keys()):
                    fff_cosmo_HOD = fff_cosmo[params]

                    xi_data = fff_cosmo_HOD[self.xi_key][...]

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * r_len)
                            , r_common
                            ))
                    
                    _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.emul_dir)
                    xi_emul         = _emulator(params_batch, transform_=TRANSFORM)

                    rel_err         = np.abs(xi_emul / xi_data - 1)[r_error_mask]
                    _err_lst_cosmo.append(rel_err)

                _err_lst_version.append(_err_lst_cosmo)
            
            fff.close()
            # err_all = np.array(_err_lst_version)
            err_mean    = np.mean(_err_lst_version)
            err_median  = np.median(_err_lst_version)
            err_stddev  = np.std(_err_lst_version)

            dur_vv = time.time() - t0_vv
            dur_vv_list.append(dur_vv)
            print(f" - {dur_vv=:.2f} s")

            #### FIX SAVEERRORS
            if not SAVEERRORS:
                """ 
                Display errors and relevant config parameters for each version
                """
                # print(f'VERSION {vv}:')
                print(f"ALL VERSION {vv}  - {dur_vv=:.2f} s:")
                print(f"TOTAL:")
                print(f" - {err_mean=:.4f}")
                print(f" - {err_median=:.4f}")
                print(f" - {err_stddev=:.4f}")
                print("PARAMS:")
                self.print_config(version=vv)
                self.print_config(version=vv)
                # print(f"zero_bias={vv_model_config.zero_bias}, patience={vv_training_config.patience}")
                print(f'mean error={err_mean:.4f}, median_error={err_median:.4f}, error_stddev={err_stddev:.4f}\n')

            else:
                """
                Save errors to file
                """
                print(f'Saving errors for version {vv}')
                file = f'{self.logs_path}/errors.txt'
                if not os.path.isfile(file):
                    print(f'creating file: {file}')
                    with open(file, 'w') as f:
                        f.write(f'# VERSION, MeanError, MedianError, ErrorStdDev, ConfigParam: \n')
                        f.close()
            
                with open(file, 'a') as f:
                    f.write(f'{vv:7d}, ')
                    f.write(f'{err_mean:10.4f}, {err_median:10.4f}, {err_stddev:10.4f}')
                    if self.print_config_param is not None:
                        # Write relevant config parameters to file
                        config_output = self.print_config(version=vv)
                        for k,v in config_output.items():
                            f.write(f', {k}={v}')

                    f.write('\n')
                    f.close()
                continue




            print()
            print("=====================================")
            print()
            

        dur_tot = time.time() - t0_tot
        dur_vv_avg = np.mean(dur_vv_list)
        print(f"TOTAL TIME: {dur_tot=:.2f} s")
        print(f"Average time per version: {dur_vv_avg=:.2f} s")



    
      

    def plot_tpcf(
            self, 
            plot_versions:          Union[List[int], range, str] = "all",
            max_r_error:            float = 60.0,
            nodes_per_simulation:   int = 1,
            masked_r:               bool = False,
            xi_ratio:               bool = True,
            # plot_every_n:           int = 1,
            ):
        flag = self.flag 
        np.random.seed(42)
        
        fff_common   = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')
        xi_fiducial_ = fff_common["xi_fiducial"][...]
        r_common_    = fff_common["r"][...]
        fff_common.close()

        if masked_r:
            r_mask  = r_common_ < max_r_error
        else:
            r_mask  = np.ones_like(r_common_, dtype=bool)

        r_common    = r_common_[r_mask]
        xi_fiducial = xi_fiducial_[r_mask]
        r_len       = len(r_common)


        if type(plot_versions) == list or type(plot_versions) == range:
            version_list = plot_versions
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

            fff = h5py.File(self.data_dir / f"TPCF_{flag}_ng_fixed.hdf5", 'r')
            N_simulations = len(fff.keys())
            for iii, simulation_key in enumerate(fff.keys()):
                if not simulation_key.startswith("AbacusSummit"):
                    # Skip the xi_fiducial and r keys
                    N_simulations -= 1
                    continue
                
                fff_cosmo = fff[simulation_key]
                nodes_idx = np.random.randint(0, len(fff_cosmo.keys()), nodes_per_simulation)


                for jj in nodes_idx:
                    fff_cosmo_HOD = fff_cosmo[f"node{jj}"]

                    xi_data = fff_cosmo_HOD[self.xi_key][...][r_mask]

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * r_len)
                            , r_common
                            ))

                    _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.emul_dir)
                    xi_emul         = _emulator(params_batch, transform_=TRANSFORM)

                    rel_err         = np.abs(xi_emul / xi_data - 1)

                    if not xi_ratio:
                        xi_data = xi_data * xi_fiducial
                        xi_emul = xi_emul * xi_fiducial

                    ax0.plot(r_common, xi_data, linewidth=0, marker='o', markersize=1, alpha=1)
                    ax0.plot(r_common, xi_emul, linewidth=1, alpha=1)
                    ax1.plot(r_common, rel_err, color="gray", linewidth=0.7, alpha=0.5)


            fff.close()
            
            for i in range(1, 4):
                ax1.plot(r_common,
                    np.ones_like(r_common) * 10**(-i),
                    linewidth=0.8,
                    linestyle="--",
                    color='gray',
                    zorder=100,
                )

            # ax0.set_xlim([-3, 0.1])
            # ax1.set_xlim([-3, 0.1])
            ax0.set_ylim([1e-2, 1e4])
            ax1.set_ylim([1e-4, 1e0])

            if xi_ratio:
                ylabel =  r"$\xi_{gg}(r)/\xi_{gg}(r, \mathcal{C}_\mathrm{fid}, \mathcal{G}_\mathrm{fid})$"
            else:
                ylabel =  r"$\xi_{gg}(r)$"

            ax1.set_xlabel(r'$\displaystyle  r/h \: [\mathrm{Mpc}]$',fontsize=18)
            ax1.set_ylabel(r'$\displaystyle \mathrm{rel. diff.}$',fontsize=20)
            ax0.set_ylabel(ylabel,fontsize=22)

            ax0.xaxis.set_ticklabels([])
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            ax1.set_yscale("log")
            ax1.set_xscale("log")

            ax0.set_title(rf"version {vv}. {N_simulations} different sets of $\displaystyle \mathcal{{C}}$")

            ax0.plot([], linewidth=0, marker='o', color='k', markersize=2, alpha=0.5, label="data")
            ax0.plot([], linewidth=1, color='k', alpha=1, label="emulator")
            ax0.legend(loc="upper right", fontsize=12)

            if not SAVEFIG:
                if masked_r:
                    ax0.plot(r_common_, np.ones_like(r_common_), lw=0)
                plt.show()
                # continue
                exit()
            
            else:
            
                # figdir = f'./plots/{self.emul_dir}'
                

                if PRESENTATION:
                    fig_dir_list = list(self.fig_dir.parts)
                    fig_dir_list.insert(1, "presentation")
                    figdir = Path("").joinpath(*fig_dir_list)
                else:
                    figdir = self.fig_dir
            
                figdir.mkdir(parents=True, exist_ok=True)
                
                figtitle = f'version{vv}'
                if xi_ratio:
                    figtitle += "_xi_ratio"
                else:
                    figtitle += "_xi"
                if masked_r:
                    figtitle += "_masked_r"
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



param_list = ["batch_size", "hidden_dims", "max_epochs", "patience"]
test = emulator_test(
    root_dir="./tpcf_data",
    dataset="xi_over_xi_fiducial",
    emul_dir="time_test",
    flag="val",
    print_config_param=param_list,
)

dropout_test = emulator_test(
    root_dir="./tpcf_data",
    dataset="xi_over_xi_fiducial",
    emul_dir="dropout_test",
    flag="val",
    print_config_param="dropout",
)

# test.plot_tpcf(range(0,3))
# test.save_tpcf_errors()
SAVEFIG = True
PRESENTATION = True
test.plot_tpcf(plot_versions=[7], nodes_per_simulation=3, masked_r=False, xi_ratio=True)
test.plot_tpcf(plot_versions=[7], nodes_per_simulation=3, masked_r=True, xi_ratio=False)

# dropout_test.save_tpcf_errors()
