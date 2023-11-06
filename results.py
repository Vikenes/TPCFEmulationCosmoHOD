import numpy as np
from pathlib import Path
from datetime import datetime 
import pandas as pd
import h5py
import os 
import yaml 
from typing import List, Optional, Union, Tuple


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
                 emul_dir:  str,
                 flag:      str = "val",
                 log10r:    bool = True,
                 log10xi:   bool = True,
                 print_config_param:     List[str] = None,
                 ):
        

        self.emul_dir           = Path(emul_dir) # name of emulator logs directory 
        self.logs_path  = Path(f"./emulators/{emul_dir}") # emulator logs path
        if not self.logs_path.exists():
            # Check if the emulator logs directory exists
            raise FileNotFoundError(f"Path {self.logs_path} does not exist.")

        
        self.flag           = flag # data set to be plotted 
        self.N_versions     = len(sorted(self.logs_path.glob("version_*"))) # number of versions of the emulator
        version_dirs        = [f"{self.logs_path}/version_{i}" for i in range(self.N_versions)]
        self.config         = yaml.safe_load(open("config.yaml", "r"))
        data_config         = DataConfig(**self.config["data"])
       
        self.param_names    = data_config.feature_columns[0:5]  # parameter names in feature columns
        self.r_key          = data_config.feature_columns[5]    # r key in feature columns, e.g. "r", "log10r"
        self.xi_key         = data_config.label_columns[0]      # xi key in label columns
        self.log10r         = log10r
        self.log10xi        = log10xi

        # The keys of the config file to be printed during plotting
        # Makes comparison of different versions easier by seeing which parameters correspond to which errors
        # For e.g. "learning_rate", "patience", the values of these parameters are printed during plotting for each version 
        self.print_config_param     = [print_config_param] if type(print_config_param) != list and print_config_param is not None else print_config_param



    def print_config(self, version):
        """
        Prints the config parameter values for an emulator version
        corresponding to the keys in self.print_config_param
        """
        vv_config           = yaml.safe_load(open(f"{self.logs_path}/version_{version}/config.yaml", "r"))
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

    
      

    def plot_tpcf(
            self, 
            plot_versions:          Union[List[int], int, str] = "all",
            plot_range:             Optional[bool] = True,
            plot_min:               Optional[int] = 0,
            plot_every_n:           int = 2,
            ):
        
        flag = self.flag 
        if plot_versions == "all":
            version_list = range(self.N_versions)
        elif type(plot_versions) == list:
            version_list = plot_versions
        else:
            if plot_range:
                version_list = range(plot_min, plot_versions if type(plot_versions)==int else self.N_versions)
            else:
                version_list = [plot_versions]
        
        for vv in version_list:


            fig = plt.figure(figsize=(10, 9))
            gs = gridspec.GridSpec(2, 1, hspace=0, height_ratios=[1.5, 1])
            plt.rc('axes', prop_cycle=custom_cycler)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax0.set_prop_cycle(custom_cycler)
            _err_lst = []    
            TPCF_DATA_FILENAME = f"TPCF_{flag}.hdf5"
            if self.r_key == "log10r" and self.xi_key == "log10xi":
                TPCF_DATA_FILENAME = f"log_{TPCF_DATA_FILENAME}"

            with h5py.File(f"./tpcf_data/{TPCF_DATA_FILENAME}", 'r') as fff:
                for ii, key_cosmo in enumerate(list(fff.keys())):
                        
                    # Load TPCF data   
                    ggg      = fff[key_cosmo]
                    r_       = ggg[self.r_key][...]
                    xi_data_ = ggg[self.xi_key][...]


                    params_cosmo    = np.array([ggg.attrs[param_name] for param_name in self.param_names])
                    params_batch    = np.column_stack((np.vstack([params_cosmo] * len(r_)), r_))

                    _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.logs_path)
                    xi_emu_         = _emulator(params_batch, transform_=TRANSFORM)

                    # Get linear r and xi if log10r and log10xi are True 
                    r       = 10**r_ if self.log10r else r_
                    xi_data = 10**xi_data_ if self.log10xi else xi_data_
                    xi_emul = 10**xi_emu_ if self.log10xi else xi_emu_

                    rel_err         = np.abs(xi_emul / xi_data - 1)
                    _err_lst.append(rel_err)

                    if (ii+1) % plot_every_n != 0:
                        continue

                    ax0.plot(r, xi_data, linewidth=0, marker='o', markersize=2, alpha=0.5)
                    ax0.plot(r, xi_emul, linewidth=1, alpha=1)
                    ax1.plot(r, rel_err, color="gray", linewidth=0.7, alpha=0.5)



            err_all     = np.array(_err_lst)
            err_mean    = np.mean(err_all)
            err_median  = np.median(err_all)
            err_stddev  = np.std(err_all)

            for i in range(1, 4):
                ax1.plot(r,
                    np.ones_like(r) * 10**(-i),
                    linewidth=0.8,
                    linestyle="--",
                    color='gray',
                    zorder=100,
                )

            if not SAVEERRORS:
                """ 
                Display errors and relevant config parameters for each version
                """
                print(f'VERSION {vv}:')
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


            

            # ax0.set_xlim([-3, 0.1])
            # ax1.set_xlim([-3, 0.1])
            ax0.set_ylim([1e-2, 1e4])
            ax1.set_ylim([1e-4, 1e0])

            ax1.set_xlabel(r'$\displaystyle  r/h \: [\mathrm{Mpc}]$',fontsize=18)
            ax1.set_ylabel(r'$\displaystyle \mathrm{rel. diff.}$',fontsize=20)
            ax0.set_ylabel(r"$\xi_{gg}(r)$",fontsize=22)

            ax0.xaxis.set_ticklabels([])
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            ax1.set_yscale("log")
            ax1.set_xscale("log")

            ax0.set_title(f"version {vv}")

            ax0.plot([], linewidth=0, marker='o', color='k', markersize=2, alpha=0.5, label="data")
            ax0.plot([], linewidth=1, color='k', alpha=1, label="emulator")
            ax0.legend(loc="upper right", fontsize=12)

            if not SAVEFIG:
                plt.show()
                # continue
            
            else:
            
                figdir = f'./plots/'

                if PRESENTATION:
                    savedir = "presentation"
                else:
                    savedir = self.emul_dir
            
                figdir = Path(f"{figdir}/{savedir}")
                figdir.mkdir(parents=True, exist_ok=True)
                
                figtitle = f'version{vv}.png' 

                if PRESENTATION:
                    week_number = datetime.now().strftime("%U")
                    figname = f"{figdir}/week{int(week_number)}_{self.emul_dir}_{figtitle}"
                else:
                    figname = f'{figdir}/{figtitle}'


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

# LR_tests = emulator_test(emul_dir="LR_test",flag="val", print_config_param=["learning_rate","patience"])
# batchsize_test = emulator_test(emul_dir="batchsize_test",print_config_param="batch_size") 
# LR_tests.plot_tpcf(plot_versions=[6],plot_range=False,plot_every_n=2)
# batchsize_test.plot_tpcf(plot_versions=[2])
# bias_patience_test = emulator_test(emul_dir="bias_patience_test",print_config_param=["zero_bias", "patience"])
# bias_patience_test.plot_tpcf(plot_versions=[3],plot_every_n=3,plot_range=False)
# hidden_dims_test = emulator_test(emul_dir="hidden_dims_test",print_config_param= ["hidden_dims", "patience"])
# hidden_dims_test2 = emulator_test(emul_dir="hidden_dims_test2",print_config_param="hidden_dims")
# hidden_dims_test.plot_tpcf(plot_versions=[9],plot_every_n=2,plot_range=False)
# hidden_dims_test2.plot_tpcf(plot_versions="all",plot_every_n=2,plot_range=False)


# dropout_test = emulator_test(emul_dir="dropout_test",print_config_param="dropout")
# grad_clip_val_test = emulator_test(emul_dir="grad_clip_val_test",print_config_param="gradient_clip_val")
# dropout_test.plot_tpcf(plot_versions="all",plot_every_n=2,plot_range=False)
# grad_clip_val_test.plot_tpcf(plot_versions="all",plot_every_n=2,plot_range=False)

# ExpLR_test = emulator_test(emul_dir="ExpLR_test",print_config_param=["learning_rate", "weight_decay", "hidden_dims"],flag="val",)
# ExpLR_test2 = emulator_test(emul_dir="ExpLR_test2",print_config_param=["learning_rate", "weight_decay", "hidden_dims"],flag="val",)

# PUSH = True 
# PRESENTATION = True 


# ExpLR_test.plot_tpcf(plot_versions=[3],plot_every_n=3,plot_range=False,)
# ExpLR_test2.plot_tpcf(plot_versions=[0],plot_every_n=3,plot_range=False,)

