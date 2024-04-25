import numpy as np
from pathlib import Path
from datetime import datetime 
import h5py
import os 
from typing import List, Optional, Union
import sys 
sys.path.append("../emul_utils")
from _predict import Predictor

import matplotlib.pyplot as plt
from matplotlib import gridspec
from _plot import set_matplotlib_settings, get_CustomCycler
set_matplotlib_settings()
custom_cycler = get_CustomCycler()

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simpson 

import warnings 
warnings.filterwarnings("ignore", category=UserWarning, message="Input line")

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
    ):
        return self.predictor(np.array(params)).reshape(-1)


class TPCF_emulator:
    def __init__(self,
                root_dir:  str = "./emulator_data",
                dataset:   str = None,
                emul_dir:  str = "first_test",
                flag:      str = "val",
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

        feature_columns    = config_file["data"]["feature_columns"]  # parameter names in feature columns
        self.param_names    = [param for param in feature_columns if param != "r"]
        self.r_key          = "r"
        self.xi_key         = "xi"      # xi key in label columns

        with h5py.File(self.data_dir / f"TPCF_{flag}.hdf5", 'r') as fff:
            self.simulation_keys = [key for key in fff.keys() if key.startswith("AbacusSummit")]
            self.N_simulations   = len(self.simulation_keys)
        self.tot_nodes_per_simulation = {
            "test": 100,
            "val": 100,
            "train": 500,
        }

    def compute_wp_from_xi_of_r(
            self,
            xi:        np.ndarray,
            r:         np.ndarray,
    ):
        """
        Computes projected correlation function wp(r_perp) from xi(r)
        given by w_p(r_perp) = 2 * int_0^r_para_max xi(r) dr_para
        with r = sqrt(r_perp^2 + r_para^2). 
        """

        # Callable func to interpolate xi(r) 
        xiR_func        = ius(r, xi)

        wp = 2.0 * simpson(
            xiR_func(self.r_from_rp_rpi), 
            self.r_para, 
            axis=-1,
            )
      
        return wp
        

    def plot_proj_corrfunc(
            self, 
            versions:          Union[List[int], range, str] = "all",
            max_r_error:            float   = np.inf,
            min_r_error:            float   = 0.0,
            nodes_per_simulation:   int     = 1,
            legend:                 bool    = False,
            outfig:                 str     = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
     
        if min_r_error == 0.0 and max_r_error == np.inf:
            masked_r = False
        else:
            masked_r = True
        flag = self.flag 
        available_nodes = np.arange(self.tot_nodes_per_simulation[flag])
        np.random.seed(42)


        r_perp_binedge      = np.geomspace(0.5, 40, 40)
        self.r_perp         = (r_perp_binedge[1:] + r_perp_binedge[:-1]) / 2
        self.r_para         = np.linspace(0, 100, int(1000))
        self.r_from_rp_rpi  = np.sqrt(self.r_perp.reshape(-1, 1)**2 + self.r_para.reshape(1, -1)**2)
        

        
        fff   = h5py.File(self.data_dir / f"TPCF_{flag}.hdf5", 'r')

        if type(versions) == list or type(versions) == range:
            version_list = versions
        elif type(versions) == int:
            version_list = [versions]
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
                nodes_idx = np.random.choice(available_nodes, nodes_per_simulation, replace=False)

                for jj in nodes_idx:
                    fff_cosmo_HOD = fff_cosmo[f"node{jj}"]

                    r_data = fff_cosmo_HOD[self.r_key][...]
                    r_mask = (r_data > min_r_error) & (r_data < max_r_error)
                    r_data = r_data[r_mask]


                    params_batch   = np.column_stack(
                        (np.vstack(
                            [[fff_cosmo_HOD.attrs[param_name] for param_name in self.param_names]] * len(r_data)
                            )
                            , r_data
                            ))
                    
                    xi_data = fff_cosmo_HOD[self.xi_key][...][r_mask] 
                    xi_emul = _emulator(params_batch) 
                    wp_data = self.compute_wp_from_xi_of_r(xi_data, r_data)
                    wp_emul = self.compute_wp_from_xi_of_r(xi_emul, r_data)

                    rel_err = np.abs(wp_emul / wp_data - 1.0) 

                    ax0.plot(self.r_perp, self.r_perp * wp_data, linewidth=0, marker="o",  markersize=2, alpha=1)
                    ax0.plot(self.r_perp, self.r_perp * wp_emul, linewidth=1, alpha=1)
                    ax1.plot(self.r_perp, rel_err, linewidth=0.7, alpha=0.5, color="gray")

            for i in range(1, 4):
                ax1.plot(
                    self.r_perp,
                    np.ones_like(self.r_perp) * 10**(-i),
                    linewidth=0.8,
                    linestyle="--",
                    color='gray',
                    zorder=100,
                )

            ax0.xaxis.set_ticklabels([])
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_ylim([1e-4, 1e-1])



            ylabel =  r"$r_\bot w_p(r_\bot)\:[h^{-2}\,\mathrm{Mpc}^{2}]$"


            ax1.set_xlabel(r'$\displaystyle  r_\bot \: [h^{-1} \mathrm{Mpc}]$',fontsize=18)
            ax0.set_ylabel(ylabel,fontsize=22)
            ax1.set_ylabel(r'$\displaystyle \left|\frac{w_p^\mathrm{pred} - w_p^\mathrm{data}}{w_p^\mathrm{pred}}\right|$',fontsize=15)

            # plot_title = f"Version {vv}. {dataset_names[flag]} data \n"
            # plot_title += rf"Showing {nodes_per_simulation} sets of $\vec{{\mathcal{{G}}_i}}$ for all {self.N_simulations} sets of $\vec{{\mathcal{{C}}_j}}$"
            # ax0.set_title(plot_title)

            ax0.plot([], linewidth=0, marker='o', color='k', markersize=2, alpha=0.5, label="Data")
            ax0.plot([], linewidth=1, color='k', alpha=1, label="Emulator")
            if legend:
                ax0.legend(loc="upper right", fontsize=12)

            if not SAVEFIG and outfig is None:
                plt.show()
            
            else:

                if outfig is None:
                    figdir = self.fig_dir
                    figdir.mkdir(parents=True, exist_ok=True)
                    figtitle = f"version{vv}_wp.png"
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
)

# SAVEFIG = True
outfig_stem = f"plots/thesis_figures/emulators/wp_from_xi_{TPCF_sliced_3040.flag}"
TPCF_sliced_3040.plot_proj_corrfunc(versions=2, nodes_per_simulation=1, legend=True, outfig=f"{outfig_stem}.png")
TPCF_sliced_3040.plot_proj_corrfunc(versions=2, nodes_per_simulation=1, legend=True, outfig=f"{outfig_stem}.pdf")


