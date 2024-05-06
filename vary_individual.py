import numpy as np
from pathlib import Path
import h5py
import pandas as pd 
from typing import List, Optional, Union

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simpson 
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, message="Input line")

import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys 
sys.path.append("../emul_utils")
from _plot import set_matplotlib_settings, get_CustomCycler
set_matplotlib_settings()
custom_cycler = get_CustomCycler()
from _predict import Predictor

sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/HOD/HaloModel/HOD_and_cosmo_emulation/parameter_samples")
from HOD_and_cosmo_prior_ranges import get_cosmo_params_prior_range, get_HOD_params_prior_range, get_fiducial_HOD_params, get_fiducial_cosmo_params # type: ignore

global SAVEFIG 
global PUSH
global PRESENTATION

SAVEFIG         = False
PUSH            = False
PRESENTATION    = False

dataset_names = {"train": "Training", "val": "Validation", "test": "Test"}

D13_DATA_PATH = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files") 
FIDUCIAL_PATH = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/fiducial_data")



FIDUCIAL_HOD    = get_fiducial_HOD_params()
FIDUCIAL_COSMO  = get_fiducial_cosmo_params()
HOD_PRIORS      = get_HOD_params_prior_range()
COSMO_PRIORS    = get_cosmo_params_prior_range()
FIDUCIAL_DICT   = {**FIDUCIAL_HOD, **FIDUCIAL_COSMO}
PRIORS_DICT     = {**HOD_PRIORS, **COSMO_PRIORS}

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
            max_r = np.floor(fff[self.simulation_keys[0]]["node0"][self.r_key][-1])
            self.r_default = fff[self.simulation_keys[0]]["node0"][self.r_key][...]
        self.N_nodes_per_simulation = {
            "test": 100,
            "val": 100,
            "train": 500,
        }

        self.r_perp_binedge = np.geomspace(0.5, 40, 40)
        self.r_perp         = (self.r_perp_binedge[1:] + self.r_perp_binedge[:-1]) / 2
        self.r_para         = np.linspace(0, 105, int(1500))
        self.r_from_rp_rpi  = np.sqrt(self.r_perp.reshape(-1, 1)**2 + self.r_para.reshape(1, -1)**2)

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
        xiR_func        = ius(r, xi, ext=1)

        wp = 2.0 * simpson(
            xiR_func(self.r_from_rp_rpi), 
            self.r_para, 
            axis=-1,
            )
      
        return wp
    


    def plot_proj_corrfunc_varying_omega_b_and_kappa(
            self, 
            versions:          Union[List[int], range, str] = "all",
            legend:                 bool    = True,
            outfigs:                str     = None,
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

        kappa_priors            = PRIORS_DICT["kappa"]
        wb_priors               = PRIORS_DICT["wb"]
        kappa_low, kappa_high   = kappa_priors[0] * 1.01, kappa_priors[1] * 0.99 
        wb_low, wb_high         = wb_priors[0] * 1.01, wb_priors[1] * 0.99
        kappas                  = [kappa_low, FIDUCIAL_DICT["kappa"], kappa_high]
        omega_bs                = [wb_low, FIDUCIAL_DICT["wb"], wb_high]
        params_list = [kappas, omega_bs]

        fiducial_params_kappa = np.zeros((len(kappas), len(self.param_names)))
        fiducial_params_omega_b = np.zeros((len(omega_bs), len(self.param_names)))
        for ii in range(3):
            for jj, param_name in enumerate(self.param_names):
                if param_name == "kappa":
                    fiducial_params_kappa[ii, jj] = kappas[ii]
                else:
                    fiducial_params_kappa[ii, jj] = FIDUCIAL_DICT[param_name]

                if param_name == "wb":
                    fiducial_params_omega_b[ii, jj] = omega_bs[ii]
                else:
                    fiducial_params_omega_b[ii, jj] = FIDUCIAL_DICT[param_name]

        param_sets = [fiducial_params_kappa, fiducial_params_omega_b] 
        param_legends = [r"$\kappa$", r"$\omega_b$"]
        colors = ["red", "blue", "green"]
        ls_ = ["dashed", "solid", "dashed"]

        for vv in version_list:

            _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.emul_dir)

            fig = plt.figure(figsize=(14, 7))
            gs = gridspec.GridSpec(1, 2, wspace=0)
            plt.rc('axes', prop_cycle=custom_cycler)
            ax0_ = plt.subplot(gs[0])
            ax1_ = plt.subplot(gs[1])

            for ii, param_set in enumerate(param_sets):  
                ax0 = plt.subplot(gs[ii])
                # ax0.set_prop_cycle(custom_cycler)
                # Load emulator for this version
                for jj, params in enumerate(param_set):

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [params] * len(self.r_default)
                            )
                            , self.r_default
                            ))
                    
                    # xi_data = fff_cosmo_HOD[self.xi_key][...]
                    xi_emul = _emulator(params_batch) 
                    # wp_data = self.compute_wp_from_xi_of_r(xi_data, r_data)
                    wp_emul = self.compute_wp_from_xi_of_r(xi_emul, self.r_default)
                    ax0.plot(self.r_perp, self.r_perp * wp_emul, linewidth=1, alpha=1, color=colors[jj], ls=ls_[jj], label=f"{param_legends[ii]} = {params_list[ii][jj]:.3f}")
                    # ax0.plot(self.r_perp,wp_emul, linewidth=1, alpha=1, color=colors[jj], ls=ls_[jj], label=f"{param_legends[ii]} = {params[0]:.3f}")

                # ax0.xaxis.set_ticklabels([])
                ax0.set_xscale("log")
                ax0.set_ylim([95,210])
                # Increase size of tick labels 
                ax0.tick_params(axis='both', which='major', labelsize=20)
                ax0.set_xlabel(r'$\displaystyle  r_\bot \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25)
                if legend:
                    ax0.legend(loc="lower left", fontsize=22)
            ylabel =  r"$r_\bot w_p(r_\bot)\quad [h^{-2}\,\mathrm{Mpc}^{2}]$"
            ax1_.yaxis.set_ticklabels([])

            ax0_.set_ylabel(ylabel,fontsize=25)
            if not SAVEFIG:
                plt.show()
                return 
            for outfig in outfigs:
                print(f'save plot to {outfig}')
                plt.savefig(
                    Path(outfig),
                    dpi=150 if outfig.endswith(".png") else None,
                    bbox_inches="tight",
                    pad_inches=0.05,        
                )
            plt.close(fig)



    def plot_proj_corrfunc_varying_omega_c_and_sigma8(
            self, 
            versions:          Union[List[int], range, str] = "all",
            legend:                 bool    = True,
            outfigs:                str     = None,
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

        sigma8_priors            = PRIORS_DICT["sigma8"]
        wc_priors               = PRIORS_DICT["wc"]
        sigma8_low, sigma8_high   = sigma8_priors[0] * 1.01, sigma8_priors[1] * 0.99 
        wc_low, wc_high         = wc_priors[0] * 1.01, wc_priors[1] * 0.99
        sigma8s                  = [sigma8_low, FIDUCIAL_DICT["sigma8"], sigma8_high]
        omega_cs                = [wc_low, FIDUCIAL_DICT["wc"], wc_high]
        params_list = [sigma8s, omega_cs]

        fiducial_params_sigma8 = np.zeros((len(sigma8s), len(self.param_names)))
        fiducial_params_omega_c = np.zeros((len(omega_cs), len(self.param_names)))
        for ii in range(3):
            for jj, param_name in enumerate(self.param_names):
                if param_name == "sigma8":
                    fiducial_params_sigma8[ii, jj] = sigma8s[ii]
                else:
                    fiducial_params_sigma8[ii, jj] = FIDUCIAL_DICT[param_name]

                if param_name == "wc":
                    fiducial_params_omega_c[ii, jj] = omega_cs[ii]
                else:
                    fiducial_params_omega_c[ii, jj] = FIDUCIAL_DICT[param_name]

        param_sets = [fiducial_params_sigma8, fiducial_params_omega_c] 
        param_legends = [r"$\sigma_8$", r"$\omega_\mathrm{cdm}$"]
        colors = ["red", "blue", "green"]
        ls_ = ["dashed", "solid", "dashed"]

        for vv in version_list:

            _emulator       = cm_emulator_class(version=vv,LIGHTING_LOGS_PATH=self.emul_dir)

            fig = plt.figure(figsize=(14, 7))
            gs = gridspec.GridSpec(1, 2, wspace=0)
            plt.rc('axes', prop_cycle=custom_cycler)
            ax0_ = plt.subplot(gs[0])
            ax1_ = plt.subplot(gs[1])

            for ii, param_set in enumerate(param_sets):  
                ax0 = plt.subplot(gs[ii])
                # ax0.set_prop_cycle(custom_cycler)
                # Load emulator for this version
                for jj, params in enumerate(param_set):

                    params_batch   = np.column_stack(
                        (np.vstack(
                            [params] * len(self.r_default)
                            )
                            , self.r_default
                            ))
                    
                    # xi_data = fff_cosmo_HOD[self.xi_key][...]
                    xi_emul = _emulator(params_batch) 
                    # wp_data = self.compute_wp_from_xi_of_r(xi_data, r_data)
                    wp_emul = self.compute_wp_from_xi_of_r(xi_emul, self.r_default)
                    ax0.plot(self.r_perp, self.r_perp * wp_emul, linewidth=1, alpha=1, color=colors[jj], ls=ls_[jj], label=f"{param_legends[ii]} = {params_list[ii][jj]:.3f}")
                    # ax0.plot(self.r_perp,wp_emul, linewidth=1, alpha=1, color=colors[jj], ls=ls_[jj], label=f"{param_legends[ii]} = {params[0]:.3f}")

                # ax0.xaxis.set_ticklabels([])
                ax0.set_xscale("log")
                ax0.set_ylim([95,250])
                # Increase size of tick labels 
                ax0.tick_params(axis='both', which='major', labelsize=20)
                ax0.set_xlabel(r'$\displaystyle  r_\bot \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25)
                if legend:
                    ax0.legend(loc="upper left", fontsize=22)
            ylabel =  r"$r_\bot w_p(r_\bot)\quad [h^{-2}\,\mathrm{Mpc}^{2}]$"
            ax1_.yaxis.set_ticklabels([])

            ax0_.set_ylabel(ylabel,fontsize=25)
            if not SAVEFIG:
                plt.show()
                return 
            for outfig in outfigs:
                print(f'save plot to {outfig}')
                plt.savefig(
                    Path(outfig),
                    dpi=150 if outfig.endswith(".png") else None,
                    bbox_inches="tight",
                    pad_inches=0.05,        
                )
            plt.close(fig)


TPCF_sliced_3040 = TPCF_emulator(
    root_dir            =   "./emulator_data",
    dataset             =   "sliced_r",
    emul_dir            =   "batch_size_3040",
    flag                =   "test",
)

SAVEFIG = False
# SAVEFIG = True
# TPCF_sliced_3040.get_rel_err_all(2, percentile=68, overwrite=True)

def test_omega_b_and_kappa():
    outfig_stem = f"plots/thesis_figures/emulators/wp_emul"
    outfig1 = f"{outfig_stem}_varying_kappa_and_omega_b.png"
    outfig2 = f"{outfig_stem}_varying_kappa_and_omega_b.pdf"

    # outfig2 = f"{outfig_stem}_varying_omega_b.png"
    TPCF_sliced_3040.plot_proj_corrfunc_varying_omega_b_and_kappa(versions=2, legend=True, outfigs=[outfig1, outfig2])
    # TPCF_sliced_3040.plot_proj_corrfunc_varying_omega_b_and_kappa(versions=2, legend=True, outfigs=None)

    # outfig1 = f"{outfig_stem}_varying_kappa.pdf"
    # outfig2 = f"{outfig_stem}_varying_omega_b.pdf"
    # TPCF_sliced_3040.plot_proj_corrfunc_varying_omega_b_and_kappa(versions=2, legend=True, outfigs=[outfig1, outfig2])


def test_omega_c_and_sigma8():
    outfig_stem = f"plots/thesis_figures/emulators/wp_emul"
    outfig1 = f"{outfig_stem}_varying_sigma8_and_omega_cdm.png"
    outfig2 = f"{outfig_stem}_varying_sigma8_and_omega_cdm.pdf"
    TPCF_sliced_3040.plot_proj_corrfunc_varying_omega_c_and_sigma8(versions=2, legend=True, outfigs=[outfig1, outfig2])


test_omega_b_and_kappa()
test_omega_c_and_sigma8()