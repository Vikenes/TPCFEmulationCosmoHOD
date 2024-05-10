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

PARAM_LABELS = {
            "wc"         : r"\omega_\mathrm{cdm}", 
            "wb"         : r"\omega_b", 
            "kappa"      : r"\kappa", 
            "sigma8"     : r"\sigma_8",
            "w0"         : r"w_0",
            "wa"         : r"w_a",
            "ns"         : r"n_s",
            "alpha_s"    : r"\mathrm{d}n_s/\mathrm{d}\ln{k}",
            "N_eff"      : r"N_\mathrm{eff}",
            "log10M1"    : r"\log{M_1}",
            "sigma_logM" : r"\sigma_{\log{M}}",
            "kappa"      : r"\kappa",
            "alpha"      : r"\alpha",
            "log10_ng"   : r"\log{n_g}",
            }

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
    
    def get_min_fid_max(
            self, 
            param_name, 
            min_prior_factor=0.01,
            max_prior_factor=0.01,
            ):
            if PRIORS_DICT[param_name][0] < 0:
                min_prior_factor = -min_prior_factor
            if PRIORS_DICT[param_name][-1] < 0:
                max_prior_factor = -max_prior_factor
            return np.array([
                PRIORS_DICT[param_name][0] * (1 + min_prior_factor), 
                FIDUCIAL_DICT[param_name], 
                PRIORS_DICT[param_name][-1] * (1 - max_prior_factor),
                ])
    

    def plot_proj_corrfunc_vary_w0_wa(
            self, 
            varying_param_keys:     List[str],
            version:                int     = 2,
            legend:                 bool    = True,
            outfig_stem:            str     = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
        varying_param_values = {}
        emul_param_inputs = {}
        for param_key in varying_param_keys:
            if param_key == "w0":
                param_values = self.get_min_fid_max(param_key, max_prior_factor=0.1)
                param_values = np.concatenate((param_values, [-0.7]))
            else:
                param_values = self.get_min_fid_max(param_key)
            varying_param_values[param_key] = param_values
            emul_param_inputs[param_key] = np.array(
                [[FIDUCIAL_DICT[key] if key != param_key else param_values[i] for key in self.param_names] for i in range(len(param_values))])
        
        colors          = {
            "wa": ["red", "blue", "green"],
            "w0": ["red", "blue", "green", "black"],
        }
        ls_             = {
            "wa": ["dashed", "solid", "dashed"],
            "w0": ["dashed", "solid", "dashed", "solid"]
        }

        alphas          = {
            "wa": [0.7, 1, 0.7],
            "w0": [0.7, 1, 0.7, 0.7]
        }

        _emulator       = cm_emulator_class(version=version,LIGHTING_LOGS_PATH=self.emul_dir)

        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, wspace=0)
        plt.rc('axes', prop_cycle=custom_cycler)
        ax0_ = plt.subplot(gs[0])
        ax1_ = plt.subplot(gs[1])

        for ii, param_key in enumerate(varying_param_keys):  
            ax0 = plt.subplot(gs[ii])
            for jj, params in enumerate(emul_param_inputs[param_key]):

                params_batch   = np.column_stack(
                    (np.vstack(
                        [params] * len(self.r_default)
                        )
                        , self.r_default
                        ))
                
                xi_emul = _emulator(params_batch) 
                wp_emul = self.compute_wp_from_xi_of_r(xi_emul, self.r_default)
                ax0.plot(
                    self.r_perp, 
                    self.r_perp * wp_emul, 
                    linewidth=1, 
                    alpha=alphas[param_key][jj], 
                    color=colors[param_key][jj], 
                    ls=ls_[param_key][jj], 
                    label=rf"${PARAM_LABELS[param_key]} = {varying_param_values[param_key][jj]:.3f}$")

            ax0.set_xscale("log")
            ax0.set_ylim([95,250])
            ax0.tick_params(axis='both', which='major', labelsize=20)
            ax0.set_xlabel(r'$\displaystyle  r_\bot \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25)
            if legend:
                ax0.legend(loc="upper left", fontsize=18)
        ylabel =  r"$r_\bot w_p(r_\bot)\quad [h^{-2}\,\mathrm{Mpc}^{2}]$"
        ax1_.yaxis.set_ticklabels([])

        ax0_.set_ylabel(ylabel,fontsize=25, labelpad=10)

        if not SAVEFIG:
            plt.tight_layout()
            plt.show()
            return 
        outfig_png = Path(f"{outfig_stem}.png")
        outfig_pdf = Path(f"{outfig_stem}.pdf")
        print(f'Saving {outfig_png}')
        plt.savefig(
            outfig_png,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f'Saving {outfig_pdf}')
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)


    def plot_proj_corrfunc_vary_log10_ng_xi(
            self, 
            varying_param_keys:     List[str] = ["log10_ng_low", "log10_ng_high"],
            version:                int     = 2,
            legend:                 bool    = True,
            outfig_stem:            str     = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
        varying_param_values = {}
        emul_param_inputs = {}
        for param_key in varying_param_keys:
            if param_key == "log10_ng_low":
                # param_values = self.get_min_fid_max(param_key, max_prior_factor=0.01)
                param_values = np.array([
                    PRIORS_DICT["log10_ng"][0],
                    PRIORS_DICT["log10_ng"][0] * (1 - 0.05),
                    FIDUCIAL_DICT["log10_ng"],
                ])
                # param_values = np.concatenate((param_values, [PRIORS_DICT["log10_ng"][-1]]))
            elif param_key == "log10_ng_high":
                param_values = np.array([
                    PRIORS_DICT["log10_ng"][-1],
                    PRIORS_DICT["log10_ng"][-1] * (1 + 0.05),
                    FIDUCIAL_DICT["log10_ng"],
                ])
                # param_values = self.get_min_fid_max(param_key, max_prior_factor=0.01)
                # param_values = np.concatenate((param_values, [PRIORS_DICT["sigma_logM"][-1]]))
                # param_values = self.get_min_fid_max(param_key)
            varying_param_values[param_key] = param_values
            emul_param_inputs[param_key] = np.array(
                [[FIDUCIAL_DICT[key] if key != "log10_ng" else param_values[i] for key in self.param_names] for i in range(len(param_values))])
        
        colors          = {
            "log10_ng_low": ["black", "red", "blue",],
            "log10_ng_high": ["black", "green", "blue"],
        }
        ls_             = {
            "log10_ng_low": ["solid", "dashed", "solid"],
            "log10_ng_high": ["solid", "dashed", "solid"]
        }

        alphas          = {
            "log10_ng_low":  [1, 0.7, 0.7],
            "log10_ng_high": [1, 0.7, 0.7],
        }

        _emulator       = cm_emulator_class(version=version,LIGHTING_LOGS_PATH=self.emul_dir)

        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, wspace=0)
        plt.rc('axes', prop_cycle=custom_cycler)
        ax0_ = plt.subplot(gs[0])
        ax1_ = plt.subplot(gs[1])

        for ii, param_key in enumerate(varying_param_keys):  
            ax0 = plt.subplot(gs[ii])
            for jj, params in enumerate(emul_param_inputs[param_key]):

                params_batch   = np.column_stack(
                    (np.vstack(
                        [params] * len(self.r_default)
                        )
                        , self.r_default
                        ))
                
                xi_emul = _emulator(params_batch) 
                ax0.plot(
                    self.r_default, 
                    self.r_default**2 * xi_emul,
                    linewidth=1, 
                    alpha=alphas[param_key][jj], 
                    color=colors[param_key][jj], 
                    ls=ls_[param_key][jj], 
                    label=rf"${PARAM_LABELS['log10_ng']} = {varying_param_values[param_key][jj]:.3f}$")
                # ax0.set_yscale("log")

            ax0.set_xscale("log")
            ax0.tick_params(axis='both', which='major', labelsize=20)
            ax0.set_xlabel(r'$\displaystyle  r \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25)
        if legend:
            ax0_.legend(loc="lower left", fontsize=18)
            ax1_.legend(loc="upper left", fontsize=18)

        ylabel =  r"$r^2 \xi^R \quad [h^{-2}\,\mathrm{Mpc}^{2}]$"
        ax1_.yaxis.set_ticklabels([])

        ax0_.set_ylabel(ylabel,fontsize=25, labelpad=10)

        if not SAVEFIG:
            plt.tight_layout()
            plt.show()
            return 
        outfig_png = Path(f"{outfig_stem}.png")
        outfig_pdf = Path(f"{outfig_stem}.pdf")
        print(f'Saving {outfig_png}')
        plt.savefig(
            outfig_png,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f'Saving {outfig_pdf}')
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)


    def plot_proj_corrfunc_vary_log10_ng(
            self, 
            varying_param_keys:     List[str] = ["log10_ng_low", "log10_ng_high"],
            version:                int     = 2,
            legend:                 bool    = True,
            outfig_stem:            str     = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """
        varying_param_values = {}
        emul_param_inputs = {}
        for param_key in varying_param_keys:
            if param_key == "log10_ng_low":
                # param_values = self.get_min_fid_max(param_key, max_prior_factor=0.01)
                param_values = np.array([
                    PRIORS_DICT["log10_ng"][0],
                    -3.681270,
                    PRIORS_DICT["log10_ng"][0] * (1 - 0.05),
                    FIDUCIAL_DICT["log10_ng"],
                ])
                # param_values = np.concatenate((param_values, [PRIORS_DICT["log10_ng"][-1]]))
            elif param_key == "log10_ng_high":
                param_values = np.array([
                    PRIORS_DICT["log10_ng"][-1],
                    -3.212319,
                    PRIORS_DICT["log10_ng"][-1] * (1 + 0.05),
                    FIDUCIAL_DICT["log10_ng"],
                ])
                # param_values = self.get_min_fid_max(param_key, max_prior_factor=0.01)
                # param_values = np.concatenate((param_values, [PRIORS_DICT["sigma_logM"][-1]]))
                # param_values = self.get_min_fid_max(param_key)
            varying_param_values[param_key] = param_values
            emul_param_inputs[param_key] = np.array(
                [[FIDUCIAL_DICT[key] if key != "log10_ng" else param_values[i] for key in self.param_names] for i in range(len(param_values))])
        
        colors          = {
            "log10_ng_low": ["black", "red", "green", "blue",],
            "log10_ng_high": ["black", "red", "green", "blue"],
        }
        ls_             = {
            "log10_ng_low": ["dashed", "dashed", "dashed", "solid"],
            "log10_ng_high": ["dashed", "dashed", "dashed", "solid"]
        }

        alphas          = {
            "log10_ng_low":  [0.7, 0.7, 0.7, 1],
            "log10_ng_high": [0.7, 0.7, 0.7, 1],
        }

        _emulator       = cm_emulator_class(version=version,LIGHTING_LOGS_PATH=self.emul_dir)

        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, wspace=0)
        plt.rc('axes', prop_cycle=custom_cycler)
        ax0_ = plt.subplot(gs[0])
        ax1_ = plt.subplot(gs[1])

        for ii, param_key in enumerate(varying_param_keys):  
            ax0 = plt.subplot(gs[ii])
            for jj, params in enumerate(emul_param_inputs[param_key]):

                params_batch   = np.column_stack(
                    (np.vstack(
                        [params] * len(self.r_default)
                        )
                        , self.r_default
                        ))
                
                xi_emul = _emulator(params_batch) 
                wp_emul = self.compute_wp_from_xi_of_r(xi_emul, self.r_default)
                ax0.plot(
                    self.r_perp, 
                    self.r_perp * wp_emul, 
                    linewidth=1, 
                    alpha=alphas[param_key][jj], 
                    color=colors[param_key][jj], 
                    ls=ls_[param_key][jj], 
                    label=rf"${PARAM_LABELS['log10_ng']} = {varying_param_values[param_key][jj]:.3f}$")

            ax0.set_xscale("log")
            ax0.set_ylim([95,250])
            ax0.tick_params(axis='both', which='major', labelsize=20)
            ax0.set_xlabel(r'$\displaystyle  r_\bot \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25)
        if legend:
            ax0_.legend(loc="lower left", fontsize=18)
            ax1_.legend(loc="upper left", fontsize=18)

        ylabel =  r"$r_\bot w_p(r_\bot)\quad [h^{-2}\,\mathrm{Mpc}^{2}]$"
        ax1_.yaxis.set_ticklabels([])

        ax0_.set_ylabel(ylabel,fontsize=25, labelpad=10)

        if not SAVEFIG:
            plt.tight_layout()
            plt.show()
            return 
        outfig_png = Path(f"{outfig_stem}.png")
        outfig_pdf = Path(f"{outfig_stem}.pdf")
        print(f'Saving {outfig_png}')
        plt.savefig(
            outfig_png,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f'Saving {outfig_pdf}')
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)


    def plot_proj_corrfunc_vary_two_params(
            self, 
            varying_param_keys:     List[str],
            version:                int     = 2,
            min_prior_factor:       float   = 0.01,
            max_prior_factor:       float   = 0.01,
            legend:                 bool    = True,
            outfig_stem:            str     = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """

        varying_param_values = {}
        emul_param_inputs = {}
        for param_key in varying_param_keys:
            param_values = self.get_min_fid_max(param_key, min_prior_factor=min_prior_factor, max_prior_factor=max_prior_factor)
            varying_param_values[param_key] = param_values
            emul_param_inputs[param_key] = np.array([[FIDUCIAL_DICT[key] if key != param_key else param_values[i] for key in self.param_names] for i in range(3)])

        
        
        colors          = ["red", "blue", "green"]
        ls_             = ["dashed", "solid", "dashed"]
        alphas          = [0.7, 1, 0.7]


        _emulator       = cm_emulator_class(version=version,LIGHTING_LOGS_PATH=self.emul_dir)

        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, wspace=0)
        plt.rc('axes', prop_cycle=custom_cycler)
        ax0_ = plt.subplot(gs[0])
        ax1_ = plt.subplot(gs[1])

        for ii, param_key in enumerate(varying_param_keys):  
            ax0 = plt.subplot(gs[ii])
            for jj, params in enumerate(emul_param_inputs[param_key]):

                params_batch   = np.column_stack(
                    (np.vstack(
                        [params] * len(self.r_default)
                        )
                        , self.r_default
                        ))
                
                xi_emul = _emulator(params_batch) 
                wp_emul = self.compute_wp_from_xi_of_r(xi_emul, self.r_default)
                ax0.plot(
                    self.r_perp, 
                    self.r_perp * wp_emul, 
                    linewidth=1, 
                    alpha=alphas[jj], 
                    color=colors[jj], 
                    ls=ls_[jj], 
                    label=rf"${PARAM_LABELS[param_key]} = {varying_param_values[param_key][jj]:.3f}$")

            ax0.set_xscale("log")
            ax0.set_ylim([95,250])
            ax0.tick_params(axis='both', which='major', labelsize=20)
            ax0.set_xlabel(r'$\displaystyle  r_\bot \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25)
            if legend:
                ax0.legend(loc="upper left", fontsize=18)

        ylabel =  r"$r_\bot w_p(r_\bot)\quad [h^{-2}\,\mathrm{Mpc}^{2}]$"
        ax1_.yaxis.set_ticklabels([])

        ax0_.set_ylabel(ylabel,fontsize=25, labelpad=10)

        if not SAVEFIG:
            plt.tight_layout()
            plt.show()
            return 
        outfig_png = Path(f"{outfig_stem}.png")
        outfig_pdf = Path(f"{outfig_stem}.pdf")
        print(f'Saving {outfig_png}')
        plt.savefig(
            outfig_png,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f'Saving {outfig_pdf}')
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
        




    def plot_proj_corrfunc_vary_four_parameters(
            self, 
            varying_param_keys: List[str],
            version:            int  = 2,
            legend:             bool = True,
            outfig_stem:        str  = None,
            ):
        """
        nodes_per_simulation: Number of nodes (HOD parameter sets) to plot per simulation (cosmology) 
        masker_r: if True, only plot r < max_r_error. Noisy data for r > 60.
        xi_ratio: if True, plot xi/xi_fiducial of xi.  
        """

        
        
    
        varying_param_values = {}
        emul_param_inputs = {}
        for param_key in varying_param_keys:
            param_values = self.get_min_fid_max(param_key)
            varying_param_values[param_key] = param_values
            emul_param_inputs[param_key] = np.array([[FIDUCIAL_DICT[key] if key != param_key else param_values[i] for key in self.param_names] for i in range(3)])

        
        
        colors          = ["red", "blue", "green"]
        ls_             = ["dashed", "solid", "dashed"]
        alphas          = [0.7, 1, 0.7]
        
        _emulator       = cm_emulator_class(version=version,LIGHTING_LOGS_PATH=self.emul_dir)

        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(2, 2, wspace=0, hspace=0)
        plt.rc('axes', prop_cycle=custom_cycler)
        ax0_ = plt.subplot(gs[0])
        ax1_ = plt.subplot(gs[1])
        ax2_ = plt.subplot(gs[2])
        ax3_ = plt.subplot(gs[3])

        for ii, param_key in enumerate(varying_param_keys):

            ax = plt.subplot(gs[ii])
            for jj, params in enumerate(emul_param_inputs[param_key]):

                params_batch   = np.column_stack(
                    (np.vstack(
                        [params] * len(self.r_default)
                        )
                        , self.r_default
                        ))
                
                xi_emul = _emulator(params_batch) 
                wp_emul = self.compute_wp_from_xi_of_r(xi_emul, self.r_default)
                ax.plot(
                    self.r_perp, 
                    self.r_perp * wp_emul, 
                    linewidth=1, 
                    alpha=alphas[jj], 
                    color=colors[jj], 
                    ls=ls_[jj], 
                    label=rf"${PARAM_LABELS[param_key]} = {varying_param_values[param_key][jj]:.3f}$")

            ax.set_xscale("log")
            ax.set_xscale("log")

            ax.set_ylim([95,250])
            ax.tick_params(axis='both', which='major', labelsize=20)
            if legend:
                ax.legend(loc="upper left", fontsize=18)

        ax1_.yaxis.set_ticklabels([])
        ax3_.yaxis.set_ticklabels([])
        # Prevent overlap of yticks from ax_0 and ax_2
        ax0_.yaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))
        ax2_.yaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))

        # Set x-label spanning two columns
        fig.supxlabel(r'$\displaystyle  r_\bot \quad [h^{-1} \mathrm{Mpc}]$',fontsize=25, y=0.05)
        fig.supylabel(r"$r_\bot w_p(r_\bot)\quad [h^{-2}\,\mathrm{Mpc}^{2}]$",fontsize=25, x=0.05)

        if not SAVEFIG:
            plt.tight_layout()
            plt.show()
            return 
        outfig_png = Path(f"{outfig_stem}.png")
        outfig_pdf = Path(f"{outfig_stem}.pdf")
        print(f'Saving {outfig_png}')
        plt.savefig(
            outfig_png,
            dpi=150,
            bbox_inches="tight",
            # pad_inches=0.05,
        )
        print(f'Saving {outfig_pdf}')
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            # pad_inches=0.05,
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

def test_omega_b_and_kappa():
    outfig_stem = f"plots/thesis_figures/emulators/wp_emul_varying_kappa_and_omega_b"
    TPCF_sliced_3040.plot_proj_corrfunc_vary_two_params(
        varying_param_keys=["wb", "kappa"],
        outfig_stem=outfig_stem
        )

def test_omega_c_and_sigma8():
    outfig_stem = f"plots/thesis_figures/emulators/wp_emul_varying_sigma8_and_omega_cdm"
    TPCF_sliced_3040.plot_proj_corrfunc_vary_two_params(
        varying_param_keys=["wc", "sigma8"],
        outfig_stem=outfig_stem
        )
    

def test_ns_and_alpha_s():
    outfig_stem = f"plots/thesis_figures/emulators/wp_emul_varying_ns_and_alpha_s"
    TPCF_sliced_3040.plot_proj_corrfunc_vary_two_params(
        varying_param_keys=["ns", "alpha_s"],
        outfig_stem=outfig_stem
        )

def vary_four():
    TPCF_sliced_3040.plot_proj_corrfunc_vary_four_parameters(
        varying_param_keys=[
            "wc", 
            "wb", 
            "sigma8", 
            "kappa", 
            ],
        outfig_stem = "plots/thesis_figures/emulators/wp_emul_varying_kappa_wb_wc_sigma8"
        )
    
def test_w0_wa():
    TPCF_sliced_3040.plot_proj_corrfunc_vary_w0_wa(
        varying_param_keys=["w0","wa",], 
        outfig_stem = "plots/thesis_figures/emulators/wp_emul_varying_w0_wa"
        )
    
def test_ng():
    # TPCF_sliced_3040.plot_proj_corrfunc_vary_log10_ng_xi(
    #     # outfig_stem="plots/test_ng_xi"
    #     outfig_stem = "plots/thesis_figures/emulators/xi_emul_varying_log10_ng"
    # )
    TPCF_sliced_3040.plot_proj_corrfunc_vary_log10_ng(
        outfig_stem = "plots/thesis_figures/emulators/wp_emul_varying_log10_ng"
        # outfig_stem = "plots/test_ng_wp"
        )
    

# test_omega_b_and_kappa()
# test_omega_c_and_sigma8()
# test_ns_and_alpha_s()
# test_w0_wa()
# test_ng()
# vary_four()
# print(PRIORS_DICT["alpha_s"])