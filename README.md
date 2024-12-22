# TPCFEmulationCosmoHOD
A neural network emulator that predicts the Two-point correlation function (TPCF) of galaxies using PyTorch Lightning. 

The data is constructed from mock galaxy catalogues from different cosmological simulations. 

**TBD:** Clean up/add doc. to `results.py` and `wp_plot.py`.

**Requirements:** 
 - The `EmulationUtilities` repository. The path to this directory must be added to the header of all scripts in this directory to work.
 - Training data, generated with the script `HOD/HaloModel/HOD_and_cosmo_emulation/make_tpcf_emulation_data_files.py` from the `HOD` repository. 

### Things to ignore
Some things are irrelevant/outdated. This includes:
 - `original_config.yaml`: Was used for a different task with an earlier version of `EmulationUtilities`. 
 - The figures found in `plots` were made prior to numerous modifications of the plotting code. Towards the end, figures were stored directly in the `thesis_figures` directory. 
 - `vary_individual.py`: Script used to test the emulator's sensitivity to changes in specific parameters.

### config.yaml
The config file used to train the final network was not stored (it was in the gitignore previously), so the current `config.yaml` file in the repository is an attempt to recreate it. Hence, minor errors and issues may have entered it. However, there are methods for comparing the new result with some of the old results. I will cover this when discussing `results.py`

### train_NN.py 
The only thing that has to be done before training can start, is to input the correct paths in the `config.yaml` file. Both paths under `data` and `training` needs updating. By using the `log_save_dir` under `training`, the emulators are stored at `default_root_dir/log_save_dir/version_X`. 



### results.py and wp_plot.py
The two scripts are mostly similar, with `results.py` concerning xi, i.e. the direct emulator output, while `wp_plot.py` concerns the projection of xi. 

Both scripts include functions to save various output errors from the emulator.
In `results.py` there are two error-related functions not found in `wp_plot.py`. These are `save_tpcf_errors(...)` and `print_tpcf_errors(...)`. These functions were written during hyperparam tests to simplify comparison of results. Essentially, they create a nicely formatted txt file, with a corresponding script for outputting the results. **Not necessary to run**.

The function `get_rel_err_all(...)` is in both scripts. For a given emulator, it computes relative errors of the emulator across all nodes in the test data (i.e. all HOD choices for every cosmo). The output is stored in the `rel_errors` subdirectory. The errors saved are the mean, median, stddev and 68th percentile, all computed as a function of separation distance. 

**Important:** The numpy arrays from the final emulator were stored in the `rel_errors` subdir. These can be used to check that the new emulator works properly, by computing new errors and compare with the old ones. 

The plotting functions require the numpy arrays to have been computed in order to work. By plotting the new predictions from the new emulator, you can compare with Fig. 5.2 and 5.3 in the thesis to check that it coincides. 

*Note:* The emulator uses a fixed seed. Results should therefore be reproducable. 