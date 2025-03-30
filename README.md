This repo is a work-in-progress attempt to hold the Patch-NN (demo) portion of CMB-ML, separately from the rest. 

# Installation instructions:

- Download the CMB-ML repository
    - `git clone git@github.com:CMB-ML/cmb-ml.git`
    - `cd cmb-ml`
    - `git switch whatever` (dev-rm-pyilc)
- Create the conda environment 
    - Be sure the current working directory is the main CMB-ML repository folder
    - Remove old conda installations (and Poetry... which can be gotten rid of as a whole)
        - `conda env remove -n cmb-ml`
    - Create the CMB-ML environment `conda env create -f env.yaml`
    - To change the name of the environment, edit the file or use a different command.
- Activate the conda environment
    - `conda activate cmb-ml`
- Install CMB-ML
    - Still within the main CMB-ML repository folder
    - `which pip` (ensure that the response is within the conda environment)
    - `pip install .`
- Install required packages for this repository
  - `cd /whatever` to navigate to this repository 
  - `conda env update -n cmb-ml -f env.yaml`
- Configure your local system
  - See [Setting up your environment](https://github.com/CMB-ML/cmb-ml-tutorials/blob/main/C_setting_up_local.ipynb) for more information
  - Set your CMB_ML_DATA environment variable, and ensure that the directory exists
    - E.g., `export CMB_ML_DATA=/data/jim/CMB_Data`
- Download some external science assets and the CMB-ML assets
  - See [Setting up your environment](https://github.com/CMB-ML/cmb-ml-tutorials/blob/main/C_setting_up_local.ipynb) for more information
  - External science assets include Planck's observations maps (from which we get information for producing noise) and Planck's NILC prediction map (for the mask; NILC is a parameter)
  - These are available from the original sources and a mirror set up for this purpose
  - CMB-ML assets include the substitute detector information and information required for downloading datasets
  - If you are not creating simulations, you only need one external science asset: "COM_CMB_IQU-nilc_2048_R3.00_full.fits" (for the mask)
  - Scripts are available in the `get_data` folder, which will download all files.
    - [Downloads from original sources](./get_data/get_assets.py) gets files from the official sources (and the CMB-ML files from this repo)
    - If you prefer to download fewer files, adjust [this executor](get_data/stage_executors/A_get_assets.py) (not recommended)