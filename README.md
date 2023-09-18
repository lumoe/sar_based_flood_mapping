# SAR Based Flood Mapping

- **Data Location**: The data is situated in the `data/` folder. Your data should be moved into this folder.
- **Data Folder Structure**:
  - `flood`: Contains all the flood training samples.
  - `water`: Contains the training samples for water.
  - `test`: Contains all the test data.
  - `tmp`: Contains the image reference pairs created for training or testing.
- **Environment File**: The `env.yml` file includes a working conda environment. For development, specific GPU drivers are required but not included in the conda package.
- **Training Dataset**: Generation of the training dataset is accomplished via `python generate_training_chips.py`.
- **Test Dataset**: Generation of the test dataset is accomplished via `python test_data_chips.py`.
- **Training Script**: Contained in the `train.py` script is all the information needed for either pre-training or fine-tuning the model. Execution is done by running `python train.py`.
- **Inference Script**: The `run_inference.py` file includes the script for running inference on the test data, searching for the best threshold value, and saving the results.
- **Old Folder**: Stored in the `old` folder are previous failed attempts at generating training and test samples.
- **Utils Folder**: Contained in the `utils` folder is Python code for loading training and test data, as well as helper functions for building paths to the images.
