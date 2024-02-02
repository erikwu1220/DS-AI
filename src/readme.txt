This folder contains the source code for the project of group FLOOD 4.

- 'animations' contains generated gifs showing results
- 'data_processing' contains notebooks to do the data normalization and augmentation
- 'models' contains .py for each of the trained model classes
- 'results' contains the demo notebook, as well as the following subfolders:
    * 'create_animations', used for creating animations
    * 'error_accumulation', used for determining the accumulated MSE for recursive predictions, and plotting them (for all models). The best_model subfolder contains the MSE over time for the dropout U-NET models (and the MSE for the best U-NET architecture, which is "unet_32_64_orig_data80_skip5_hardmask5")
    * 'trained_models' contains subfolders for each of the trained model types, i.e. the best settings (based on the best validation error during training)
- 'training' contains notebooks for training each of the models
- 'utils' contains .py files used throughout the training, animation, demo, etc. notebooks