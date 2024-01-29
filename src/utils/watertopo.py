import os
import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random


class WaterTopo():
    @staticmethod
    def load_simulations(save_folder, sim_amount, number_grids, use_augmented_data=True, random_state=42):
        """
        Static method to create a simulation from a filepath.
        - save_folder: string, that provides the filepath of where to look for the following folders:
            - DEM   (topography)
            - VX    (x-velocities)
            - VY    (y-velocities)
            - WD (waterlevels)
        - sim_amount: int, amount of simulations to load.
        - number_grid: int, predefined grid dimension to help shape the data in the correct form.

        return:
        - List of Simulation objects.
        """
        fnames = os.listdir(save_folder + "/DEM")

        if not use_augmented_data:
            fnames = [name for name in fnames if "o" in name]
            
        if fnames[0][0].isupper():
            name_dict = {"DEM":"DEM",
                         "WD": "WD"}
        else:
            name_dict = {"DEM":"dem",
                         "WD": "wd"}

        if sim_amount > len(fnames):
            raise Exception(f"Please select an amount smaller than {len(fnames)}.")
        
        np.random.seed(random_state)
        idx = random.sample(range(len(fnames)), sim_amount)
        sims = []

        for i in range(sim_amount):
            name = fnames[idx[i]]

            topo = np.loadtxt(f"{save_folder}/DEM/" + name_dict["DEM"] + name[3:])[:, 2].reshape(number_grids,number_grids)
            wd = np.loadtxt(f"{save_folder}/WD/" + name_dict["WD"] + name[3:]).reshape(-1,number_grids,number_grids)

            sims.append(WaterTopo(topo, wd))

        return sims
    
    def __init__(self, topography, wd):
        """
        A class that contains for a simulation:
        - topography:   2D array
        - wd:           3D array
        """
        self.topography = topography
        self.wd = wd

    def return_timestep(self, timestep):
        """
        timestep: timestep at which you want the data to be returned

        returns: wd, vx, vy at specified timestep
        """
        return self.wd[timestep]
    
    def plot_animation(self, cmap_topo="terrain", cmap_flood="Blues"):
        fig, axs = plt.subplots(1,2,figsize=(8,4))
        axs[0].imshow(self.topography, cmap=cmap_topo, origin="lower")

        # Create list of images that are animated
        imgs = []
        for i in range(self.wd.shape[0]):
            im = axs[1].imshow(self.wd[i], cmap=cmap_flood, origin='lower', animated=True)

            if i==0:
                axs[1].imshow(self.wd[i], cmap=cmap_flood, origin='lower')
            imgs.append([im])

        # Create the animation
        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True,
                                        repeat_delay=1000)
        return ani