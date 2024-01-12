import glob
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class Simulation():
    @staticmethod
    def load_simulation(save_folder, sim_number, number_grids):
        """
        Static method to create a simulation from a filepath.
        - save_folder: string, that provides the filepath of where to look for the following folders:
            - DEM (topography)
            - VX (x-velocities)
            - VY (y-velocities)
            - WD (waterlevels)
        - sim_number: int or float, which simulation number to load from this filepath.
            - Training: 1 - 80
            - Test 1: 501-520
            - Test 2: 10001-10020
        - number_grid: int, predefined grid dimension to help shape the data in the correct form.

        return:
        - Simulation object.
        """
        coords = np.loadtxt(f"{save_folder}\\DEM\\DEM_{sim_number}.txt")[:, :2]
        topo = np.loadtxt(f"{save_folder}\\DEM\\DEM_{sim_number}.txt")[:, 2].reshape(number_grids,number_grids)
        wd = np.loadtxt(f"{save_folder}\\WD\\WD_{sim_number}.txt").reshape(-1,number_grids,number_grids)
        vx = np.loadtxt(f"{save_folder}\\VX\\VX_{sim_number}.txt").reshape(-1,number_grids,number_grids)
        vy = np.loadtxt(f"{save_folder}\\vy\\vy_{sim_number}.txt").reshape(-1,number_grids,number_grids)
        
        return Simulation(coords, topo, wd, vx, vy)
    
    @staticmethod
    def load_simulations(save_folder, sim_amount, number_grids, random_state=42):
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
        fnames = os.listdir(save_folder + "\\DEM")

        if sim_amount > len(fnames):
            raise Exception(f"Please select an amount smaller than {len(fnames)}.")
        
        np.random.seed(random_state)
        idx = np.random.randint(0, len(fnames), sim_amount)
        sims = []

        for i in range(sim_amount):
            name = fnames[idx[i]]

            coords = np.loadtxt(f"{save_folder}\\DEM\\{name}")[:, :2]
            topo = np.loadtxt(f"{save_folder}\\DEM\\{name}")[:, 2].reshape(number_grids,number_grids)
            wd = np.loadtxt(f"{save_folder}\\WD\\wd{name[3:]}").reshape(-1,number_grids,number_grids)
            vx = np.loadtxt(f"{save_folder}\\VX\\vx{name[3:]}").reshape(-1,number_grids,number_grids)
            vy = np.loadtxt(f"{save_folder}\\vy\\vy{name[3:]}").reshape(-1,number_grids,number_grids)

            sims.append(Simulation(coords, topo, wd, vx, vy))

        return sims

                    
    # @staticmethod
    # def load_simulation_processed(save_folder_topo, sim_number_topo, save_folder_data, sim_number_data, number_grids):
    #     """
    #     Static method to create a simulation from a filepath.
    #     - save_folder_topo: string, that provides the filepath of where to look for the following folders:
    #         - DEM (topography)
    #     - sim_number_topo: int, float, or string, which topography number to load from this filepath.
    #         - Training: 1 - 80
    #         - Test 1: 501-520
    #         - Test 2: 10001-10020
    #     - save_folder_data: string, that provides the filepath of where to look for the following folders:
    #         - VX (x-velocities)
    #         - VY (y-velocities)
    #         - WD (waterlevels)
    #     - sim_number: int or float, which data number to load from this filepath.
    #         - Training: 1 - 80
    #         - Test 1: 501-520
    #         - Test 2: 10001-10020
    #     - number_grid: int, predefined grid dimension to help shape the data in the correct form.

    #     return:
    #     - Simulation object.
    #     """
    #     coords = np.loadtxt(f"{save_folder_topo}\\DEM\\DEM_{sim_number_topo}.txt")[:, :2]
    #     topo = np.loadtxt(f"{save_folder_topo}\\DEM\\DEM_{sim_number_topo}.txt")[:, 2].reshape(number_grids,number_grids)
    #     wd = np.loadtxt(f"{save_folder_data}\\WD\\wd_tra_norm{sim_number_data}").reshape(-1,number_grids,number_grids)
    #     vx = np.loadtxt(f"{save_folder_data}\\VX\\vx_tra_norm{sim_number_data}").reshape(-1,number_grids,number_grids)
    #     vy = np.loadtxt(f"{save_folder_data}\\VY\\vy_tra__norm{sim_number_data}").reshape(-1,number_grids,number_grids)
        
    #     return Simulation(coords, topo, wd, vx, vy)
    

    def __init__(self, coordinates, topography, wd, vx, vy):
        """
        A class that contains for a simulation:
        - topography:   2D array
        - wd:           3D array
        - vx:           3D array
        - vy:           3D array
        """
        self.coordinates = coordinates
        self.topography = topography
        self.wd = wd
        self.vx = vx
        self.vy = vy

        # Store coordinates for later
        self.x = np.arange(0, topography.shape[0])
        self.y = np.arange(0, topography.shape[1])

    def return_timestep(self, timestep):
        """
        timestep: timestep at which you want the data to be returned

        returns: wd, vx, vy at specified timestep
        """
        return self.wd[timestep], self.vx[timestep], self.vy[timestep]


    def save_simulation(self, save_folder, sim_number):
        """
        Saves a simulation similarly to the syntax used in the original "raw_data" folder.

        save_folder: str, location of DEM, VX, VY, WD folders.

        return: -
        """
        dem = np.vstack((np.round(self.coordinates.T, 1), self.topography.reshape(-1)))
        # print(dem.T)

        fmt = '%1.1f', '%1.1f', '%1.5f'
        np.savetxt(f"{save_folder}\\DEM\\DEM_{sim_number}.txt", dem.T, fmt=fmt)

        fmt = '%1.4f'
        format_size = self.wd.shape[1]**2
        np.savetxt(f"{save_folder}\\WD\\WD_{sim_number}.txt", self.wd.reshape(format_size,-1), fmt=fmt)
        np.savetxt(f"{save_folder}\\VX\\VX_{sim_number}.txt", self.vx.reshape(format_size,-1), fmt=fmt)
        np.savetxt(f"{save_folder}\\VY\\VY_{sim_number}.txt", self.vy.reshape(format_size,-1), fmt=fmt)

        return

    def plot_vector(self, timestep, cmap_topo="terrain", cmap_flood="Blues"):

        fig, axs = plt.subplots(1,2,figsize=(8,4))

        axs[0].imshow(self.topography, cmap=cmap_topo, origin="lower")
        axs[1].quiver(self.x, self.y, self.vx[timestep], self.vy[timestep], cmap=cmap_flood)

        return fig


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