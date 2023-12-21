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
        topo = np.loadtxt(f"{save_folder}\\DEM\\DEM_{sim_number}.txt")[:, 2].reshape(number_grids,number_grids)
        wd = np.loadtxt(f"{save_folder}\\WD\\WD_{sim_number}.txt").reshape(-1,number_grids,number_grids)
        vx = np.loadtxt(f"{save_folder}\\VX\\VX_{sim_number}.txt").reshape(-1,number_grids,number_grids)
        vy = np.loadtxt(f"{save_folder}\\vy\\vy_{sim_number}.txt").reshape(-1,number_grids,number_grids)
        
        return Simulation(topo, wd, vx, vy)
    

    def __init__(self, topography, wd, vx, vy):
        """
        A class that contains for a simulation:
        - topography:   2D array
        - wd:           3D array
        - vx:           3D array
        - vy:           3D array
        """
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