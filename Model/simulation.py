import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class Simulation():
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

        fig, axs = plt.subplots(1,2,figsize=(14,6))

        axs[0].imshow(self.topography, cmap=cmap_topo, origin="lower")
        axs[1].quiver(self.x, self.y, self.vx[timestep], self.vy[timestep], cmap=cmap_flood)

        return fig


    def plot_animation(self, cmap_topo="terrain", cmap_flood="Blues"):
        fig, axs = plt.subplots(1,2,figsize=(14,6))
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