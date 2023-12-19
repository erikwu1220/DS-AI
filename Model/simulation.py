import matplotlib.animation as animation
import matplotlib.pyplot as plt


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