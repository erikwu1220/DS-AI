import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import animation
import numpy as np


def animation_create(savepath,X,Y,Z,wd,N,fps = 10, color_dem=cm.gist_earth, mask_threshold=0):
    """
    Domanin size and of the topography and water depth should be the same.
    savepath: saving the animation to direct folder
    X: Input Dem terrain data for animation (N x N)
    Y: Input Dem trrrain data for animation (N x N)
    Z: Input Dem terrain data for animation (N x N)
    wd: Input water depth grid data as ( t x N x N)
    N: grid size

    """
    h = []
    newh = []



    # Create the modified water depth
    for i in range(len(wd)):
        h.append(wd[i] + Z)
        mask = np.zeros((N,N))
        mask[wd[i] != 0] = 1
        newh.append(mask * h[i])
        newh[i][newh[i] <= mask_threshold] = np.nan

    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    frn = len(wd)
    

    def change_plot(frame_number, newh, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, newh[frame_number], rstride=1, cstride=1, cmap = 'Blues',linewidth=0, vmin=np.min(h), vmax=np.max(h))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = [ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap =  color_dem, linewidth=0, edgecolor='none',alpha = 1)]
    plot = [ax.plot_surface(X, Y, newh[0], rstride=1, cstride=1, cmap = 'Blues',linewidth=0, vmin=np.min(h), vmax=np.max(h))]

    # Angle of showing the animation
    ax.view_init(60, 60)

    # Set the Title, labels, and fontsize
    ax.set_title('Flooding Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.xaxis.get_label().set_fontsize(13)
    ax.yaxis.get_label().set_fontsize(13)
    ax.zaxis.get_label().set_fontsize(13)
    ax.title.set_fontsize(13)


    # converts the values of any array to RGB colors defined by a colormap
    m = cm.ScalarMappable(cmap='Blues')
    m.set_array(newh[0])

    # Set the colorbar
    plt.colorbar(m,fraction=0.02)
    ani = animation.FuncAnimation(fig, change_plot, frn, fargs=(newh, plot), interval=1)

    plt.show()

    writergif = animation.PillowWriter(fps=fps)
    ani.save(savepath, writer=writergif)
    print("done")

    return