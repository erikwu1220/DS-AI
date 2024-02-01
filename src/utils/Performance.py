import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
def Performance(prediction,Real,*args,**kwargs):
    '''
    This function is used to present the accuracy of the pridicted result.
    The accuracy is based on computing the diffference of predicted water depth and true water depth of each pixel.
    In view of the temporal evolution, perfect model means that accuracy of each pixel should be converged to 100% through time.
    '''
    prediction = np.array(prediction)
    Real = np.array(Real)


    if args[0] == 'specific':
        print("Selecting specific timestep")
        t = kwargs["timestep"]

        # Calculate accuracy
        acc = 1-np.divide((prediction[t] - Real[t]),Real[t])
        plt.figure(figsize=(6, 6))
        plt.imshow(acc, cmap='RdBu_r', origin='lower')
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = 0, vmax=1),
                                       cmap='RdBu_r'), fraction=0.05, shrink=0.9)
        plt.title("Accuracy Map")

    if args[0] == 'animation':
        print("Creating an animation with gif")
        fig = plt.figure()

        savepath = kwargs["save_path"]
        ims = []

        for i in range(len(prediction)):
            # Calculate accuracy
            acc = 1-np.divide((prediction[i] - Real[i]),Real[i])

            im = plt.imshow(acc,cmap='RdBu_r', animated=True, origin='lower')
            ims.append([im])
        ni = ArtistAnimation(fig, ims, interval=1)

        # Srtting Figure Information
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = 0, vmax=1),
                                       cmap='RdBu_r'), fraction=0.05, shrink=0.9)
        plt.title("Accuracy Map")

        # Srtting GIF Information
        fps = 5
        writer = PillowWriter(fps=fps)
        ni.save(savepath, writer = writer)

        plt.show()
        print("done")