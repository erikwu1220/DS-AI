import numpy as np
import matplotlib.pyplot as plt

def Performance(pred_WD,True_WD,timestep,):
    '''
    This function is used to present the accuracy of the pridicted result.
    The accuracy is based on computing the diffference of predicted water depth and true water depth of each pixel.
    In view of the temporal evolution, perfect model means that accuracy of each pixel should be converged to 100% through time.
    '''


    diff_WD = np.divide((True_WD[timestep] - pred_WD[timestep]),True_WD[timestep])
    plt.figure(figsize=(8, 8))
    plt.imshow(diff_WD, cmap='RdBu', origin='lower')
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = demx.min(), vmax=demx.max()),
                                       cmap='RdBu_r'), fraction=0.05, shrink=0.9, ax=axs[0])

    # if animation:



    # # Results
    # print("Confusion Matrix:\n", conf_matrix)
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score: {f1:.4f}")