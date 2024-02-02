import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
import seaborn as sns

def performance(prediction,Real,*args,**kwargs):
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
        acc = 1-np.divide(abs((prediction[t] - Real[t])),Real[t])
        plt.figure(figsize=(6, 6))
        plt.imshow(acc, cmap='RdBu', origin='lower')
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = 0, vmax=1),
                                       cmap='RdBu'), fraction=0.05, shrink=0.9)
        plt.title("Accuracy Map")

    if args[0] == 'animation':
        print("Creating an animation with gif")
        fig = plt.figure()

        savepath = kwargs["save_path"]
        ims = []

        for i in range(len(prediction)):
            # Calculate accuracy
            acc = 1-np.divide(abs((prediction[i] - Real[i])),Real[i])
            ttl = plt.text(60, 5, f"timestep ={i*6}", horizontalalignment='right', verticalalignment='bottom')
            im = plt.imshow(acc,cmap='inferno_r', animated=True, origin='lower')

            ims.append([im,ttl])
        ni = ArtistAnimation(fig, ims, interval=1)

        # Srtting Figure Information
        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = 0, vmax=1),
                                       cmap='inferno_r'), fraction=0.05, shrink=0.9)

        plt.title("Accuracy Map")
        # Srtting GIF Information
        fps = 3
        writer = PillowWriter(fps=fps)
        ni.save(savepath, writer = writer)

        plt.show()
        print("done")

def confusionmatrix(prediction,real,timestep):
    '''
    This function is used to built up the confusion matrix.
    Pixel with water and without water will be classified into two different classes, and
    from which calculate the accuracy, recall, and precisin.
    Note that this function can not tell the accuracy of water depth within the specific pixel.
    To better quantitatively study the model behaviours, please use "Performance" function.

    prediction: prediction from the model
    real: real value
    timestep: specific timestep for confusion matrix
    '''
    pre = prediction[timestep].reshape(-1)
    pre[pre <= 0] = 0
    pre[pre > 0] = 1

    rea = real[timestep].reshape(-1)
    rea[rea <= 0] = 0
    rea[rea>0] = 1

    conf_matrix = confusion_matrix(rea, pre)
    accuracy = accuracy_score(rea, pre)
    recall = recall_score(rea, pre)
    precision = precision_score(rea, pre)
    f1 = f1_score(rea, pre)

    plt.figure(figsize=(3, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Results
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")