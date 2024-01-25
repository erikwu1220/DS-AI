import numpy as np
from utils.watertopo import WaterTopo
from utils.simulation import Simulation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_sequence(series, T=5, H=1):
    """
    Create sequence of input X and output Y from a last of Simulation or WaterTopo objects:
    - series: list of Simulation or WaterTopo objects


    - X: input for NN, shape [T * channels * height * width]
    - Y: output for NN [H * height * width]
    """
    if isinstance(series[0], Simulation):
        channels = 4
    elif isinstance(series[0], WaterTopo):
        channels = 2
    else:
        raise Exception("Wrong input type, use a list of Simulations or WaterTopo objects.")
    

    duration = series[0].wd.shape[0]

    height = series[0].wd.shape[1]
    width = series[0].wd.shape[2]

    seq_per_sim = duration-T-H
    num_sims = len(series)

    X = np.zeros((seq_per_sim*num_sims, T, channels, height, width))
    Y = np.zeros((seq_per_sim*num_sims, H, channels-1, height, width))

    for i,serie in enumerate(series):
        j = i * seq_per_sim
        for t in range(seq_per_sim):
            if channels == 2:               
                X[j+t:j+t+T, :,0,:,:] = np.tile(serie.topography, (T,1,1))
                X[j+t:j+t+T, :,1,:,:] = serie.wd[t:t+T]
                
                Y[j+t+T : j+t+T+H, :,:,:,:] = serie.wd[t+T:t+T+H]

            elif channels == 4:
                X[j+t:j+t+T, :,0,:,:] = np.tile(serie.topography, (T,1,1))
                X[j+t:j+t+T, :,1,:,:] = serie.wd[t:t+T]
                X[j+t:j+t+T, :,2,:,:] = serie.wd[t:t+T]
                X[j+t:j+t+T, :,3,:,:] = serie.wd[t:t+T]

                Y[j+t+T: j+t+T+H, :,0,:,:] = serie.wd[t+T:t+T+H]
                Y[j+t+T: j+t+T+H, :,1,:,:] = serie.vx[t+T:t+T+H]
                Y[j+t+T: j+t+T+H, :,2,:,:] = serie.vy[t+T:t+T+H]

    return X, Y