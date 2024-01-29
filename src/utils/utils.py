import numpy as np
import torch
from utils.watertopo import WaterTopo
from utils.simulation import Simulation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_sequence(series, T=5, H=1, skip=0):
    """
    Create sequence of input X and output Y from a last of Simulation or WaterTopo objects:
    - series: list of Simulation or WaterTopo objects
    - T: int, amount of timesteps to use as input
    - H: int,  amount of timesteps that are predicted in the output
    - skip: int, step size between timesteps

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

    seq_per_sim = duration-T-H+1-skip
    num_sims = len(series)

    X = np.zeros((seq_per_sim*num_sims, T, channels, height, width))
    Y = np.zeros((seq_per_sim*num_sims, H, channels-1, height, width))

    for i,serie in enumerate(series):
        j = i * seq_per_sim
        for t in range(seq_per_sim):
            if channels == 2:
                X[j+t:j+t+T, :,0,:,:] = np.tile(serie.topography, (T,1,1))
                X[j+t:j+t+T, :,1,:,:] = serie.wd[t:t+T]
                
                Y[j+t+T : j+t+T+H, :,:,:,:] = serie.wd[t+T+skip:t+T+H+skip]

            elif channels == 4:
                X[j+t:j+t+T, :,0,:,:] = np.tile(serie.topography, (T,1,1))
                X[j+t:j+t+T, :,1,:,:] = serie.wd[t:t+T]
                X[j+t:j+t+T, :,2,:,:] = serie.wd[t:t+T]
                X[j+t:j+t+T, :,3,:,:] = serie.wd[t:t+T]

                Y[j+t+T: j+t+T+H, :,0,:,:] = serie.wd[t+T:t+T+H]
                Y[j+t+T: j+t+T+H, :,1,:,:] = serie.vx[t+T:t+T+H]
                Y[j+t+T: j+t+T+H, :,2,:,:] = serie.vy[t+T:t+T+H]
    print(X.shape)
    print(Y.shape)
    return X, Y


def recursive_pred(model, inputs, timesteps, include_first_timestep=False):
    """
    Recursively predicts next time step given a certain input.
    - model: Neural network
    - inputs: array-like: first time-step, including topography. Shape should be [channels x height x width]
    - timesteps: int, number of timesteps to predict.
    - include_first_timestep: bool, whether to include the input timestep in the output.
    
    returns:
    - mse: model output at everytimestep, with shape [timesteps x channels x height x width]
    """   
    outputs = torch.zeros([1, timesteps+1, 2, inputs.shape[-2], inputs.shape[-1]])

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    
    outputs[:,:,0,:,:] = torch.tile(inputs[0], dims=[timesteps+1,1,1])

    for t in range(1, timesteps):
        outputs[:,t,1,:,:] = model(outputs[:,t-1,:,:,:])

    if not include_first_timestep:
        outputs = outputs[:,1:,:,:,:]

    return outputs[0,:,1,:,:]


def mse_per_timestep(targets, outputs):
    """
    Calculates the MSE for each timestep between the targets and outputs.
    - targets: numpy-array, model targets
    - outputs: torch-array, model predictions
    
    returns:
    - mse: numpy array with MSE of each timestep
    """
    from sklearn.metrics import mean_squared_error
    # targets = targets.squeeze()
    
    timesteps = outputs.shape[0]
    mse = np.zeros(timesteps)
    for t in range(timesteps):
        mse[t] = mean_squared_error(targets[t], outputs[t].detach().numpy())

    return mse

def get_corner(wd):
    """
    in:
    - wd: 2D array

    returns:
    - corner index, with [top_left, top_right, bottom_left, bottom_right] = [1, 2, 3, 4]
    """
    idx = np.argmax(wd) + 1
    grid_size = wd.shape[0]

    if idx == 1:
        corner = 1
    elif idx == grid_size:
        corner = 2
    elif idx == grid_size**2 - grid_size + 1:
        corner = 3
    else:
        corner = 4
    return corner