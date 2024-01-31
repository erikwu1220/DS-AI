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
    Y = np.zeros((seq_per_sim*num_sims, H,  channels-1, height, width))
    # Y = np.zeros((seq_per_sim*num_sims, H, channels-1, height, width))

    for i,serie in enumerate(series):
        j = i * seq_per_sim
        for t in range(seq_per_sim):
            if channels == 2:
                X[j+t:j+t+T, :,0,:,:] = np.tile(serie.topography, (T,1,1))
                X[j+t:j+t+T, :,1,:,:] = serie.wd[t:t+T]
                
                # Y[j+t+T : j+t+T+H, :,:,:,:] = serie.wd[t+T+skip:t+T+H+skip]
                Y[0,:,0,:,:] = serie.wd[t+T+skip:t+T+H+skip]

            elif channels == 4:
                X[j+t:j+t+T, :,0,:,:] = np.tile(serie.topography, (T,1,1))
                X[j+t:j+t+T, :,1,:,:] = serie.wd[t:t+T]
                X[j+t:j+t+T, :,2,:,:] = serie.wd[t:t+T]
                X[j+t:j+t+T, :,3,:,:] = serie.wd[t:t+T]

                Y[j+t+T: j+t+T+H, :,0,:,:] = serie.wd[t+T+skip:t+T+H+skip]
                Y[j+t+T: j+t+T+H, :,1,:,:] = serie.vx[t+T+skip:t+T+H+skip]
                Y[j+t+T: j+t+T+H, :,2,:,:] = serie.vy[t+T+skip:t+T+H+skip]

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
    
    outputs[:,0,:,:,:] = inputs[1]
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


def distance_to_nonzero(matrix):
    """"
    Example usage:
    input_matrix = [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]

    result_np = distance_to_nonzero(input_matrix)[:,:,0]
    """
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    # Create an array with indices corresponding to each element in the input matrix
    indices = np.indices((rows, cols), dtype=float)

    # Find the indices where the matrix is nonzero
    nonzero_indices = np.argwhere(matrix != 0)

    # Calculate distances using broadcasting
    distances = np.abs(indices[0][:, :, np.newaxis, np.newaxis] - nonzero_indices[:, 0]) + \
                np.abs(indices[1][:, :, np.newaxis, np.newaxis] - nonzero_indices[:, 1])

    # Find the minimum distance for each element and set the result matrix
    result_matrix = np.min(distances, axis=-1)

    return result_matrix

def distance_torch(matrix):
        rows, cols = matrix.shape

        # Create an array with indices corresponding to each element in the input matrix
        indices = torch.meshgrid(torch.arange(rows), torch.arange(cols))

        # Find the indices where the matrix is nonzero
        nonzero_indices = torch.nonzero(matrix, as_tuple=True)

        # Calculate distances using broadcasting
        distances = torch.abs(indices[0][:, :, None, None] - nonzero_indices[0]) + \
                    torch.abs(indices[1][:, :, None, None] - nonzero_indices[1])

        # Find the minimum distance for each element and set the result matrix
        result_matrix, _ = torch.min(distances, dim=-1)

        return result_matrix
