
import torch
import math
from botorch.utils.transforms import normalize

from src.bayesian.training import fit_model
from src.params import load_param_space

def encode_params(params_dict, space):
    """
    Converts a dictionary of raw hyperparameters into a normalized tensor [0, 1].
    """
    raw_values = []
    
    for p in space.params:
        val = params_dict[p.name]
        
        # 1. Map to internal raw scale based on ParamDef types
        if p.type == "log_float":
            # Convert to log10 space
            raw_val = math.log10(val)
        elif p.type == "categorical":
            # Convert choice string/value to its integer index
            raw_val = float(p.choices.index(val))
        else:
            # float and int are already on the correct raw scale
            raw_val = float(val)
            
        raw_values.append(raw_val)
    
    # 2. Create the tensor
    x_raw = torch.tensor(raw_values, dtype=torch.double).unsqueeze(0)
    
    # 3. Normalize to [0, 1] using the bounds calculated in ParamSpace
    x_norm = normalize(x_raw, space.bounds_raw)
    
    return x_norm

def main(): 
    # 1. Load the state and space
    state = torch.load("path/to/state_iterXXX.pt")
    space = load_param_space(config_path="hpo_config.yaml")

    # 2. Fit the model using historical data from the state
    model = fit_model(state["train_x"], state["train_y"], space.n_objectives)

    # 3. Prepare your test parameters (as a normalized tensor)
    # This uses the ParamSpace logic to ensure parameters are in [0, 1]
    test_params_raw = {"learning_rate": 0.001, "batch_size": 32}
    # (You would need a helper to encode back to tensor if not already available)
    test_x_tensor = encode_params(test_params_raw, space)

    # 4. Get the prediction
    with torch.no_grad():
        posterior = model.posterior(test_x_tensor)
        means = posterior.mean  # This contains the [IoU, -Energy] estimates
        print(means)


if __name__ == "__main__": 
    main()