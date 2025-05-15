#!/usr/bin/env python3
from barcgp.common.utils.custom_file_utils import *
import numpy as np
import gpytorch
from barcgp.common.utils.custom_scenario_utils import SampleGenerator, Sample
from barcgp.prediction.gpytorch_models import ExactGPModel, MultitaskGPModel, MultitaskGPModelApproximate, \
    IndependentMultitaskGPModelApproximate
from barcgp.prediction.gp_controllers import GPControllerApproximate
import time

policy_name = 'aggressive_blocking'
policy_dir = os.path.join(train_dir, policy_name)
scencurve_dir = os.path.join(policy_dir, 'curve')
scenstraight_dir = os.path.join(policy_dir, 'straight')
scenchicane_dir = os.path.join(policy_dir, 'chicane')
# with open("/home/jeanho/gp-opponent-prediction-models-main/sample_batch", 'rb') as f:
#     loaded_sample = pickle.load(f)

# Training
def main():
    start  = time.time()
    sampGen = SampleGenerator("/home/jeanho/gp-opponent-prediction-models-main/sample_batch", randomize=True)
    #sampGen.plotStatistics('c')
    # sampGen.even_augment('s', 0.3)
    # if not dir_exists(scencurve_dir):
    #     raise RuntimeError(
    #         f"Directory: {scencurve_dir} does not exist, need to train using `gen_training_data` first")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=5)  # should be same as output_size
    gp_controller = GPControllerApproximate(sampGen, IndependentMultitaskGPModelApproximate, likelihood,
                                            input_size=11, output_size=5, inducing_points=200, enable_GPU=True)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood
    # gp_controller = GPControllerExact(sampGen, ExactGPModel, likelihood, 10, 5, True)

    gp_controller.train()
    gp_controller.set_evaluation_mode()
    trained_model = gp_controller.model, gp_controller.likelihood

    create_dir(path=model_dir)
    gp_controller.save_model(policy_name)
    gp_controller.evaluate()
    end = time.time()
    print("Time taken to train the model: ", end-start) 


if __name__ == "__main__":
    main()
