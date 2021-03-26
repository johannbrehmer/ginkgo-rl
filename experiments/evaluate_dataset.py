import sys
import numpy as np
import logging
import torch

sys.path.append("../")
from ginkgo_rl import GinkgoRLInterface, PolicyMCTSAgent

logger = logging.getLogger("We")
logging.basicConfig(format="%(asctime)s %(levelname).1s: %(message)s", datefmt="%y-%m-%d %H:%M")

silence_list = ["matplotlib", "ginkgo", "ClusterTrellis"]
for key in logging.Logger.manager.loggerDict:
    logging.getLogger(key).setLevel(logging.INFO)
    for check_key in silence_list:
        if check_key in key:
            logging.getLogger(key).setLevel(logging.ERROR)
            break


settings = {
    "n_max": 40,
    "n_min": 4,
    "w_jet": False,
    "w_rate": 3.0,
    "qcd_rate": 1.5,
    "pt_min": 1.1**2,
    "qcd_mass": 30.0,
    "w_mass": 80.0,
    "jet_momentum": 400.0,
    "jetdir": (1, 1, 1),
    "beamsize": 20,
    "n_mc_target": 2,
    "n_mc_max": 50,
    "device": torch.device("cpu"),
}


state_dict_filename = "./data/runs/mcts_nn_xs_20210324_154917_1003/model.pty"

grl = GinkgoRLInterface(state_dict_filename, **settings)

jets = grl.generate(n=1)

clustered_jets, log_likelihoods, illegal_actions, costs = grl.cluster(jets, filename="reclustered_jets.pickle")

print(" log_likelihoods, illegal_actions, costs = ",  log_likelihoods, illegal_actions, costs)