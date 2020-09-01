import sys
import logging
from line_profiler import LineProfiler

sys.path.append("../../")
from ginkgo_rl import GinkgoLikelihood1DEnv
from ginkgo_rl import MCTSAgent
from ginkgo_rl import GinkgoEvaluator

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
        datefmt='%H:%M',
        level=logging.DEBUG
    )
    for key in logging.Logger.manager.loggerDict:
        if "ginkgo_rl" not in key:
            logging.getLogger(key).setLevel(logging.WARNING)
    logger.info("Hi!")

    # Set up env, model and evaluator
    env = GinkgoLikelihood1DEnv()
    model = MCTSAgent(env)
    evaluator = GinkgoEvaluator(filename="temp.pickle", redraw_existing_jets=True, n_jets=3)

    # Profile
    lp = LineProfiler()
    lp.add_function(model._mcts)
    lp.add_function(model._parse_path)
    lp.add_function(model._evaluate_policy)
    lp.add_function(model._batch_state)
    lp.add_function(model._parse_action)

    lp(evaluator.eval)("MCTS", model, "GinkgoLikelihood1D-v0", n_repeats=1)

    # Results
    logger.info("Line profiling results:")
    lp.print_stats()
    logger.info("All done! Have a nice day!")
