# n_test = 10
# env = gym.make("GinkgoLikelihood-v0")
#
# val_internal_states = []
# val_log_likelihoods = []
#
# for _ in range(n_test):
#     env.reset()
#     val_internal_states.append(env.get_internal_state())
#     val_log_likelihoods.append(sum(env.jet["logLH"]))
#
# env.close()
#
#
# class GinkgoEvalCallback(BaseCallback):
#     def __init__(self, eval_env, eval_freq=100, verbose=0):
#         super(GinkgoEvalCallback, self).__init__(verbose)
#
#         self.eval_env = eval_env
#         self.eval_env.min_reward = -1000.0
#         self.eval_freq = eval_freq
#
#         self.steps = []
#         self.log_likelihoods = []
#         self.errors = []
#
#     def _on_step(self):
#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             log_likelihood = 0.
#             errors = 0.
#
#             for internal_state in test_internal_states:
#                 _ = self.eval_env.reset()
#                 self.eval_env.set_internal_state(internal_state)
#                 state = self.eval_env.get_state()
#                 done = False
#                 steps = 0
#
#                 while not done and steps < int(1.e5):
#                     action, _ = self.model.predict(state)
#                     state, reward, done, info = self.eval_env.step(action)
#
#                     steps += 1
#                     if info["legal"]:
#                         log_likelihood += reward / n_test
#                     else:
#                         errors += 1. / n_test
#
#             self.steps.append(self.n_calls)
#             self.log_likelihoods.append(log_likelihood)
#             self.errors.append(errors)
#             print(log_likelihood, errors)
#         return True
#
