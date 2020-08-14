import torch

def ginkgo1d_legal_action_extractor(state, env):
    particles = [i for i, p in enumerate(state) if torch.max(p) > 0]

    actions = []
    try:  # 1D-wrapped envs
        for i, pi in enumerate(particles):
            for j, pj in enumerate(particles[:i]):
                actions.append(env.wrap_action((pi, pj)))
    except:
        for i, pi in enumerate(particles):
            for j, pj in enumerate(particles[:i]):
                actions.append((pi, pj))

    return actions
