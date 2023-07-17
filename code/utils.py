import random


def gen_random_prob(actions):
    total = 1
    probs = []
    for _ in actions[:-1]:
        prob = random.uniform(0, total)
        probs.append(prob)
        total -= prob
    probs.append(total)
    return probs

def deep_replace(inst, replace_map):
    if not isinstance(inst, list):
        return replace_map(inst)

    return [deep_replace(i, replace_map) for i in inst]


if __name__ == "__main__":
    print(gen_random_prob([1]))