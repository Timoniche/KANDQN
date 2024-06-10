from agents.dqn import DQN
from agents.kaqn import KAQN
from agents.riiswa_kaqn import RiiswaKAQN

AGENT_DICT = {
    "dqn": DQN,
    "riiswa_kaqn": RiiswaKAQN,
    "kaqn": KAQN,
}


def init_agent(
        training_args,
        n_actions,
        n_observations,
        device,
):
    agent_kwargs = (
        {} if training_args["agent_kwargs"] is None else training_args["agent_kwargs"]
    )

    agent_kwargs['n_actions'] = n_actions
    agent_kwargs['n_observations'] = n_observations
    agent_kwargs['device'] = device

    agent = AGENT_DICT[training_args["agent"]](**agent_kwargs)

    return agent
