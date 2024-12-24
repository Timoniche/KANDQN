from agents.dqn import DQN
from agents.efficient_kaqn import EfficientKAQN
from agents.fastkaqn import FASTKAQN
from agents.kaqn import KAQN
from agents.riiswa_kaqn import RiiswaKAQN
from agents.simple_riiswa_kaqn import SimpleRiiswaKAQN

AGENT_DICT = {
    "dqn": DQN,
    "riiswa_kaqn": RiiswaKAQN,
    "kaqn": KAQN,
    "fkaqn": FASTKAQN,
    "efficient_kaqn": EfficientKAQN,
    "simple_riiswa_kaqn": SimpleRiiswaKAQN,
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
