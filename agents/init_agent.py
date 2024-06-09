from agents.dqn import DQN
from agents.kandqn import KANDQN
from agents.playground import Playground
from agents.riiswa_kan_dqn import RiiswaKANDQN

AGENT_DICT = {
    "dqn": DQN,
    "kandqn": KANDQN,
    "riiswa_kan_dqn": RiiswaKANDQN,
    "playground": Playground,
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
