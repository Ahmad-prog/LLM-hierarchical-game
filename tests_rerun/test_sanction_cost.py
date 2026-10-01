"""The manager pays for what it spends (Eq. 2), cannot sanction itself, and budgets are capped."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.enums import MgrType, MgrPowerType  # noqa: E402
from game.manager import Manager  # noqa: E402
from providers.base import LLMResponse  # noqa: E402


class StubProvider:
    def __init__(self, action):
        self.action = action

    def generate(self, *a, **k):
        return LLMResponse(public_message="", structured_action=self.action)


class StubAgent:
    def __init__(self, aid, action=None):
        self.agent_id, self.temperature = aid, 0.7
        self.provider = StubProvider(action or {})

    def build_manager_prompt(self, **k):
        return "prompt"


def run(action):
    agents = [StubAgent("agent_0", action)] + [StubAgent(f"agent_{i}") for i in range(1, 5)]
    m = Manager(mgr_type=MgrType.MGR_FIXED, power_type=MgrPowerType.MGR_FULL, agents=agents)
    contributions = {a.agent_id: 10.0 for a in agents}
    adj, _, _ = m.run_manager_action(1, contributions, {a.agent_id: "" for a in agents}, agents)
    return adj


adj = run({"punish": {"agent_1": 2}, "reward": {"agent_2": 3, "agent_0": 5}})
assert adj["agent_1"] == -6 and adj["agent_2"] == 9, adj
assert adj["agent_0"] == -5, f"manager should pay 2 + 3 = 5 (self-reward ignored), got {adj['agent_0']}"
adj = run({"punish": {"agent_1": 8, "agent_2": 8}})
assert adj["agent_0"] == -10 and adj["agent_1"] == -24 and adj["agent_2"] == -6, adj
print("sanction cost tests passed:", adj)
