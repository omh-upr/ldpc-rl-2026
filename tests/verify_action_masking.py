#!/usr/bin/env python3
"""
Checks the "no immediate re-flip" restriction end to end.

Flipping the bit that was just flipped returns the codeword to its previous
value, so the pair spends two of `max_steps` and cannot change the syndrome.
The environment can refuse it. This script checks that the refusal is actually
enforced everywhere it has to be:

  1. The environment narrows its action set, and only when asked to, and
     `step()` refuses the excluded action rather than trusting its caller.
  2. Both branches of epsilon-greedy honour the mask, exploration included.
  3. The bootstrap target drops the action the environment will refuse, which
     is what stops the agent learning a value it can never collect.
  4. A full rollout contains no consecutive repeat of the same flip.
  5. With the flag off, nothing changes.

Run:
    PYTHONPATH=. python tests/verify_action_masking.py
"""
import sys

import numpy as np
import torch

from src.core.environment import LDPCEnvironment
from src.core.ldpc_code import LDPC_Code
from src.core.learning_agent import DQNAgent


CODE_ID = '24x48'
SEED = 12345


def make_env(forbid):
    code = LDPC_Code.from_default(CODE_ID)
    return code, LDPCEnvironment(code=code, max_steps=12, max_flips_vector=7,
                                 forbid_immediate_reflip=forbid)


def a_noisy_word(code, rng, num_errors=4):
    """A codeword with a known number of bit errors."""
    message = rng.integers(0, 2, size=code.k)
    original = code.encode(message)
    noisy = original.copy()
    for bit in rng.choice(code.n, size=num_errors, replace=False):
        noisy[bit] ^= 1
    return noisy, original


def check_environment():
    print("1. environment narrows the action set")
    rng = np.random.default_rng(SEED)
    code, env = make_env(True)
    n = code.n
    noisy, original = a_noisy_word(code, rng)
    env.reset(noisy, original)

    assert env.get_action_mask() is None, "nothing is excluded before the first flip"
    assert len(env.get_legal_actions()) == env.action_dim

    env.step(7)
    mask = env.get_action_mask()
    legal = env.get_legal_actions()
    assert mask is not None and not mask[7], "the bit just flipped must be masked"
    assert mask.sum() == env.action_dim - 1, "exactly one action is excluded"
    assert 7 not in legal, "the bit just flipped must be off the list"
    assert len(legal) == env.action_dim - 1
    assert mask[n] and n in legal, "STOP is never excluded"

    env.step(19)
    mask = env.get_action_mask()
    assert not mask[19], "the restriction follows the most recent flip"
    assert mask[7], "the earlier bit becomes available again"

    env.reset(noisy, original)
    assert env.get_action_mask() is None, "reset clears the restriction"

    # The restriction is the environment's, not a convention the callers keep.
    env.step(7)
    try:
        env.step(7)
    except ValueError as e:
        assert '7' in str(e)
    else:
        raise AssertionError("step() applied a flip the environment had masked")
    print("   masks one action, keeps STOP, follows the latest flip, clears on "
          "reset, and step() refuses the masked action outright")


def check_environment_off():
    print("2. flag off leaves the environment untouched")
    rng = np.random.default_rng(SEED)
    code, env = make_env(False)
    noisy, original = a_noisy_word(code, rng)
    env.reset(noisy, original)
    env.step(7)
    assert env.get_action_mask() is None, "no mask when the flag is off"
    assert env.get_legal_actions() == list(range(env.action_dim))
    env.step(7)
    assert env.get_legal_actions() == list(range(env.action_dim))
    assert env.step_count == 2, "both flips were applied"
    print("   full action set retained, immediate re-flip still permitted")


def check_selection_honours_mask():
    print("3. both epsilon-greedy branches honour the mask")
    code, env = make_env(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     device='cpu', forbid_immediate_reflip=True)

    state = np.zeros(env.state_dim, dtype=np.float32)
    blocked = 11
    mask = np.ones(env.action_dim, dtype=bool)
    mask[blocked] = False

    # Exploring branch: epsilon 1.0 makes every choice random.
    agent.epsilon = 1.0
    picks = {agent.select_action(state, training=True, action_mask=mask)
             for _ in range(4000)}
    assert blocked not in picks, "exploration picked a masked action"
    assert len(picks) == env.action_dim - 1, (
        f"exploration reached {len(picks)} of {env.action_dim - 1} allowed actions")

    # Greedy branch: force the masked action to be the network's favourite.
    agent.epsilon = 0.0
    with torch.no_grad():
        q = agent.network(torch.as_tensor(state).unsqueeze(0))
    forced = int(q.argmax().item())
    mask_forced = np.ones(env.action_dim, dtype=bool)
    mask_forced[forced] = False
    assert agent.select_action(state, training=False) == forced, "control: unmasked argmax"
    picked = agent.select_action(state, training=False, action_mask=mask_forced)
    assert picked != forced, "greedy selection ignored the mask"
    order = torch.argsort(q, dim=1, descending=True)[0].tolist()
    assert picked == order[1], "greedy selection must fall to the next-best action"
    print("   exploration covers exactly the allowed actions, greedy falls to second best")


def check_target_masking():
    print("4. bootstrap target excludes the action the environment will refuse")
    code, env = make_env(True)
    n = code.n
    batch = 32

    def run(forbid, actions):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                         device='cpu', gamma=0.9, forbid_immediate_reflip=forbid)
        rng = np.random.default_rng(SEED)
        states = rng.standard_normal((batch, env.state_dim)).astype(np.float32)
        next_states = rng.standard_normal((batch, env.state_dim)).astype(np.float32)
        rewards = np.zeros(batch, dtype=np.float32)
        dones = np.zeros(batch, dtype=np.float32)
        with torch.no_grad():
            next_q = agent.target_network(torch.from_numpy(next_states))
        metrics = agent.update((states, actions, rewards, next_states, dones))
        return metrics['target_q_mean'], next_q

    # Every transition is produced by the action the target network likes most in
    # the state it led to, so masking has to change the target on every row.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    probe = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     device='cpu', gamma=0.9, forbid_immediate_reflip=True)
    rng = np.random.default_rng(SEED)
    _ = rng.standard_normal((batch, env.state_dim))
    next_states = rng.standard_normal((batch, env.state_dim)).astype(np.float32)
    with torch.no_grad():
        q_next = probe.target_network(torch.from_numpy(next_states))
    best = q_next.argmax(dim=1)
    # Keep them flip actions: STOP ends the episode and is never masked.
    actions = np.where(best.numpy() == n, 0, best.numpy()).astype(np.int64)

    masked_target, next_q = run(True, actions)
    plain_target, _ = run(False, actions)

    top2 = torch.topk(next_q, 2, dim=1)
    expected_masked = []
    for row in range(batch):
        values, idx = top2.values[row], top2.indices[row]
        expected_masked.append(values[1].item() if idx[0].item() == actions[row]
                               else values[0].item())
    expected_masked = 0.9 * float(np.mean(expected_masked))
    expected_plain = 0.9 * float(next_q.max(dim=1).values.mean().item())

    assert abs(masked_target - expected_masked) < 1e-4, (
        f"masked target {masked_target} != expected {expected_masked}")
    assert abs(plain_target - expected_plain) < 1e-4, (
        f"unmasked target {plain_target} != expected {expected_plain}")
    assert masked_target < plain_target - 1e-6, (
        "masking must lower the target when the taken action was the argmax")
    print(f"   masked {masked_target:.5f} vs unmasked {plain_target:.5f}, both exact")


def check_rollouts():
    print("5. rollouts contain no immediate re-flip")
    code, env = make_env(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     device='cpu', forbid_immediate_reflip=True)
    agent.epsilon = 0.5
    rng = np.random.default_rng(SEED)

    repeats = 0
    steps = 0
    for _ in range(300):
        noisy, original = a_noisy_word(code, rng, num_errors=rng.integers(1, 7))
        state = env.reset(noisy, original)
        done = False
        previous = None
        while not done:
            action = agent.select_action(state, training=True,
                                         action_mask=env.get_action_mask())
            if previous is not None and action == previous and action < code.n:
                repeats += 1
            previous = action if action < code.n else None
            state, _, done, _ = env.step(action)
            steps += 1
    assert repeats == 0, f"{repeats} immediate re-flips slipped through"

    # Control: the same rollout without the restriction should produce some.
    _, open_env = make_env(False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    open_agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                          device='cpu', forbid_immediate_reflip=False)
    open_agent.epsilon = 0.5
    rng = np.random.default_rng(SEED)
    open_repeats = 0
    for _ in range(300):
        noisy, original = a_noisy_word(code, rng, num_errors=rng.integers(1, 7))
        state = open_env.reset(noisy, original)
        done = False
        previous = None
        while not done:
            action = open_agent.select_action(state, training=True,
                                              action_mask=open_env.get_action_mask())
            if previous is not None and action == previous and action < code.n:
                open_repeats += 1
            previous = action if action < code.n else None
            state, _, done, _ = open_env.step(action)
    assert open_repeats > 0, (
        "control produced no immediate re-flips, so the check proves nothing")
    print(f"   0 repeats over {steps} restricted steps, {open_repeats} without the flag")


def main():
    print("Verifying the no-immediate-re-flip restriction\n")
    check_environment()
    check_environment_off()
    check_selection_honours_mask()
    check_target_masking()
    check_rollouts()
    print("\nAll checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
