"""Regression checks for target synchronization and update-metric attribution.

Run from the project root:

    PYTHONPATH=. python tests/verify_training_updates.py

Exercises the real LDPCTrainer, LDPCEnvironment, DQNAgent and ReplayBuffer over
a stub data generator, on the 24x48 code and on CPU, and asserts:

  B01 - the target network is synchronized on exactly the updates whose index is
        a multiple of target_update_freq, whatever the episode lengths are.
        Before the fix, synchronization was tested only at episode boundaries,
        so most multiples were stepped over.
  B10 - every gradient update is recorded on a step of the episode that
        performed it, in position. Before the fix, each update was attached to
        the following step, and an episode's trailing update leaked into the
        next episode.

Exits nonzero on the first failed assertion.
"""
import numpy as np
import torch

from src.core.ldpc_code import LDPC_Code
from src.core.environment import LDPCEnvironment
from src.core.learning_agent import DQNAgent
from src.training.replay_buffer import ReplayBuffer
from src.training.training import LDPCTrainer
from src.evaluation.metrics_callbacks import EpisodeDataCollector


class StubGenerator:
    """Minimal stand-in for RobustDataGenerator."""

    def __init__(self, code, num_samples, rng):
        self.config = {'curriculum': [{'phase_name': 'P1', 'error_models': [
            {'type': 'uniform_count', 'params': {'num_errors': 2}}]}]}
        self.samples = []
        n = code.n
        for _ in range(num_samples):
            original = np.zeros(n, dtype=np.int8)
            noisy = original.copy()
            error = np.zeros(n, dtype=np.int8)
            for pos in rng.choice(n, size=2, replace=False):
                noisy[pos] ^= 1
                error[pos] = 1
            self.samples.append({
                'original': original, 'noisy': noisy, 'error': error,
                'meta': {'num_errors': 2, 'error_model': 'uniform_count'},
                'phase': 1,
            })

    def get_num_phases(self):
        return 1

    def get_phase_info(self):
        return {'num_phases': 1}

    def get_iterator(self, split):
        for s in self.samples:
            yield s

    def get_phase_iterator(self, split, phases=None, max_samples=None):
        return iter(self.samples[:max_samples or 0])


def build(target_update_freq, update_frequency, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    code = LDPC_Code.from_default(code_id='24x48')
    env = LDPCEnvironment(code=code, max_steps=8)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     epsilon_start=0.9, epsilon_decay=0.99, epsilon_min=0.05,
                     network_config={'type': 'SimpleMLP'}, device='cpu')
    buffer = ReplayBuffer(capacity=5000, code=code)
    gen = StubGenerator(code, num_samples=60, rng=rng)
    collector = EpisodeDataCollector()
    trainer = LDPCTrainer(env, agent, buffer, gen,
                          batch_size=8, target_update_freq=target_update_freq,
                          update_frequency=update_frequency,
                          min_replay_size=8, collectors=[collector])
    return trainer, agent, collector


def run(target_update_freq, update_frequency):
    trainer, agent, collector = build(target_update_freq, update_frequency)

    syncs_at = []
    real_sync = agent.update_target_network
    real_update = agent.update
    update_index = {'n': 0}
    losses = []

    def traced_update(batch):
        metrics = real_update(batch)
        update_index['n'] += 1
        losses.append((update_index['n'], metrics['loss']))
        return metrics

    def traced_sync():
        syncs_at.append(update_index['n'])
        return real_sync()

    agent.update = traced_update
    agent.update_target_network = traced_sync

    # Record the half-open range of update indices each episode performed, so a
    # recorded update can be traced back to the episode it belongs to.
    episode_spans = []
    real_episode = trainer.train_episode

    def traced_episode(sample, training=True, **kw):
        start = update_index['n']
        data = real_episode(sample, training=training, **kw)
        if training:
            episode_spans.append((start, update_index['n']))
        return data

    trainer.train_episode = traced_episode

    trainer.train(num_epochs=1, eval_frequency=10**9, eval_episodes=1)
    return trainer, collector, syncs_at, losses, update_index['n'], episode_spans


def check_b01(freq, update_frequency):
    trainer, _, syncs_at, _, total_updates, _ = run(freq, update_frequency)
    expected = list(range(freq, total_updates + 1, freq))
    assert syncs_at == expected, (
        f"freq={freq} uf={update_frequency}: synced at {syncs_at}, expected {expected}")
    assert trainer.training_steps == total_updates, (
        f"counter {trainer.training_steps} != {total_updates} actual updates")
    print(f"  B01 ok: freq={freq} update_frequency={update_frequency} -> "
          f"{total_updates} updates, {len(syncs_at)} syncs (expected {len(expected)})")


def check_b10():
    """Every update must be recorded on a step of the episode that performed it,
    in position. A shift by one step is exactly what B10 described, and it
    survives a count-only check because each episode's trailing update simply
    lands on the next episode's first step."""
    _, collector, _, losses, total_updates, spans = run(target_update_freq=7,
                                                        update_frequency=1)
    index_of_loss = {loss: i for i, loss in losses}
    assert len(index_of_loss) == len(losses), "losses are not distinguishable"

    recorded_per_episode = []
    for episode in collector.episodes:
        indices = [index_of_loss[step['training_update']['loss']]
                   for step in episode['steps']
                   if step.get('training_update') is not None]
        recorded_per_episode.append(indices)

    assert len(recorded_per_episode) == len(spans), (
        f"{len(recorded_per_episode)} episodes collected, {len(spans)} run")

    total_recorded = 0
    for position, (indices, (start, end)) in enumerate(zip(recorded_per_episode, spans)):
        expected = list(range(start + 1, end + 1))
        assert indices == expected, (
            f"episode {position}: recorded updates {indices}, but the episode "
            f"performed updates {expected}")
        total_recorded += len(indices)

    assert total_recorded == total_updates, (
        f"{total_recorded} updates recorded, {total_updates} performed")
    print(f"  B10 ok: {total_updates} updates, each recorded on a step of the "
          f"episode that performed it, none shifted or lost")


def main():
    print("Training update regression checks")
    for freq, uf in ((7, 1), (3, 1), (5, 2), (1, 1)):
        check_b01(freq, uf)
    check_b10()
    print("all checks passed")


if __name__ == '__main__':
    main()
