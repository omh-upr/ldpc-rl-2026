# Master Design

## 1. Purpose and Scope

This document gives a high-level, code-accurate description of the research problem implemented in this repository: hard-decision LDPC decoding formulated as a reinforcement learning problem. It is intended to help a researcher understand the mathematical object being solved, the Markov decision process used in code, and the learning architecture used to approximate the resulting policy.

This document does not describe the full experiment workflow, dataset orchestration, artifact generation, or the flagship `64x128_Full` run configuration in detail. Those belong in [MASTER_EXPERIMENT_DESIGN.md](MASTER_EXPERIMENT_DESIGN.md).

## 2. Research Problem

The repository studies decoding of binary linear LDPC codes by sequential bit flipping under a hard-decision observation model.

Given:

- A parity-check matrix \(H \in \{0,1\}^{m \times n}\)
- A transmitted codeword \(x \in \{0,1\}^n\) satisfying \(Hx = 0 \mod 2\)
- A corrupted received vector \(y = x \oplus e\), where \(e\) is an unknown binary error pattern

the goal is to recover a valid codeword by applying a sequence of bit-flip actions to the current estimate. The agent does not output soft reliabilities or a full decoded codeword in one step. Instead, it interacts with an environment that exposes the current decoding state and asks the agent to choose one bit to flip, or to stop.

This is therefore a sequential decision problem over partial decoding trajectories rather than a one-shot classification problem.

## 3. MDP Formulation

### 3.1 Episode Definition

Each episode starts from one noisy codeword sampled from a pre-generated dataset. The environment receives both:

- `noisy_codeword`: the agent-visible corrupted input
- `original_codeword`: the ground-truth codeword, used internally for BER reporting and omniscient expert generation

The decoder then iteratively modifies the current estimate until it chooses `STOP` or violates episode constraints.

### 3.2 State

The implemented observation is a flat real-valued vector composed of:

1. Current codeword estimate
2. Current syndrome
3. Flip-history vector
4. Penalty vector derived from the syndrome
5. Metadata: syndrome weight, penalty sum, step count

Formally,

\[
s_t = [\hat{x}_t,\ \sigma_t,\ f_t,\ p_t,\ w(\sigma_t),\ \sum_i p_t(i),\ t]
\]

where:

- \(\hat{x}_t \in \{0,1\}^n\) is the current codeword estimate
- \(\sigma_t = H\hat{x}_t \mod 2 \in \{0,1\}^m\) is the syndrome
- \(f_t \in \{0,1\}^n\) records which positions have been flipped an odd number of times
- \(p_t \in \mathbb{R}^n\) is the penalty vector computed from the current syndrome

The total state dimension is:

\[
|s_t| = 3n + m + 3
\]

For the main `64x128` code:

- \(n = 128\)
- \(m = 64\)
- state dimension = \(3 \cdot 128 + 64 + 3 = 451\)

### 3.3 Action Space

The action space is discrete:

\[
\mathcal{A} = \{0,1,\dots,n-1,\text{STOP}\}
\]

- Actions `0` through `n-1` flip the corresponding bit in the current estimate.
- Action `n` is the explicit `STOP` action.

Therefore the action dimension is:

\[
|\mathcal{A}| = n + 1
\]

For `64x128`, the action dimension is `129`.

The environment can optionally restrict this set. With `forbid_immediate_reflip` enabled, the bit flipped at step \(t-1\) is not a legal action at step \(t\): flipping it back restores \(\hat{x}_{t-1}\) exactly, so the pair spends two steps and cannot change the syndrome. The restriction is exposed to every agent through the legal-action list, and the learned agent also excludes that action from the bootstrap maximum in Section 5.1. It changes the MDP, so it is off by default.

### 3.4 Transition Dynamics

If the selected action is a bit index \(a < n\), the environment toggles bit \(a\) in the current estimate, recomputes the syndrome, recomputes the penalty vector, updates the flip-history vector, and increments the step counter.

If the selected action is `STOP`, the state is not further modified and the episode terminates immediately.

Conditioned on the current state and the selected action, the transition is deterministic.

### 3.5 Termination and Success

An episode terminates when either:

- the agent selects `STOP`
- the maximum number of steps is reached
- the total number of flips exceeds the configured limit

The implemented notion of successful decoding is stricter than merely reaching zero syndrome at an intermediate state. In the current code, success is registered only when the agent explicitly selects `STOP` and the terminal state satisfies:

\[
w(\sigma_t) = 0 \quad \text{and} \quad w(f_t) \leq F_{\max}
\]

where \(F_{\max}\) is `max_flips_vector`.

This explicit-stop design matters: the agent must learn both how to repair the word and when to terminate.

### 3.6 Reward

The default reward function is `ImprovedRewardFunction`. It is intentionally sparse and avoids dense per-action "good flip / bad flip" shaping.

Its components are:

- terminal success bonus
- explicit `STOP` reward or penalty
- per-step penalty
- exponential penalty for exceeding the nominal correction regime through too many flips

With base value \(b = 5\), the implemented reward is:

\[
r_t =
  25 \cdot \mathbf{1}[\text{done} \land w(\sigma_{t+1}) = 0] + 5 \cdot \mathbf{1}[a_t=\text{STOP} \land w(\sigma_{t+1})=0] - 7.5 \cdot \mathbf{1}[a_t=\text{STOP} \land w(\sigma_{t+1})>0] - 0.75 - 0.75 \exp(\min(w(f_{t+1}) - c,\ 20))
\]

where \(c\) is the LDPC code's configured correction capability.

Interpretation:

- Correct decoding is primarily signaled through terminal reward.
- `STOP` is trained explicitly instead of being treated as a passive environment-side check.
- Efficiency matters, but only as a secondary pressure.
- Excessive flipping is discouraged without forcing a greedy syndrome-reduction policy.

**Example**: For `64x128` (\(c = 6\), \(b = 5\)), consider an episode where the agent makes 3 correct flips then selects STOP (4 actions total, each incurring step and flip penalties):

- Step penalties: \(4 \times (-0.75) = -3.0\)
- Flip penalties (cumulative across all 4 steps): \(-0.75(e^{-5} + e^{-4} + e^{-3} + e^{-3}) \approx -0.09\)
- Terminal success bonus: \(+25\)
- STOP correct bonus: \(+5\)
- Total episode return: \(\approx +26.9\)

The exponential flip penalty is negligible at 3 flips (\(c - 3 = 3\) margin), but escalates sharply past the correction capability: at \(w(f_t) = 8\), the single-step flip penalty reaches \(-0.75 \cdot e^2 \approx -5.5\).

## 4. Why RL Is Used Here

Classical bit-flipping heuristics typically rely on local scores such as syndrome participation or penalty values. This project instead treats decoding as a sequential control problem in which local greedy decisions may be suboptimal. The RL formulation is meant to learn policies that account for:

- delayed effects of a flip on future parity consistency
- interactions across multiple parity checks
- the need to stop only when the trajectory has plausibly converged

In this framing, the agent is not learning the code itself. It is learning a policy over the handcrafted decoding state induced by that code.

## 5. Learning Formulation

The implemented learning agent is a DQN-style value-based agent:

- a neural network \(Q(s,a;\theta)\) approximates the optimal action-value function
- a target network \(Q(s,a;\theta^-)\) stabilizes bootstrapped learning
- experience replay stores transition tuples \((s_t, a_t, r_t, s_{t+1}, d_t)\)
- updates minimize temporal-difference error on sampled mini-batches

### 5.1 Loss and Target Computation

The Q-network parameters \(\theta\) are updated by minimizing the mean squared TD error over a mini-batch \(\mathcal{B}\) sampled from the replay buffer:

\[
L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s',d) \in \mathcal{B}} \left( y - Q(s,a;\theta) \right)^2
\]

where the target \(y\) is:

\[
y = \begin{cases}
r & \text{if } d = 1 \text{ (terminal)} \\
r + \gamma \max_{a'} Q(s', a'; \theta^-) & \text{if } d = 0
\end{cases}
\]

When the re-flip restriction of Section 3.3 is enabled, the maximum over \(a'\) excludes the action \(a\) that produced the transition, since it is illegal in \(s'\). No mask has to be stored with the transition: the excluded action is the one already in the tuple.

The target network parameters are synchronized periodically:

\[
\theta^- \leftarrow \theta \quad \text{every } C \text{ training steps}
\]

The discount factor \(\gamma\) is set close to 1 (default 0.99) to encourage long-horizon planning over multi-step decoding trajectories. The optimizer is Adam with configurable learning rate and weight decay.

### 5.2 Curriculum and Expert Replay Seeding

Training is off-policy and curriculum-driven. The dataset is organized into six sequential phases of increasing error difficulty, and the agent trains on each phase in order, optionally for several epochs; each epoch restarts at phase 1. Exploration is controlled by per-phase epsilon schedules: each phase has its own \(\epsilon_{\text{start}}\), \(\epsilon_{\text{decay}}\), and \(\epsilon_{\min}\), and epsilon is optionally reset to \(\epsilon_{\text{start}}\) at each phase boundary. Each epoch may carry its own row of these values, and is always re-seeded to its own opening value. Two forms of decay exist: an explicit per-update decrement, and `auto_episodes`, which sets epsilon at the start of every episode from the fraction of the phase completed, falling linearly to \(\epsilon_{\min}\) after a configured fraction of the phase's episodes and holding it thereafter.

The replay buffer can be warm-started with pregenerated expert demonstrations before training begins. In the flagship configuration, 500,000 omniscient expert transitions are loaded into the buffer at initialization. This is important because the action space is large (\(n+1\) discrete actions) and terminal success rewards are sparse, making unguided exploration from scratch difficult.

The concrete phase definitions, epsilon arrays, and expert generation workflow are documented in [MASTER_EXPERIMENT_DESIGN.md](MASTER_EXPERIMENT_DESIGN.md).

## 6. Training Algorithm

The following pseudocode summarizes the actual training loop, including curriculum phases, expert seeding, update-frequency gating, and phase-wise epsilon scheduling.

```
Input: LDPC code (H, G), pre-generated curriculum dataset D, hyperparameters
Output: Trained Q-network Q(s, a; θ)

1. Initialize:
   - Q-network Q(s, a; θ) with random weights
   - Target network Q(s, a; θ⁻) ← θ
   - Replay buffer B with capacity N
   - Load epsilon schedules: one row per epoch of ε_start[], ε_decay[], ε_min[]

2. Optional: Pre-fill B with expert demonstration transitions

3. Set segment ← (1, 1), env_steps ← 0, updates ← 0

4. For each epoch k:
   - Install row k of the epsilon schedules; ε ← ε_start[k][1]
   For each sample (noisy_codeword, original_codeword, phase, error_info) in D[train]:

     a. Segment transition check:
        If (k, phase) ≠ segment:
          - Save the closing segment's metrics, clear memory
          - Update ε parameters: ε_start, ε_decay, ε_min ← schedules[k][phase]
          - If epsilon_phase_reset: ε ← ε_start[k][phase]
          - segment ← (k, phase)

     b. If auto_episodes: ε ← schedule(k, phase, index of this episode within the phase)

     c. Initialize episode from sample

     d. For each step until terminal:
        i.   Obtain the legal actions L from the environment
             (all n+1 actions, or all but the bit flipped at the previous step)
             Select action aₜ:
             - With probability ε: uniform over L
             - Otherwise: aₜ = argmax_{a ∈ L} Q(sₜ, a; θ)

        ii.  Execute aₜ in environment, observe (sₜ₊₁, rₜ, done)

        iii. Store (sₜ, aₜ, rₜ, sₜ₊₁, done) in B

        iv.  env_steps ← env_steps + 1
             If env_steps mod update_frequency = 0 and |B| ≥ min_replay_size:
               - Sample mini-batch from B
               - Compute targets y = r + γ(1-d)·max_a' Q(s', a'; θ⁻),
                 excluding a' = a when the re-flip restriction is on
               - Update θ via gradient descent on MSE loss
               - Unless auto_episodes: decay ε according to the phase schedule
               - updates ← updates + 1; if updates mod C = 0: θ⁻ ← θ

   - If save_epoch_checkpoints: write θ as the epoch-k checkpoint

5. Return trained Q-network Q(s, a; θ)
```

## 7. Neural Function Approximation

Two network families are implemented and exposed through the network factory:

- `SimpleMLP`
- `LDPCOptimizedMLP`

The flagship configuration uses `LDPCOptimizedMLP`, a deep multilayer perceptron with:

- input dimension `state_dim`
- hidden widths `640 -> 480 -> 360 -> 270`
- `LayerNorm` after each hidden linear layer
- `LeakyReLU` activations
- dropout `0.05` in middle layers
- output dimension `action_dim`

For the `64x128` problem, this network maps a 451-dimensional state to 129 Q-values.

The architecture is still generic in the sense that it operates on the flattened state vector. It is "LDPC-optimized" by sizing and normalization choices, not by graph message passing or explicit Tanner-graph neural operators.

## 8. Implemented Problem Boundary

The current repository should be understood as implementing:

- binary hard-decision decoding
- sequential bit-flipping control
- explicit-stop episodic decoding
- off-policy value learning with replay
- curriculum-based training support
- optional expert replay seeding

It does not currently implement:

- belief-propagation-style differentiable message passing as the learned policy class
- soft-input channel likelihood modeling in the RL state
- a decoder that outputs a full correction in one action

## 9. Recommended Use of This Document

Researchers should use this document to answer:

- What is the decoding problem being solved?
- What are the state, action, reward, and success definitions?
- What exactly does the agent control?
- What function approximator is used?