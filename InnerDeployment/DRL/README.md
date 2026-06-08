# DRL Sensor Placement

`SensorDRL` is a lightweight DQN optimizer implemented with the repository's
existing `torch` dependency. It does not require `gymnasium` or
`stable-baselines3`.

Use `algorithm="drl"` in the existing Engine API. `algorithm="dqn"` is also
accepted as an alias.

```python
final_points, out_path = run_pipeline(
    ...,
    algorithm="drl",
    optimizer_params={
        ...,
        "drl": {
            "generations": 100,
            "candidate_stride": 5,
            "max_candidates": 512,
            "fitness_kwargs": {"target_coverage": 90.0},
        },
    },
    optimizer_run_params={
        ...,
        "drl": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.985,
            "heuristic_warmup_episodes": 1,
        },
    },
)
```

Main tuning parameters:

- `generations`: number of training episodes.
- `candidate_stride`: spatial candidate sampling interval. Lower values search
  more positions and increase training cost.
- `max_candidates`: maximum candidate action count after mixing high-gain and
  spatially distributed candidates.
- `hidden_dim`, `learning_rate`, `gamma`: Q-network training parameters.
- `fitness_kwargs`: final selection and pruning objective passed to
  `FitnessFunc`. The DQN step reward is the change in this common fitness.
