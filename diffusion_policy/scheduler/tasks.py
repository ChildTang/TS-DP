"""Per-task settings of the PPO scheduler.

Every task shares the same scheduler: at each decision step PPO picks one of
3 x 3 x 2 x 4 x 2 = 144 combinations of
(sigma_scale, accept_threshold, high / medium / low noise draft budget).
What differs between tasks is the candidate values, the observation given to
PPO and the task metrics.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROBOMIMIC_DRAFT_BUDGETS = dict(
    high_steps_options=(5, 10),
    medium_steps_options=(30, 35, 40, 45),
    low_steps_options=(5, 10),
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    # how the PPO observation is built: 'robomimic' | 'pusht' | 'lowdim'
    obs_type: str
    # task metrics: 'binary' | 'pusht' | 'kitchen' | 'blockpush'
    outcome: str
    # environment steps between two scheduler decisions
    update_interval: int
    # environment steps per episode
    max_steps: int
    # action candidates
    threshold_options: Tuple[float, ...]
    sigma_options: Tuple[float, ...] = (0.1, 0.3, 0.5)
    high_steps_options: Tuple[int, ...] = ROBOMIMIC_DRAFT_BUDGETS['high_steps_options']
    medium_steps_options: Tuple[int, ...] = ROBOMIMIC_DRAFT_BUDGETS['medium_steps_options']
    low_steps_options: Tuple[int, ...] = ROBOMIMIC_DRAFT_BUDGETS['low_steps_options']
    # robomimic observation
    image_keys: Tuple[str, ...] = ()
    robots: Tuple[int, ...] = (0,)
    # decision interval at evaluation (None: same as update_interval)
    eval_update_interval: Optional[int] = None

    @property
    def action_dims(self) -> List[int]:
        return [len(self.sigma_options), len(self.threshold_options),
                len(self.high_steps_options), len(self.medium_steps_options),
                len(self.low_steps_options)]


def _robomimic(name, image_keys=('agentview_image',), robots=(0,),
        update_interval=48, max_steps=400, threshold_options=(0.7, 0.8, 0.9)):
    return TaskSpec(name=name, obs_type='robomimic', outcome='binary',
        image_keys=image_keys, robots=robots,
        update_interval=update_interval, max_steps=max_steps,
        threshold_options=threshold_options)


TASKS: Dict[str, TaskSpec] = {
    'can': _robomimic('can'),
    'lift': _robomimic('lift'),
    'square': _robomimic('square'),
    'tool_hang': _robomimic('tool_hang', image_keys=('sideview_image',), max_steps=700),
    'transport': _robomimic('transport',
        image_keys=('shouldercamera0_image', 'shouldercamera1_image'), robots=(0, 1),
        update_interval=80, max_steps=700, threshold_options=(0.5, 0.7, 0.9)),
    'pusht': TaskSpec(name='pusht', obs_type='pusht', outcome='pusht',
        update_interval=48, max_steps=300, threshold_options=(0.5, 0.7, 0.9)),
    'kitchen': TaskSpec(name='kitchen', obs_type='lowdim', outcome='kitchen',
        update_interval=40, max_steps=280, threshold_options=(0.1, 0.3, 0.5)),
    'block_pushing': TaskSpec(name='block_pushing', obs_type='lowdim', outcome='blockpush',
        update_interval=40, max_steps=200, threshold_options=(0.7, 0.8, 0.9),
        high_steps_options=(3, 5), medium_steps_options=(10, 15, 20, 25),
        low_steps_options=(3, 5), eval_update_interval=5),
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[name]
