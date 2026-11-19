"""Data contracts shared by the synthetic test and the later real-data loaders."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StackDef:
    name: str
    angle_min: float
    angle_max: float


@dataclass
class Well:
    """A well with logs in TVDSS on a regular depth step.

    logs columns: tvdss, vp, vs, rho, phi, vsh, sw, litho, fluid, vs_source.
    checkshot columns: tvdss, twt (seconds).
    """
    name: str
    il: int
    xl: int
    logs: pd.DataFrame
    checkshot: pd.DataFrame
    hugin_top: float
    hugin_base: float

    @property
    def dz(self):
        return float(self.logs["tvdss"].iloc[1] - self.logs["tvdss"].iloc[0])

    def twt(self):
        return np.interp(self.logs["tvdss"], self.checkshot["tvdss"], self.checkshot["twt"])


@dataclass
class Cube:
    """Partial-stack seismic cube. stacks: name -> array (n_il, n_xl, n_t)."""
    stacks: dict
    stack_defs: list
    t0: float
    dt: float
    horizon_twt: np.ndarray = None
    truth_class: np.ndarray = None
    meta: dict = field(default_factory=dict)

    @property
    def nt(self):
        return next(iter(self.stacks.values())).shape[2]

    @property
    def shape(self):
        return next(iter(self.stacks.values())).shape[:2]

    @property
    def t(self):
        return self.t0 + np.arange(self.nt) * self.dt

    def trace(self, il, xl):
        return np.stack([self.stacks[s.name][il, xl] for s in self.stack_defs])
