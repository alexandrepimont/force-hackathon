"""AVO lithology/fluid workflow (FORCE Hackathon 2026)."""

import numpy as np

LITHO_SHALE, LITHO_SAND, LITHO_TIGHT = 0, 1, 2
FLUID_NONE, FLUID_BRINE, FLUID_OIL = 0, 1, 2

CLASS_NAMES_FLUID = ["shale", "tight", "brine_sand", "oil_sand"]
CLASS_NAMES_LITHO = ["shale", "tight", "sand"]


def joint_class(litho, fluid):
    """Map lithology and fluid codes to the joint class used for classification."""
    litho = np.asarray(litho)
    fluid = np.asarray(fluid)
    c = np.zeros(litho.shape, dtype=int)
    c[litho == LITHO_TIGHT] = 1
    sand = litho == LITHO_SAND
    c[sand & (fluid != FLUID_OIL)] = 2
    c[sand & (fluid == FLUID_OIL)] = 3
    return c


def collapse_fluid(y):
    """Merge brine and oil sand when fluid is not a viable target (gate G3 failed)."""
    y = np.array(y, copy=True)
    y[y == 3] = 2
    return y


def class_names(use_fluid):
    return CLASS_NAMES_FLUID if use_fluid else CLASS_NAMES_LITHO
