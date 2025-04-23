import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from pathlib import Path
import os

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore


os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

