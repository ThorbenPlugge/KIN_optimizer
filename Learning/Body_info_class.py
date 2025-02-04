from dataclasses import dataclass
from typing import List


@dataclass
class TimeState:
    """Represents the state of a celestial body at a specific time."""
    time: int
    position: List[float]
    velocity: List[float]


@dataclass
class CelestialBody:
    """Represents a celestial body, including its mass and states over time."""
    name: str
    mass: float
    states: List[TimeState]


# example usage:
# earth = CelestialBody(
#     name="Earth",
#     mass=5.972e24,
#     states=[
#         TimeState(time=5,  position=[
#                   0.0, 3.0, 12.004], velocity=[0.0, 0.0, 0.1]),
#         TimeState(time=8,  position=[0.0, 2.0, 14.0],
#                   velocity=[0.1, 345.0, 9.0])
#     ]
# )
