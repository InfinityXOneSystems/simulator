"""Particle module for physics simulation."""

import numpy as np
from typing import Tuple


class Particle:
    """Represents a particle in a physics simulation.

    Attributes:
        position: Current position as (x, y) coordinates.
        velocity: Current velocity as (vx, vy) components.
        mass: Mass of the particle.
        radius: Radius of the particle for collision detection.
    """

    def __init__(
        self,
        position: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.0, 0.0),
        mass: float = 1.0,
        radius: float = 1.0,
    ):
        """Initialize a particle.

        Args:
            position: Initial position as (x, y) tuple.
            velocity: Initial velocity as (vx, vy) tuple.
            mass: Mass of the particle (must be positive).
            radius: Radius for collision detection (must be positive).

        Raises:
            ValueError: If mass or radius is not positive.
        """
        if mass <= 0:
            raise ValueError("Mass must be positive")
        if radius <= 0:
            raise ValueError("Radius must be positive")

        self._position = np.array(position, dtype=float)
        self._velocity = np.array(velocity, dtype=float)
        self._mass = float(mass)
        self._radius = float(radius)

    @property
    def position(self) -> np.ndarray:
        """Get the current position."""
        return self._position.copy()

    @position.setter
    def position(self, value: Tuple[float, float]) -> None:
        """Set the position."""
        self._position = np.array(value, dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        """Get the current velocity."""
        return self._velocity.copy()

    @velocity.setter
    def velocity(self, value: Tuple[float, float]) -> None:
        """Set the velocity."""
        self._velocity = np.array(value, dtype=float)

    @property
    def mass(self) -> float:
        """Get the mass."""
        return self._mass

    @property
    def radius(self) -> float:
        """Get the radius."""
        return self._radius

    def update(self, dt: float, acceleration: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Update particle position and velocity.

        Uses simple Euler integration to update the particle state.

        Args:
            dt: Time step for the update.
            acceleration: External acceleration (ax, ay) applied to the particle.
        """
        acc = np.array(acceleration, dtype=float)
        self._velocity += acc * dt
        self._position += self._velocity * dt

    def kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the particle.

        Returns:
            Kinetic energy (0.5 * m * v^2).
        """
        speed_squared = np.dot(self._velocity, self._velocity)
        return 0.5 * self._mass * speed_squared

    def momentum(self) -> np.ndarray:
        """Calculate the momentum of the particle.

        Returns:
            Momentum vector (m * v).
        """
        return self._mass * self._velocity

    def distance_to(self, other: "Particle") -> float:
        """Calculate distance to another particle.

        Args:
            other: Another particle.

        Returns:
            Distance between the centers of the two particles.
        """
        diff = self._position - other._position
        return float(np.linalg.norm(diff))

    def collides_with(self, other: "Particle") -> bool:
        """Check if this particle collides with another.

        Args:
            other: Another particle.

        Returns:
            True if the particles overlap, False otherwise.
        """
        distance = self.distance_to(other)
        return distance < (self._radius + other._radius)

    def __repr__(self) -> str:
        """Return string representation of the particle."""
        return (
            f"Particle(position={tuple(self._position)}, "
            f"velocity={tuple(self._velocity)}, "
            f"mass={self._mass}, radius={self._radius})"
        )
