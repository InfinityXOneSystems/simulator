"""Simulation module for running physics simulations."""

import numpy as np
from typing import List, Tuple, Optional

from simulator.particle import Particle


class Simulation:
    """A physics simulation environment.

    Manages a collection of particles and runs the simulation with
    configurable gravity and boundary conditions.

    Attributes:
        particles: List of particles in the simulation.
        gravity: Gravitational acceleration vector.
        bounds: Simulation boundaries as (width, height).
    """

    def __init__(
        self,
        gravity: Tuple[float, float] = (0.0, -9.81),
        bounds: Optional[Tuple[float, float]] = None,
    ):
        """Initialize a simulation.

        Args:
            gravity: Gravitational acceleration as (gx, gy).
            bounds: Optional boundary dimensions as (width, height).
                    If None, simulation has no boundaries.
        """
        self._particles: List[Particle] = []
        self._gravity = np.array(gravity, dtype=float)
        self._bounds = bounds
        self._time = 0.0

    @property
    def particles(self) -> List[Particle]:
        """Get the list of particles."""
        return self._particles.copy()

    @property
    def gravity(self) -> np.ndarray:
        """Get the gravity vector."""
        return self._gravity.copy()

    @gravity.setter
    def gravity(self, value: Tuple[float, float]) -> None:
        """Set the gravity vector."""
        self._gravity = np.array(value, dtype=float)

    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        """Get the simulation bounds."""
        return self._bounds

    @property
    def time(self) -> float:
        """Get the current simulation time."""
        return self._time

    def add_particle(self, particle: Particle) -> None:
        """Add a particle to the simulation.

        Args:
            particle: The particle to add.
        """
        self._particles.append(particle)

    def remove_particle(self, particle: Particle) -> bool:
        """Remove a particle from the simulation.

        Args:
            particle: The particle to remove.

        Returns:
            True if the particle was found and removed, False otherwise.
        """
        try:
            self._particles.remove(particle)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Remove all particles from the simulation."""
        self._particles.clear()
        self._time = 0.0

    def step(self, dt: float) -> None:
        """Advance the simulation by one time step.

        Args:
            dt: Time step size (must be positive).

        Raises:
            ValueError: If dt is not positive.
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")

        # Update all particles
        for particle in self._particles:
            particle.update(dt, tuple(self._gravity))

        # Handle boundary collisions if bounds are defined
        if self._bounds is not None:
            self._handle_boundary_collisions()

        # Handle particle-particle collisions
        self._handle_particle_collisions()

        self._time += dt

    def _handle_boundary_collisions(self) -> None:
        """Handle collisions between particles and boundaries."""
        if self._bounds is None:
            return

        width, height = self._bounds

        for particle in self._particles:
            pos = particle.position
            vel = particle.velocity
            radius = particle.radius

            # Left boundary
            if pos[0] - radius < 0:
                pos[0] = radius
                vel[0] = abs(vel[0])

            # Right boundary
            if pos[0] + radius > width:
                pos[0] = width - radius
                vel[0] = -abs(vel[0])

            # Bottom boundary
            if pos[1] - radius < 0:
                pos[1] = radius
                vel[1] = abs(vel[1])

            # Top boundary
            if pos[1] + radius > height:
                pos[1] = height - radius
                vel[1] = -abs(vel[1])

            particle.position = tuple(pos)
            particle.velocity = tuple(vel)

    def _handle_particle_collisions(self) -> None:
        """Handle elastic collisions between particles."""
        n = len(self._particles)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self._particles[i]
                p2 = self._particles[j]

                if p1.collides_with(p2):
                    self._resolve_collision(p1, p2)

    def _resolve_collision(self, p1: Particle, p2: Particle) -> None:
        """Resolve an elastic collision between two particles.

        Args:
            p1: First particle.
            p2: Second particle.
        """
        # Get positions and velocities
        pos1 = p1.position
        pos2 = p2.position
        vel1 = p1.velocity
        vel2 = p2.velocity
        m1 = p1.mass
        m2 = p2.mass

        # Calculate collision normal
        normal = pos1 - pos2
        distance = np.linalg.norm(normal)
        if distance == 0:
            return  # Particles at same position, skip

        normal = normal / distance

        # Relative velocity
        rel_vel = vel1 - vel2

        # Relative velocity along collision normal
        vel_along_normal = np.dot(rel_vel, normal)

        # Don't resolve if velocities are separating
        if vel_along_normal > 0:
            return

        # Calculate impulse scalar (elastic collision with restitution = 1)
        # Formula: j = -(1 + e) * v_rel_n / (1/m1 + 1/m2) where e = 1
        impulse = -2 * vel_along_normal / (1 / m1 + 1 / m2)

        # Apply impulse to velocities
        vel1 += (impulse / m1) * normal
        vel2 -= (impulse / m2) * normal

        p1.velocity = tuple(vel1)
        p2.velocity = tuple(vel2)

        # Separate particles to prevent overlap
        overlap = (p1.radius + p2.radius) - distance
        if overlap > 0:
            separation = normal * (overlap / 2 + 0.01)
            p1.position = tuple(pos1 + separation)
            p2.position = tuple(pos2 - separation)

    def total_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of all particles.

        Returns:
            Sum of kinetic energies of all particles.
        """
        return sum(p.kinetic_energy() for p in self._particles)

    def total_momentum(self) -> np.ndarray:
        """Calculate total momentum of all particles.

        Returns:
            Sum of momentum vectors of all particles.
        """
        if not self._particles:
            return np.array([0.0, 0.0])
        return sum(p.momentum() for p in self._particles)

    def run(self, duration: float, dt: float = 0.01) -> List[List[Tuple[float, float]]]:
        """Run the simulation for a specified duration.

        Args:
            duration: Total time to simulate.
            dt: Time step size.

        Returns:
            List of position histories for each particle.
            Each history is a list of (x, y) positions.
        """
        histories = [[] for _ in self._particles]
        num_steps = int(duration / dt)

        for _ in range(num_steps):
            for i, particle in enumerate(self._particles):
                pos = particle.position
                histories[i].append((float(pos[0]), float(pos[1])))
            self.step(dt)

        return histories

    def __repr__(self) -> str:
        """Return string representation of the simulation."""
        return (
            f"Simulation(particles={len(self._particles)}, "
            f"gravity={tuple(self._gravity)}, bounds={self._bounds})"
        )
