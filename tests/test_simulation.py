"""Tests for the Simulation class."""

import pytest
import numpy as np

from simulator.particle import Particle
from simulator.simulation import Simulation


class TestSimulationInit:
    """Tests for Simulation initialization."""

    def test_default_initialization(self):
        """Test simulation with default values."""
        sim = Simulation()
        assert len(sim.particles) == 0
        assert np.allclose(sim.gravity, [0.0, -9.81])
        assert sim.bounds is None
        assert sim.time == 0.0

    def test_custom_gravity(self):
        """Test simulation with custom gravity."""
        sim = Simulation(gravity=(0.0, -10.0))
        assert np.allclose(sim.gravity, [0.0, -10.0])

    def test_custom_bounds(self):
        """Test simulation with custom bounds."""
        sim = Simulation(bounds=(100.0, 200.0))
        assert sim.bounds == (100.0, 200.0)


class TestSimulationParticles:
    """Tests for particle management in Simulation."""

    def test_add_particle(self):
        """Test adding a particle."""
        sim = Simulation()
        p = Particle()
        sim.add_particle(p)
        assert len(sim.particles) == 1

    def test_add_multiple_particles(self):
        """Test adding multiple particles."""
        sim = Simulation()
        for _ in range(5):
            sim.add_particle(Particle())
        assert len(sim.particles) == 5

    def test_remove_particle(self):
        """Test removing a particle."""
        sim = Simulation()
        p = Particle()
        sim.add_particle(p)
        result = sim.remove_particle(p)
        assert result is True
        assert len(sim.particles) == 0

    def test_remove_nonexistent_particle(self):
        """Test removing a particle that doesn't exist."""
        sim = Simulation()
        p = Particle()
        result = sim.remove_particle(p)
        assert result is False

    def test_clear(self):
        """Test clearing all particles."""
        sim = Simulation()
        for _ in range(5):
            sim.add_particle(Particle())
        sim.clear()
        assert len(sim.particles) == 0
        assert sim.time == 0.0

    def test_particles_returns_copy(self):
        """Test that particles getter returns a copy."""
        sim = Simulation()
        p = Particle()
        sim.add_particle(p)
        particles = sim.particles
        particles.clear()
        assert len(sim.particles) == 1


class TestSimulationStep:
    """Tests for Simulation step method."""

    def test_step_advances_time(self):
        """Test that step advances simulation time."""
        sim = Simulation()
        sim.step(0.1)
        assert np.isclose(sim.time, 0.1)

    def test_step_with_zero_gravity(self):
        """Test step with zero gravity."""
        sim = Simulation(gravity=(0.0, 0.0))
        p = Particle(position=(0.0, 0.0), velocity=(1.0, 0.0))
        sim.add_particle(p)
        sim.step(1.0)
        assert np.allclose(p.position, [1.0, 0.0])

    def test_step_with_gravity(self):
        """Test step with gravity."""
        sim = Simulation(gravity=(0.0, -10.0))
        p = Particle(position=(0.0, 100.0), velocity=(0.0, 0.0))
        sim.add_particle(p)
        sim.step(1.0)
        # After 1 second: v = -10, pos = 100 + (-10) = 90
        assert np.isclose(p.velocity[1], -10.0)
        assert np.isclose(p.position[1], 90.0)

    def test_step_invalid_dt(self):
        """Test that non-positive dt raises ValueError."""
        sim = Simulation()
        with pytest.raises(ValueError, match="Time step must be positive"):
            sim.step(0)
        with pytest.raises(ValueError, match="Time step must be positive"):
            sim.step(-0.1)


class TestSimulationBoundaries:
    """Tests for boundary collision handling."""

    def test_left_boundary(self):
        """Test collision with left boundary."""
        sim = Simulation(gravity=(0.0, 0.0), bounds=(100.0, 100.0))
        p = Particle(position=(1.0, 50.0), velocity=(-10.0, 0.0), radius=1.0)
        sim.add_particle(p)
        sim.step(1.0)  # Would put particle at x = -9
        assert p.position[0] >= 1.0  # Should be at least radius
        assert p.velocity[0] > 0  # Should be moving right now

    def test_right_boundary(self):
        """Test collision with right boundary."""
        sim = Simulation(gravity=(0.0, 0.0), bounds=(100.0, 100.0))
        p = Particle(position=(99.0, 50.0), velocity=(10.0, 0.0), radius=1.0)
        sim.add_particle(p)
        sim.step(1.0)
        assert p.position[0] <= 99.0  # Should be at most width - radius
        assert p.velocity[0] < 0  # Should be moving left now

    def test_bottom_boundary(self):
        """Test collision with bottom boundary."""
        sim = Simulation(gravity=(0.0, 0.0), bounds=(100.0, 100.0))
        p = Particle(position=(50.0, 1.0), velocity=(0.0, -10.0), radius=1.0)
        sim.add_particle(p)
        sim.step(1.0)
        assert p.position[1] >= 1.0
        assert p.velocity[1] > 0

    def test_top_boundary(self):
        """Test collision with top boundary."""
        sim = Simulation(gravity=(0.0, 0.0), bounds=(100.0, 100.0))
        p = Particle(position=(50.0, 99.0), velocity=(0.0, 10.0), radius=1.0)
        sim.add_particle(p)
        sim.step(1.0)
        assert p.position[1] <= 99.0
        assert p.velocity[1] < 0


class TestSimulationCollisions:
    """Tests for particle-particle collision handling."""

    def test_head_on_collision(self):
        """Test head-on collision between two particles."""
        sim = Simulation(gravity=(0.0, 0.0))
        p1 = Particle(position=(0.0, 0.0), velocity=(1.0, 0.0), mass=1.0, radius=1.0)
        p2 = Particle(position=(1.5, 0.0), velocity=(-1.0, 0.0), mass=1.0, radius=1.0)
        sim.add_particle(p1)
        sim.add_particle(p2)

        # Particles should collide and exchange velocities (equal mass)
        sim.step(0.1)
        # After collision, velocities should be approximately reversed
        assert p1.velocity[0] < 0 or p2.velocity[0] > 0


class TestSimulationEnergy:
    """Tests for energy and momentum calculations."""

    def test_total_kinetic_energy_empty(self):
        """Test kinetic energy with no particles."""
        sim = Simulation()
        assert sim.total_kinetic_energy() == 0.0

    def test_total_kinetic_energy(self):
        """Test total kinetic energy calculation."""
        sim = Simulation(gravity=(0.0, 0.0))
        p1 = Particle(velocity=(3.0, 0.0), mass=2.0)  # KE = 9
        p2 = Particle(velocity=(0.0, 4.0), mass=2.0)  # KE = 16
        sim.add_particle(p1)
        sim.add_particle(p2)
        assert np.isclose(sim.total_kinetic_energy(), 25.0)

    def test_total_momentum_empty(self):
        """Test momentum with no particles."""
        sim = Simulation()
        mom = sim.total_momentum()
        assert np.allclose(mom, [0.0, 0.0])

    def test_total_momentum(self):
        """Test total momentum calculation."""
        sim = Simulation(gravity=(0.0, 0.0))
        p1 = Particle(velocity=(2.0, 0.0), mass=3.0)  # p = (6, 0)
        p2 = Particle(velocity=(0.0, 4.0), mass=2.0)  # p = (0, 8)
        sim.add_particle(p1)
        sim.add_particle(p2)
        mom = sim.total_momentum()
        assert np.allclose(mom, [6.0, 8.0])


class TestSimulationRun:
    """Tests for the run method."""

    def test_run_returns_histories(self):
        """Test that run returns position histories."""
        sim = Simulation(gravity=(0.0, 0.0))
        p = Particle(position=(0.0, 0.0), velocity=(1.0, 0.0))
        sim.add_particle(p)
        histories = sim.run(duration=1.0, dt=0.1)
        assert len(histories) == 1
        assert len(histories[0]) == 10  # 1.0 / 0.1 = 10 steps

    def test_run_multiple_particles(self):
        """Test run with multiple particles."""
        sim = Simulation(gravity=(0.0, 0.0))
        sim.add_particle(Particle())
        sim.add_particle(Particle())
        histories = sim.run(duration=0.5, dt=0.1)
        assert len(histories) == 2
        assert len(histories[0]) == 5
        assert len(histories[1]) == 5


class TestSimulationRepr:
    """Tests for Simulation string representation."""

    def test_repr(self):
        """Test string representation."""
        sim = Simulation(gravity=(0.0, -10.0), bounds=(100.0, 100.0))
        sim.add_particle(Particle())
        repr_str = repr(sim)
        assert "Simulation" in repr_str
        assert "particles=1" in repr_str
        assert "gravity" in repr_str
        assert "bounds" in repr_str
