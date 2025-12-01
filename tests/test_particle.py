"""Tests for the Particle class."""

import pytest
import numpy as np

from simulator.particle import Particle


class TestParticleInit:
    """Tests for Particle initialization."""

    def test_default_initialization(self):
        """Test particle with default values."""
        p = Particle()
        assert np.allclose(p.position, [0.0, 0.0])
        assert np.allclose(p.velocity, [0.0, 0.0])
        assert p.mass == 1.0
        assert p.radius == 1.0

    def test_custom_initialization(self):
        """Test particle with custom values."""
        p = Particle(
            position=(1.0, 2.0),
            velocity=(3.0, 4.0),
            mass=5.0,
            radius=2.0,
        )
        assert np.allclose(p.position, [1.0, 2.0])
        assert np.allclose(p.velocity, [3.0, 4.0])
        assert p.mass == 5.0
        assert p.radius == 2.0

    def test_invalid_mass(self):
        """Test that non-positive mass raises ValueError."""
        with pytest.raises(ValueError, match="Mass must be positive"):
            Particle(mass=0)
        with pytest.raises(ValueError, match="Mass must be positive"):
            Particle(mass=-1)

    def test_invalid_radius(self):
        """Test that non-positive radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            Particle(radius=0)
        with pytest.raises(ValueError, match="Radius must be positive"):
            Particle(radius=-1)


class TestParticleProperties:
    """Tests for Particle property setters."""

    def test_set_position(self):
        """Test setting position."""
        p = Particle()
        p.position = (5.0, 10.0)
        assert np.allclose(p.position, [5.0, 10.0])

    def test_set_velocity(self):
        """Test setting velocity."""
        p = Particle()
        p.velocity = (3.0, 4.0)
        assert np.allclose(p.velocity, [3.0, 4.0])

    def test_position_returns_copy(self):
        """Test that position getter returns a copy."""
        p = Particle(position=(1.0, 2.0))
        pos = p.position
        pos[0] = 999.0
        assert np.allclose(p.position, [1.0, 2.0])

    def test_velocity_returns_copy(self):
        """Test that velocity getter returns a copy."""
        p = Particle(velocity=(1.0, 2.0))
        vel = p.velocity
        vel[0] = 999.0
        assert np.allclose(p.velocity, [1.0, 2.0])


class TestParticleUpdate:
    """Tests for Particle update method."""

    def test_update_with_no_acceleration(self):
        """Test update without acceleration."""
        p = Particle(position=(0.0, 0.0), velocity=(1.0, 2.0))
        p.update(1.0)
        assert np.allclose(p.position, [1.0, 2.0])
        assert np.allclose(p.velocity, [1.0, 2.0])

    def test_update_with_acceleration(self):
        """Test update with acceleration."""
        p = Particle(position=(0.0, 0.0), velocity=(0.0, 0.0))
        p.update(1.0, acceleration=(1.0, 2.0))
        assert np.allclose(p.velocity, [1.0, 2.0])
        assert np.allclose(p.position, [1.0, 2.0])

    def test_update_multiple_steps(self):
        """Test multiple update steps."""
        p = Particle(position=(0.0, 0.0), velocity=(1.0, 0.0))
        for _ in range(10):
            p.update(0.1)
        assert np.allclose(p.position, [1.0, 0.0], atol=1e-10)


class TestParticlePhysics:
    """Tests for Particle physics calculations."""

    def test_kinetic_energy_stationary(self):
        """Test kinetic energy of stationary particle."""
        p = Particle(velocity=(0.0, 0.0))
        assert p.kinetic_energy() == 0.0

    def test_kinetic_energy_moving(self):
        """Test kinetic energy of moving particle."""
        p = Particle(velocity=(3.0, 4.0), mass=2.0)
        # KE = 0.5 * m * v^2 = 0.5 * 2 * 25 = 25
        assert np.isclose(p.kinetic_energy(), 25.0)

    def test_momentum(self):
        """Test momentum calculation."""
        p = Particle(velocity=(2.0, 3.0), mass=4.0)
        mom = p.momentum()
        assert np.allclose(mom, [8.0, 12.0])


class TestParticleCollisions:
    """Tests for Particle collision detection."""

    def test_distance_to(self):
        """Test distance calculation between particles."""
        p1 = Particle(position=(0.0, 0.0))
        p2 = Particle(position=(3.0, 4.0))
        assert np.isclose(p1.distance_to(p2), 5.0)

    def test_collides_with_overlapping(self):
        """Test collision detection for overlapping particles."""
        p1 = Particle(position=(0.0, 0.0), radius=1.0)
        p2 = Particle(position=(1.5, 0.0), radius=1.0)
        assert p1.collides_with(p2)

    def test_collides_with_not_overlapping(self):
        """Test collision detection for non-overlapping particles."""
        p1 = Particle(position=(0.0, 0.0), radius=1.0)
        p2 = Particle(position=(3.0, 0.0), radius=1.0)
        assert not p1.collides_with(p2)

    def test_collides_with_touching(self):
        """Test collision detection for touching particles."""
        p1 = Particle(position=(0.0, 0.0), radius=1.0)
        p2 = Particle(position=(2.0, 0.0), radius=1.0)
        # Exactly touching should not be considered a collision
        assert not p1.collides_with(p2)


class TestParticleRepr:
    """Tests for Particle string representation."""

    def test_repr(self):
        """Test string representation."""
        p = Particle(position=(1.0, 2.0), velocity=(3.0, 4.0), mass=5.0, radius=2.0)
        repr_str = repr(p)
        assert "Particle" in repr_str
        assert "position" in repr_str
        assert "velocity" in repr_str
        assert "mass" in repr_str
        assert "radius" in repr_str
