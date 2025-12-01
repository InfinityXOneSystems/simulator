# Simulator

A simple 2D physics simulation library written in Python. This library provides tools for simulating particle dynamics with gravity, boundary collisions, and particle-particle interactions.

## Features

- **Particle Simulation**: Create particles with position, velocity, mass, and radius
- **Physics Engine**: Euler integration for motion updates
- **Collision Detection**: Particle-particle and boundary collision handling
- **Energy Calculations**: Track kinetic energy and momentum
- **Configurable Environment**: Set custom gravity and simulation boundaries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/InfinityXOneSystems/simulator.git
cd simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from simulator import Particle, Simulation

# Create a simulation with gravity and boundaries
sim = Simulation(gravity=(0.0, -9.81), bounds=(100.0, 100.0))

# Add particles
particle1 = Particle(
    position=(50.0, 80.0),
    velocity=(2.0, 0.0),
    mass=1.0,
    radius=2.0
)
sim.add_particle(particle1)

particle2 = Particle(
    position=(50.0, 20.0),
    velocity=(-1.0, 5.0),
    mass=2.0,
    radius=3.0
)
sim.add_particle(particle2)

# Run simulation
histories = sim.run(duration=5.0, dt=0.01)

# Access particle trajectories
for i, history in enumerate(histories):
    print(f"Particle {i}: {len(history)} positions recorded")
```

## API Reference

### Particle

```python
Particle(
    position=(0.0, 0.0),  # Initial position (x, y)
    velocity=(0.0, 0.0),  # Initial velocity (vx, vy)
    mass=1.0,             # Mass (must be positive)
    radius=1.0            # Collision radius (must be positive)
)
```

**Methods:**
- `update(dt, acceleration)`: Update position and velocity
- `kinetic_energy()`: Calculate kinetic energy
- `momentum()`: Calculate momentum vector
- `distance_to(other)`: Distance to another particle
- `collides_with(other)`: Check collision with another particle

### Simulation

```python
Simulation(
    gravity=(0.0, -9.81),  # Gravitational acceleration
    bounds=None            # Optional (width, height) boundaries
)
```

**Methods:**
- `add_particle(particle)`: Add a particle to the simulation
- `remove_particle(particle)`: Remove a particle
- `clear()`: Remove all particles
- `step(dt)`: Advance simulation by one time step
- `run(duration, dt)`: Run simulation and return position histories
- `total_kinetic_energy()`: Sum of all particle kinetic energies
- `total_momentum()`: Sum of all particle momenta

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License