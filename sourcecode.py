import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

current_dir = os.path.dirname(os.path.abspath(__file__))
JSON_FILE_PATH = os.path.join(current_dir, "parameters_solar.json")

class CelestialBody:
    """
    Represents a single celestial body (e.g., planet, star, or satellite) in the N-body simulation.
    
    Attributes:
        name (str): The name of the celestial body.
        mass (float): The mass of the body (in scaled units, relative to solar mass).
        position (np.ndarray): The current 2D position vector [x, y] in AU.
        velocity (np.ndarray): The current 2D velocity vector [vx, vy] in AU/year.
        acceleration (np.ndarray): The current 2D acceleration vector a(t) [ax, ay].
        prev_acceleration (np.ndarray): The acceleration at the previous time step a(t-dt). 
                                        This is strictly required for the Beeman integration algorithm.
        colour (str): The color string used for Matplotlib visualization.
        prev_y (float): Stores the y-coordinate from the previous step. Used for orbital period detection.
        period (float): The calculated orbital period in Earth years.
    """

    def __init__(self, name, mass, initial_position, initial_velocity, colour="black"):
        self.name = name
        self.mass = float(mass)
        
        # Using NumPy arrays for efficient vectorized vector arithmetic (addition/scalar multiplication)
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        
        # Initialize accelerations to zero vectors. 
        # Beeman integrator uniquely requires tracking both current and previous accelerations.
        self.acceleration = np.zeros(2, dtype=float)
        self.prev_acceleration = np.zeros(2, dtype=float)
        
        # Visualization property
        self.colour = colour
        
        # Variables specifically designed for Experiment 1 (Orbital Period Calculation).
        # By tracking the sign change of the y-coordinate, the program can automatically 
        # detect when the celestial body completes exactly one full orbit (crosses the positive x-axis).
        self.prev_y = float(initial_position[1]) 
        self.period = None

class SolarSystem:
    """
    Core physics engine managing the N-body simulation.
    Handles the calculation of gravitational interactions, total system energy, 
    and time integration (advancing the simulation step-by-step).
    """
    def __init__(self, G):
        """
        Initializes the solar system environment.
        Args:
            G (float): The gravitational constant in scaled units 
                       (e.g., AU^3 * M_sun^-1 * yr^-2).
        """
        self.bodies = []  # List to store all CelestialBody objects in the system
        self.G = float(G)

    def add_body(self, body):
        """Adds a CelestialBody object to the simulation system."""
        self.bodies.append(body)

    def calculate_accelerations(self):
        """
        Calculates the instantaneous acceleration for all bodies in the system 
        based on Newton's Law of Universal Gravitation.
        Employs the principle of superposition to sum up gravitational forces.
        """
        # 1. Reset accelerations to zero vectors before computing forces for the current time step.
        for body in self.bodies:
            body.acceleration = np.zeros(2, dtype=float)

        # 2. Compute pairwise gravitational interactions. 
        # Computational complexity is O(N^2), which is perfectly acceptable for the Solar System scale.
        for i, body_i in enumerate(self.bodies):
            for j, body_j in enumerate(self.bodies):
                if i == j:
                    continue  # A body does not exert a gravitational force on itself

                # Calculate the relative position vector (r_vec) from body_i to body_j
                r_vec = body_j.position - body_i.position
                
                # Calculate the Euclidean distance between the two bodies
                distance = np.linalg.norm(r_vec)
                
                # Singularity prevention (Critical for robustness):
                # Avoids ZeroDivisionError during extremely close encounters 
                # (e.g., launching a satellite exactly from Earth's center in Experiment 3).
                if distance == 0:
                    continue

                # Vectorized acceleration calculation: a = G * m_j * r_vec / |r|^3
                # This is mathematically equivalent to (G * m_j / distance^2) * (r_vec / distance) 
                # but computationally more efficient as it avoids calculating the unit vector separately.
                a_magnitude = (self.G * body_j.mass) / (distance ** 3)
                
                # Accumulate the acceleration vector using the principle of superposition
                body_i.acceleration += a_magnitude * r_vec

    def step_forward(self, dt, method="beeman"):
        """
        Advances the physical simulation by a single time step (dt) using numerical integration.
        Provides a comparative framework to analyze the stability and energy conservation 
        of different integration schemes.
        
        Args:
            dt (float): The time step size in years.
            method (str): The integration algorithm. Supported: 'euler', 'euler-cromer', 'beeman'.
        """
        if method == "euler":
            # 1. Forward Euler Method (Explicit)
            # A first-order method. It is non-symplectic and systematically introduces artificial 
            # energy into the system (truncation error), causing orbits to spiral outward rapidly. 
            # Included primarily as a baseline to demonstrate numerical instability in Experiment 2.
            self.calculate_accelerations()
            for body in self.bodies:
                new_position = body.position + body.velocity * dt
                new_velocity = body.velocity + body.acceleration * dt
                body.position = new_position
                body.velocity = new_velocity
                
        elif method == "euler-cromer":
            # 2. Euler-Cromer Method (Semi-implicit / Symplectic Euler)
            # Unlike Forward Euler, this updates the velocity FIRST, and uses the *new* velocity 
            # to update the position. This subtle modification makes the algorithm symplectic, 
            # preserving phase-space volume and yielding stable, closed orbits over time.
            self.calculate_accelerations()
            for body in self.bodies:
                body.velocity += body.acceleration * dt
                body.position += body.velocity * dt
                
        elif method == "beeman":
            # 3. Beeman Integration Method (Predictor-Corrector approach)
            # The primary algorithm for this simulation. It provides superior energy conservation 
            # and numerical stability for periodic planetary orbits compared to Euler methods.
            
            # Step 1: Position Predictor.
            # Calculates the position at t + dt using a Taylor expansion approximation 
            # involving current a(t) and previous a(t-dt) accelerations.
            self.calculate_accelerations()
            old_accelerations = []
            
            for body in self.bodies:
                old_accelerations.append(np.copy(body.acceleration))
                
                # Bootstrap condition for the very first time step (t=0) where a(t-dt) does not exist.
                # We approximate a(t-dt) using the initial a(t).
                if np.all(body.prev_acceleration == 0):
                    body.prev_acceleration = np.copy(body.acceleration)
                
                body.position += body.velocity * dt + (1/6) * (4 * body.acceleration - body.prev_acceleration) * (dt**2)
            
            # Step 2: Force Evaluation.
            # Compute the new acceleration a(t+dt) based on the newly predicted positions.
            self.calculate_accelerations()
            
            # Step 3: Velocity Corrector.
            # Corrects the velocity at t + dt using the newly computed a(t+dt), current a(t), and past a(t-dt).
            # This implicit-like correction step is what grants Beeman its high accuracy.
            for i, body in enumerate(self.bodies):
                a_t_plus_dt = body.acceleration       # Newly computed a(t+dt)
                a_t = old_accelerations[i]            # Cached a(t)
                a_t_minus_dt = body.prev_acceleration # Past a(t-dt)
                
                body.velocity += (1/6) * (2 * a_t_plus_dt + 5 * a_t - a_t_minus_dt) * dt
                
                # Crucial state update: Store current a(t) as a(t-dt) for the next iteration step.
                body.prev_acceleration = np.copy(a_t)
        else:
            raise ValueError("Unknown integration method! Please select 'beeman', 'euler-cromer', or 'euler'")
    
    def calculate_energy(self):
        """
        Calculates the instantaneous total mechanical energy (E = K + U) of the N-body system.
        This serves as a critical diagnostic tool to evaluate the energy conservation 
        properties and symplectic nature of the chosen numerical integrator.
        
        Returns:
            tuple: (kinetic_energy, potential_energy) in scaled energy units.
        """
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        # 1. Total Kinetic Energy (K): K(t) = Sum( 0.5 * m_i * |v_i|^2 )
        for body in self.bodies:
            # np.linalg.norm efficiently computes the Euclidean norm (scalar speed) of the velocity vector
            speed = np.linalg.norm(body.velocity)
            kinetic_energy += 0.5 * body.mass * (speed ** 2)
            
        # 2. Total Gravitational Potential Energy (U): U(t) = -0.5 * Sum( G * m_i * m_j / r_ij )
        for i, body_i in enumerate(self.bodies):
            for j, body_j in enumerate(self.bodies):
                if i == j:
                    continue # Skip self-interaction
                
                # Calculate scalar distance r_ij between body i and body j
                r_vec = body_j.position - body_i.position
                distance = np.linalg.norm(r_vec)
                
                if distance == 0:
                    continue # Singularity safeguard
                    
                # Accumulate the raw potential scalar 
                potential_energy += (self.G * body_i.mass * body_j.mass) / distance
                
        # Critical Physics Correction:
        # 1. Multiply by 0.5: The nested loops calculate every pair twice (i->j and j->i). 
        #    This factor eliminates the double-counting error.
        # 2. Apply negative sign: Gravitational potential energy must be negative to represent 
        #    a bound physical state (a gravitational potential well).
        potential_energy = -0.5 * potential_energy
        
        return kinetic_energy, potential_energy
    


def load_system_from_json(json_filename, system):
    """
    Parses an external JSON configuration file to instantiate celestial bodies.
    Automatically calculates the initial state vectors (position and velocity) 
    to establish stable circular orbits around the central mass.
    
    Args:
        json_filename (str): The absolute or relative path to the parameter file.
        system (SolarSystem): The target simulation environment to populate.
    """
    try:
        with open(json_filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f" Error: Configuration file '{json_filename}' not found!")
        return
        
    # Extract the list of body dictionaries from the parsed JSON data
    bodies_list = data.get("bodies", [])

    # 1. First Pass: Identify the central mass (Sun).
    # This is strictly required as the central mass dictates the gravitational 
    # potential well used to calculate the stable orbital velocities of the planets.
    sun_mass = 0
    for info in bodies_list:
        if info['name'].lower() == 'sun':
            sun_mass = float(info['mass'])
            break

    # 2. Second Pass: Initialize kinematics and instantiate objects.
    for info in bodies_list:
        name = info['name'].capitalize() 
        mass = float(info['mass'])
        radius = float(info['orbital_radius']) 
        colour = info.get('colour', 'black') 
        
        if radius == 0.0:
            # The central star (Sun) is placed stationary at the origin of the coordinate system.
            pos = [0.0, 0.0]
            vel = [0.0, 0.0]
        else:
            # For planets, assume a perfectly circular orbit for initial stability.
            # Position the planet on the positive x-axis.
            pos = [radius, 0.0]
            
            # Calculate the required tangential velocity using Keplerian orbital mechanics:
            # v_orbital = sqrt(G * M_sun / r)
            # The velocity vector is directed entirely along the positive y-axis, 
            # ensuring it is strictly perpendicular to the position vector.
            v_mag = np.sqrt((system.G * sun_mass) / radius)
            vel = [0.0, v_mag]
            
        body = CelestialBody(name=name, mass=mass, 
                             initial_position=pos, 
                             initial_velocity=vel, 
                             colour=colour)
        system.add_body(body)


def run_animation():
    """
    Initializes and executes the 2D scientific visualization of the Solar System.
    Provides a real-time, visual verification of the N-body orbital dynamics and 
    the numerical stability of the Beeman integrator.
    """
    
    # 1. Initialize the physical environment
    G_new = 1.185e-4 
    system = SolarSystem(G=G_new)
    load_system_from_json(JSON_FILE_PATH, system)
    dt = 0.002 
    
    # 2. Configure the Matplotlib canvas for deep-space aesthetics
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-6, 6) 
    ax.set_ylim(-6, 6)
    
    # CRITICAL: Set aspect ratio to 'equal' to ensure circular orbits are not 
    # visually distorted into ellipses by the window dimensions.
    ax.set_aspect('equal') 
    ax.set_facecolor('#050505') 
    ax.set_title("Solar System Orbital Dynamics (Beeman Integrator)", color='white')

    # Arrays to store Matplotlib artist objects
    lines = []
    points = []
    
    # Data structures to store historical coordinates for the orbital trail effect
    x_history = [[] for _ in system.bodies]
    y_history = [[] for _ in system.bodies]

    # Initialize plotting elements for each celestial body
    for body in system.bodies:
        size = 12 if body.name.lower() == 'sun' else 6
        
        # Orbital trail (alpha=0.4 for transparency)
        line, = ax.plot([], [], '-', color=body.colour, alpha=0.4, linewidth=1.5)
        lines.append(line)
        
        # Celestial body marker
        point, = ax.plot([], [], 'o', color=body.colour, label=body.name, markersize=size)
        points.append(point)

    # Configure legend layout
    legend = ax.legend(loc="upper right", facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    # Time-scaling factor: Decouples the simulation time-step (dt) from the rendering frame rate.
    # The physics engine calculates 10 high-precision steps before rendering 1 visual frame.
    steps_per_frame = 10 
    
    def animate(frame):
        # Advance the physics engine
        for _ in range(steps_per_frame):
            system.step_forward(dt, method="beeman")
            
        # Update graphical data
        for i, body in enumerate(system.bodies):
            x_history[i].append(body.position[0])
            y_history[i].append(body.position[1])
            
            # Memory management: Cap the trail length to 800 data points.
            # Prevents Memory Leak during long-duration simulations.
            if len(x_history[i]) > 800:
                x_history[i].pop(0)
                y_history[i].pop(0)
                
            lines[i].set_data(x_history[i], y_history[i])
            points[i].set_data([body.position[0]], [body.position[1]])
            
        return lines + points
    
    # Global declaration prevents Python's Garbage Collector from destroying the animation 
    # object prematurely in certain IDEs (e.g., Spyder, Jupyter).
    global ani 
    ani = animation.FuncAnimation(fig, animate, frames=2000, interval=20, blit=False)
    plt.show()
    
def run_experiment_1():
    """
    Experiment 1: Empirical Determination of Orbital Periods.
    This experiment validates the simulation accuracy by measuring the time taken 
    for each planet to complete one full revolution around the Sun.
    """
   
    # Re-initialize a clean system to ensure no state pollution from previous runs
    system = SolarSystem(G=1.185e-4)
    load_system_from_json(JSON_FILE_PATH, system)
    
    # Numerical Precision Note:
    # The time-step dt defines the temporal resolution of our 'event detection'.
    # Since we only check the position at discrete intervals (dt=0.002), the planet 
    # will likely cross the x-axis *between* two steps. 
    # Thus, the measured period will have a small discretization error of approx. ±dt.
    dt = 0.002
    current_time = 0.0
    
    # Simulate for 12 Earth years (sufficient duration for Jupiter to complete one orbit)
    max_years = 12.0 
    
    planets_completed = 0
    total_planets = len(system.bodies) - 1 # Excluding the central Sun
    
    while current_time <= max_years:
        system.step_forward(dt, method="beeman")
        current_time += dt
        
        for body in system.bodies:
            if body.name.lower() == 'sun':
                continue
                
            # Core Detection Logic: Zero-Crossing Algorithm
            # We identify a completed orbit when the planet crosses the positive x-axis.
            # This is triggered when the y-coordinate transitions from negative to positive.
            if body.prev_y < 0 and body.position[1] >= 0:
                # Ensure we only capture the first completion event
                if body.period is None: 
                    body.period = current_time
                    planets_completed += 1
                    
            # Critical State Update: Store current y-pos for comparison in the next iteration
            body.prev_y = body.position[1]
            
        # Optimization: Terminate early if all planetary periods have been determined
        if planets_completed == total_planets:
            break
            
    print("\n--- Experiment 1: Orbital Periods ---")
    for body in system.bodies:
        if body.name.lower() != 'sun':
            print(f"{body.name} Period: {body.period:.4f} Earth Years")
    
def run_experiment_2():
    """
    Experiment 2: Energy Conservation and Algorithmic Stability.
    This experiment systematically compares the long-term energy conservation 
    properties of three numerical integrators: Explicit Euler, Euler-Cromer, and Beeman.
    It demonstrates the secular energy drift in non-symplectic methods versus 
    the bounded energy oscillations in symplectic/predictor-corrector methods.
    """

    # Define the integration schemes to be benchmarked
    methods = ["euler", "euler-cromer", "beeman"]
    labels = ["Forward Euler (Explicit)", "Euler-Cromer (Semi-Implicit)", "Beeman (Predictor-Corrector)"]
    colors = ["red", "green", "blue"]

    dt = 0.002
    # Simulate over 100 Earth years to clearly observe secular energy drift.
    # While longer times (e.g., 500 years) are possible, 100 years is sufficient 
    # to demonstrate the catastrophic failure of the Forward Euler method.
    max_years = 100.0 
    steps = int(max_years / dt)
    
    # Data Downsampling: Record energy every 100 steps.
    # This prevents memory exhaustion and extremely slow rendering times during plotting,
    # while preserving the macroscopic physical trends perfectly.
    record_interval = 100 

    energy_data = {m: [] for m in methods}
    time_data = []

    # Iterate through each numerical integration method
    for m_idx, method in enumerate(methods):
        print(f"Benchmarking {labels[m_idx]} method (please wait)...")
        
        # Re-initialize a clean, unpolluted physical system for a fair comparison
        system = SolarSystem(G=1.185e-4)
        load_system_from_json(JSON_FILE_PATH, system)
        
        current_time = 0.0
        times = []
        energies = []

        for step in range(steps):
            system.step_forward(dt, method=method)
            current_time += dt

            # Periodic energy sampling
            if step % record_interval == 0:
                k, u = system.calculate_energy()
                times.append(current_time)
                energies.append(k + u) # Total Mechanical Energy E = K + U

        energy_data[method] = energies
        # Time array only needs to be populated once since all methods use the same dt
        if not time_data:
            time_data = times

    # ------------------ Visualization Section ------------------
    
    # Plot 1: Macroscopic Comparison (The "Big Picture")
    # Demonstrates the massive artificial energy gain inherent to the Forward Euler method.
    plt.figure(figsize=(10, 6))
    for m_idx, method in enumerate(methods):
        plt.plot(time_data, energy_data[method], color=colors[m_idx], label=labels[m_idx], linewidth=1.5)

    plt.title("Total System Energy vs. Time (Integration Scheme Comparison)")
    plt.xlabel("Time (Earth Years)")
    plt.ylabel("Total Mechanical Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Microscopic Detail of the Beeman Integrator
    # Because Euler's error is so large, it compresses the y-axis scale in Plot 1.
    # We isolate Beeman here so Matplotlib auto-scales the y-axis to reveal the 
    # extremely tiny, bounded periodic fluctuations characteristic of stable orbits.
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, energy_data["beeman"], color="blue", label="Beeman Only")
    plt.title("Total Energy vs. Time (Beeman Integrator - Microscopic View)")
    plt.xlabel("Time (Earth Years)")
    plt.ylabel("Total Mechanical Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment_3():
    """
    Experiment 3: Orbital Transfer Optimization (Mars Mission).
    This experiment utilizes a parameter sweep to determine the optimal initial 
    velocity boost (Delta-v) required for a spacecraft to travel from Earth to Mars.
    The goal is to approximate a Hohmann transfer orbit by finding the minimum 
    Euclidean distance between the spacecraft and Mars.
    """
    print("\n--- Executing Experiment 3: Mars Transfer Window Optimization ---")
    
    # Parameter Sweep Setup: 
    # Systematically testing 10 distinct Delta-v values uniformly distributed 
    # between 0.4 and 0.8 AU/year to find the optimal launch speed.
    test_speeds = np.linspace(0.4, 0.8, 10)
    min_distances = []
    
    dt = 0.002
    max_years = 2.0 # Simulate for 2 Earth years (sufficient for a one-way Mars transit)
    
    best_speed = 0
    absolute_min_dist = float('inf')
    best_time = 0
    
    print("Executing automated parameter sweep (Please wait)...\n")
    
    # 1. Automated Grid Search
    for speed in test_speeds:
        # Re-initialize the N-body system for each test to ensure independent simulations
        system = SolarSystem(G=1.185e-4)
        load_system_from_json(JSON_FILE_PATH, system)
        
        earth = next(b for b in system.bodies if b.name.lower() == 'earth')
        mars = next(b for b in system.bodies if b.name.lower() == 'mars')
        
        # Spacecraft Launch Kinematics:
        # Positioned slightly outside Earth's center to avoid division-by-zero singularity.
        # The boost (speed) is added tangentially to Earth's orbital velocity (y-direction).
        sat_pos = [earth.position[0] + 0.005, earth.position[1]] 
        sat_vel = [earth.velocity[0], earth.velocity[1] + speed] 
        satellite = CelestialBody(name="Satellite", mass=1e-10, initial_position=sat_pos, initial_velocity=sat_vel)
        system.add_body(satellite)
        
        current_time = 0.0
        local_min_dist = float('inf')
        local_best_time = 0
        
        # Time evolution and continuous distance monitoring
        while current_time <= max_years:
            system.step_forward(dt, method="beeman")
            current_time += dt
            
            # Calculate the instantaneous Euclidean distance to Mars
            dist_to_mars = np.linalg.norm(satellite.position - mars.position)
            if dist_to_mars < local_min_dist:
                local_min_dist = dist_to_mars
                local_best_time = current_time
                
        min_distances.append(local_min_dist)
        print(f"Tested Boost: {speed:.3f} AU/yr -> Min Distance to Mars: {local_min_dist:.4f} AU")
        
        # Track the global minimum across all parameter iterations
        if local_min_dist < absolute_min_dist:
            absolute_min_dist = local_min_dist
            best_speed = speed
            best_time = local_best_time

    # 2. Optimization Landscape Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(test_speeds, min_distances, marker='o', color='purple')
    plt.axvline(best_speed, color='red', linestyle='--', label=f'Optimal Boost: {best_speed:.3f}')
    plt.title("Minimum Distance to Mars vs. Launch Boost Speed (Delta-v)")
    plt.xlabel("Boost Speed / Delta-v (AU/Year)")
    plt.ylabel("Minimum Distance to Mars (AU)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Final Mission Report Analysis
    print("\n--- Optimization Results ---")
    print(f"Optimal Boost (Delta-v): {best_speed:.3f} AU/yr")
    print(f"Closest Approach: {absolute_min_dist:.4f} AU")
    print(f"One-way Transit Time: {best_time:.4f} Years (Approx. {best_time * 365:.1f} Days)")
    
    # Real-world benchmark against NASA's Perseverance Rover
    print(f"NASA Perseverance transit time: ~203 Days.")
    print(f"Your probe is {'FASTER' if (best_time*365) < 203 else 'SLOWER'} by {abs((best_time*365) - 203):.1f} Days!")
# ==========================================
# Execution Control Panel (Main Entry Point)
# ==========================================
if __name__ == "__main__":
    
    # INSTRUCTIONS: 
    # Uncomment the specific function call below to execute that module.
    # CRITICAL: Due to Matplotlib's event loop (plt.show()), executing multiple 
    # modules sequentially may cause the main thread to block. 
    # Please run strictly ONE module at a time.
    
    # -------------------------
    # Module 0: Real-time 2D Visualization (Orbital Dynamics)
    # -------------------------
    #run_animation()

    # -------------------------
    # Module 1: Empirical Determination of Orbital Periods
    # -------------------------
    #run_experiment_1()
    
    # -------------------------
    # Module 2: Numerical Stability and Energy Benchmarking
    # -------------------------
    #run_experiment_2()
    
    # -------------------------
    # Module 3: Parameter Sweep for Hohmann Transfer Optimization
    # -------------------------
    run_experiment_3()