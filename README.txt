====================================================================
Solar System N-Body Simulation Project
====================================================================

1. FILES INCLUDED
--------------------------------------------------------------------
This submission contains the following files:

* project.py 
  The primary Python codebase. It contains the OOP physics engine (CelestialBody and SolarSystem classes) and the execution logic for the default animation and all three experiments.

* parameters_solar.json 
  The configuration data file containing the initial masses and orbital radii of the Sun, the inner planets, and Jupiter.

* README.txt 
  This instructional file.


2. HOW TO RUN THE CODE
--------------------------------------------------------------------
Prerequisites: 
The code requires standard scientific libraries to run: `numpy` and `matplotlib`.

Execution Instructions:
To ensure stability and prevent Matplotlib's `plt.show()` event loop from blocking subsequent code, the simulation is designed to run ONE module at a time. 

Please open `project.py` and scroll to the very bottom to the "Execution Control Panel" (under `if __name__ == "__main__":`). You can execute different configurations by uncommenting the specific function call you wish to test, while keeping the others commented out with a `#`.

* To run the Default Animation (Real-time 2D Visualization):
  Uncomment: run_animation()

* To run Experiment 1 (Empirical Determination of Orbital Periods):
  Uncomment: run_experiment_1()

* To run Experiment 2 (Numerical Stability and Energy Benchmarking):
  Uncomment: run_experiment_2()

* To run Experiment 3 (Mars Transfer Window Optimization):
  Uncomment: run_experiment_3()

Example: If you only want to view the Mars optimization sweep, the bottom of the script should look exactly like this:
    # run_animation()
    # run_experiment_1()
    # run_experiment_2()
    run_experiment_3()