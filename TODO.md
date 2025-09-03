
1. Display more heuristics visually on the frontend.
2. Make apex animal red. (Animal with highest fitness function)
3. Parralelize game into multiple games with different configs running. The best being chosen after n games with j timesteps.
4. Make front end interactive. Possibly launch on website.
5. Quadruple the enivronment size and run back the experiment.

Struct Config: # Used for parrallel evolution.
Now, How the fuck do we make it work?

will contain: fov_range, fov_distance, # animals, # foods, lifetime (before evolution), brain size, mutation rate,
              selection_algorithm, mutation_magnitude

Some other selection algorithms that can be used are:
Botlzman scaling: T
Truncation Selection: Top %
Elitisim: top K
Tournoment Selection: K subset size.

# Implement parralell simulations that can be displayed on front end

# Make a grid in js that shows these simulations

# Make a Config for the current simulation and see how it holds

# Try varying it a bit with the simple parameters

# Implement the new selection algorithms

# Add the selection algorithms to the Config

