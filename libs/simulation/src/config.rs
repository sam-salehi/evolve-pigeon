use std::f32::consts::PI;
use std::fmt;
use once_cell::sync::Lazy;

// Used for running initalizing simulations. Helpful for finding optimal params
#[derive(Debug, Clone)]
pub struct Config {
    pub fov_range: f32,           //done
    pub fov_angle: f32,           //done
    pub animal_count: usize,      // 20,500 (done)
    pub food_count: usize,        //  20, 500 (done)
    pub generation_length: usize, // number of steps before evolution (done)
    pub num_cells: usize,         // analogous to brain size. Determines magnitued of neurons to have on
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "fov_range: {}, fov_angle: {}, animal_count: {}, food_count: {},generation_length: {}",
            self.fov_range,
            self.fov_angle,
            self.animal_count,
            self.food_count,
            self.generation_length
        )
    }
}

fn range_with_step(start: f32, end: f32, step: f32) -> Vec<f32> {
    let mut values = Vec::new();
    let mut x = start;

    while x <= end {
        values.push(x);
        x += step;
    }

    values
}

fn range_with_step_usize(start: usize, end: usize, step: usize) -> Vec<usize> {
    (start..=end).step_by(step).collect()
}

// used for generating multiple simulations.
// TODO: move somewhere else.
struct ConfigSpace {
    fov_ranges: Vec<f32>,           // 0.1 to 1
    fov_angles: Vec<f32>,           // 0.25pi to 2pi
    animal_counts: Vec<usize>,      // 10 ti 80
    food_counts: Vec<usize>,        // 20 to 100
    generation_lengths: Vec<usize>, // 40 to 80
    cell_sizes: Vec<usize>,         // 5 to 30
}

static DEFAULT_CONFIG_SPACE: Lazy<ConfigSpace> = Lazy::new(|| ConfigSpace {
    fov_ranges: range_with_step(0.1, 1.0, 1.0), // 0.2
    fov_angles: range_with_step(0.25 * PI, 2.0 * PI, 2.0 * PI), // 0.25pi
    animal_counts: range_with_step_usize(10, 90, 20),
    food_counts: range_with_step_usize(20, 100, 100),
    generation_lengths: range_with_step_usize(500, 5000, 500),
    cell_sizes: range_with_step_usize(8, 10, 2), //TODO: Issue with size of
                                                 //weight. It has to be nine.
                                                 // Should deinitely work, its
                                                 // most important metric.
});

impl ConfigSpace {
    fn all_combinations(&self) -> Vec<Config> {
        let mut configs = Vec::new();

        for &fov_range in &self.fov_ranges {
            for &fov_angle in &self.fov_angles {
                for &animal_count in &self.animal_counts {
                    for &food_count in &self.food_counts {
                        for &generation_length in &self.generation_lengths {
                            for &num_cells in &self.cell_sizes {
                                configs.push(Config {
                                    fov_range,
                                    fov_angle,
                                    animal_count,
                                    food_count,
                                    generation_length,
                                    num_cells,
                                })
                            }
                        }
                    }
                }
            }
        }
        configs
    }
}

pub fn get_default_configs() -> Vec<Config> {
    DEFAULT_CONFIG_SPACE.all_combinations()
}
