use std::f32::consts::PI;
use std::fmt;
use once_cell::sync::Lazy;
use lib_genetic_algorithm as ga;

#[derive(Debug, Clone)]
pub enum EvolutionaryAlgorithm {
    StandardGA {
        selection: SelectionType,
        crossover: CrossoverType,
        mutation: MutationType,
    },
    ConservativeGA {
        selection: SelectionType,
        crossover: CrossoverType,
        mutation: MutationType,
    },
    AggressiveGA {
        selection: SelectionType,
        crossover: CrossoverType,
        mutation: MutationType,
    },
}

#[derive(Debug, Clone)]
pub enum SelectionType {
    RouletteWheel,
    Tournament { size: usize },
    Rank,
}

#[derive(Debug, Clone)]
pub enum CrossoverType {
    Uniform,
    SinglePoint,
    Arithmetic { alpha: f32 },
}

#[derive(Debug, Clone)]
pub enum MutationType {
    Gaussian { chance: f32, coeff: f32 },
    Uniform { chance: f32, min: f32, max: f32 },
    Cauchy { chance: f32, scale: f32 },
}

impl EvolutionaryAlgorithm {
    pub fn create_genetic_algorithm(&self) -> ga::GeneticAlgorithm<ga::RouletteWheelSelection> {
        match self {
            EvolutionaryAlgorithm::StandardGA { selection: _, crossover, mutation } => {
                Self::create_ga_with_roulette(crossover, mutation)
            }
            EvolutionaryAlgorithm::ConservativeGA { selection: _, crossover, mutation } => {
                Self::create_ga_with_roulette(crossover, mutation)
            }
            EvolutionaryAlgorithm::AggressiveGA { selection: _, crossover, mutation } => {
                Self::create_ga_with_roulette(crossover, mutation)
            }
        }
    }

    fn create_ga_with_roulette(
        crossover: &CrossoverType,
        mutation: &MutationType,
    ) -> ga::GeneticAlgorithm<ga::RouletteWheelSelection> {
        // Create concrete instances instead of trait objects
        match (crossover, mutation) {
            (CrossoverType::Uniform, MutationType::Gaussian { chance, coeff }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::UniformCrossOver,
                    ga::GuassianMutation::new(*chance, *coeff),
                )
            }
            (CrossoverType::SinglePoint, MutationType::Gaussian { chance, coeff }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::SinglePointCrossOver,
                    ga::GuassianMutation::new(*chance, *coeff),
                )
            }
            (CrossoverType::Arithmetic { alpha }, MutationType::Gaussian { chance, coeff }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::ArithmeticCrossOver::new(*alpha),
                    ga::GuassianMutation::new(*chance, *coeff),
                )
            }
            (CrossoverType::Uniform, MutationType::Uniform { chance, min, max }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::UniformCrossOver,
                    ga::UniformMutation::new(*chance, *min, *max),
                )
            }
            (CrossoverType::SinglePoint, MutationType::Uniform { chance, min, max }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::SinglePointCrossOver,
                    ga::UniformMutation::new(*chance, *min, *max),
                )
            }
            (CrossoverType::Arithmetic { alpha }, MutationType::Uniform { chance, min, max }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::ArithmeticCrossOver::new(*alpha),
                    ga::UniformMutation::new(*chance, *min, *max),
                )
            }
            (CrossoverType::Uniform, MutationType::Cauchy { chance, scale }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::UniformCrossOver,
                    ga::CauchyMutation::new(*chance, *scale),
                )
            }
            (CrossoverType::SinglePoint, MutationType::Cauchy { chance, scale }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::SinglePointCrossOver,
                    ga::CauchyMutation::new(*chance, *scale),
                )
            }
            (CrossoverType::Arithmetic { alpha }, MutationType::Cauchy { chance, scale }) => {
                ga::GeneticAlgorithm::new(
                    ga::RouletteWheelSelection,
                    ga::ArithmeticCrossOver::new(*alpha),
                    ga::CauchyMutation::new(*chance, *scale),
                )
            }
        }
    }
}

// Used for running initalizing simulations. Helpful for finding optimal params
#[derive(Debug, Clone)]
pub struct Config {
    pub fov_range: f32,           //done
    pub fov_angle: f32,           //done
    pub animal_count: usize,      // 20,500 (done)
    pub food_count: usize,        //  20, 500 (done)
    pub generation_length: usize, // number of steps before evolution (done)
    pub num_cells: usize,         // analogous to brain size. Determines magnitued of neurons to have on
    pub evolutionary_algorithm: EvolutionaryAlgorithm,
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "fov_range: {}, fov_angle: {}, animal_count: {}, food_count: {}, generation_length: {}, algorithm: {:?}",
            self.fov_range,
            self.fov_angle,
            self.animal_count,
            self.food_count,
            self.generation_length,
            self.evolutionary_algorithm
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
    algorithms: Vec<EvolutionaryAlgorithm>,
}

static DEFAULT_CONFIG_SPACE: Lazy<ConfigSpace> = Lazy::new(|| ConfigSpace {
    fov_ranges: range_with_step(0.1, 1.0, 0.2), // 0.1 to 1.0 in steps of 0.2
    fov_angles: range_with_step(0.25 * PI, 2.0 * PI, 0.25 * PI), // π/4 to 2π in steps of π/4
    animal_counts: range_with_step_usize(10, 50, 20), // 10 to 50 in steps of 20
    food_counts: range_with_step_usize(20, 60, 20), // 20 to 60 in steps of 20
    generation_lengths: range_with_step_usize(1000, 3000, 1000), // 1000 to 3000 in steps of 1000
    cell_sizes: range_with_step_usize(8, 10, 1), // 8 to 10 in steps of 1
    algorithms: vec![
        // Standard GA - balanced approach
        EvolutionaryAlgorithm::StandardGA {
            selection: SelectionType::RouletteWheel,
            crossover: CrossoverType::Uniform,
            mutation: MutationType::Gaussian { chance: 0.01, coeff: 0.03 },
        },
        // Conservative GA - less mutation, more exploitation
        EvolutionaryAlgorithm::ConservativeGA {
            selection: SelectionType::Tournament { size: 3 },
            crossover: CrossoverType::SinglePoint,
            mutation: MutationType::Gaussian { chance: 0.005, coeff: 0.01 },
        },
        // Aggressive GA - more mutation, more exploration
        EvolutionaryAlgorithm::AggressiveGA {
            selection: SelectionType::Rank,
            crossover: CrossoverType::Arithmetic { alpha: 0.5 },
            mutation: MutationType::Cauchy { chance: 0.05, scale: 0.1 },
        },
    ],
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
                                for algorithm in &self.algorithms {
                                    configs.push(Config {
                                        fov_range,
                                        fov_angle,
                                        animal_count,
                                        food_count,
                                        generation_length,
                                        num_cells,
                                        evolutionary_algorithm: algorithm.clone(),
                                    })
                                }
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
