use self::animal_individual::*;
use lib_genetic_algorithm as ga;
use lib_neural_network as nn;
use nalgebra as na;
use rand::prelude::*;
use rand::thread_rng;
use rand::{Rng, RngCore};
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::prelude::*;
use uuid::Uuid;

pub use self::{animal::*, brain::*, eye::*, food::*, world::*};

mod animal;
mod animal_individual;
mod brain;
mod eye;
mod food;
mod world;

pub struct ParallelEngine {
    pub sims: Vec<Simulation>,
}

// Used for running initalizing simulations. Helpful for finding optimal params
pub struct Config {
    fov_range: f32,           //done
    fov_angle: f32,           //done
    animal_count: usize,      // 20,500 (done)
    food_count: usize,        //  20, 500 (done)
    generation_length: usize, // number of steps before evolution (done)
    brain_size: usize,
    mutation_rate: f32,
    selection_algorithm: todo!(), // Make a trait that gets implemented. Selection algorihtm
                                  // follows the trait
}

impl ParallelEngine {
    pub fn new() -> Self {
        let sims: Vec<Simulation> = (0..3).map(|_| Simulation::random()).collect();

        Self { sims }
    }

    pub fn step_all(&mut self) {
        let stats = self.sims.par_iter_mut().for_each(|sim| {
            sim.step();
        });
        stats
    }

    pub fn worlds(&mut self) -> Vec<(String, &World)> {
        self.sims
            .iter()
            .map(|sim| (sim.id().to_string(), sim.world()))
            .collect()
    }

    pub fn train(&mut self, id: Uuid) {
        if let Some(sim) = self.sims.iter_mut().find(|sim| sim.id == id) {
            sim.train();
        } else {
            panic!("Given id {:?} not found in running sims.", id)
        }
    }

    pub fn eval_all(&self) -> Vec<Logistic> {
        self.sims.par_iter().map(|sim| sim.logistics()).collect()
    }
}

use std::f32::consts::FRAC_PI_2;
const SPEED_MIN: f32 = 0.001;
const SPEED_MAX: f32 = 0.005;
const SPEED_ACCEL: f32 = 0.2;
const ROTATION_ACCEL: f32 = FRAC_PI_2;
const GENERATION_LENGTH: usize = 2500; // the simulation runs for GENERATION_LENGTH before evolving.

pub struct Simulation {
    id: Uuid,
    world: World,
    rng: ThreadRng,
    ga: ga::GeneticAlgorithm<ga::RouletteWheelSelection>,
    age: usize,
    generation_length: usize,
}

// Signifying traits are  to be used by rayon.
unsafe impl Send for Simulation {}
unsafe impl Sync for Simulation {}

impl From<Config> for Simulation {
    fn from(cfg: Config) -> Self {
        let mut rng = thread_rng();
        let world = World::from_with_rng(&cfg, &mut rng);
        let ga = ga::GeneticAlgorithm::from(&cfg);
        Self {
            id: Uuid::new_v4(),
            world,
            rng,
            ga,
            age: 0,
            generation_length: cfg.generation_length,
        }
    }
}

impl Simulation {
    pub fn random() -> Self {
        let world = World::random(&mut thread_rng());
        let rng = thread_rng();
        let ga = ga::GeneticAlgorithm::new(
            ga::RouletteWheelSelection,
            ga::UniformCrossOver,
            ga::GuassianMutation::new(0.01, 0.03),
        );

        Self {
            id: Uuid::new_v4(),
            world,
            rng,
            ga,
            age: 0,
            generation_length: GENERATION_LENGTH,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn step(&mut self) -> Option<ga::Statistics> {
        self.process_movements();
        self.process_brains();
        self.process_collisions();

        self.age += 1;

        if self.age > self.generation_length {
            let mut temp_rng = thread_rng();
            Some(self.evolve(&mut temp_rng))
        } else {
            None
        }
    }

    pub fn train(&mut self) -> ga::Statistics {
        loop {
            if let Some(summary) = self.step() {
                return summary;
            }
        }
    }

    fn evolve(&mut self, rng: &mut dyn RngCore) -> ga::Statistics {
        self.age = 0;
        let current_population: Vec<_> = self
            .world
            .animals
            .iter()
            .map(AnimalIndividual::from_animal)
            .collect();

        let (evolved_population, stats) = self.ga.evolve(rng, &current_population);

        self.world.animals = evolved_population
            .into_iter()
            .map(|individual| individual.into_animal(rng))
            .collect();

        for food in &mut self.world.foods {
            food.position = rng.r#gen();
        }

        stats
    }

    fn process_movements(&mut self) {
        for animal in &mut self.world.animals {
            animal.position += animal.rotation * na::Vector2::new(0.0, animal.speed);

            animal.position.x = na::wrap(animal.position.x, 0.0, 1.0);
            animal.position.y = na::wrap(animal.position.y, 0.0, 1.0);
        }
    }

    fn process_brains(&mut self) {
        for animal in &mut self.world.animals {
            let vision =
                animal
                    .eye
                    .process_vision(animal.position, animal.rotation, &self.world.foods);
            let response = animal.brain.nn.propogate(vision);
            let speed = response[0].clamp(-SPEED_ACCEL, SPEED_ACCEL);
            let rotation = response[1].clamp(-ROTATION_ACCEL, ROTATION_ACCEL);

            animal.speed = (animal.speed + speed).clamp(SPEED_MIN, SPEED_MAX);
            animal.rotation = na::Rotation2::new(animal.rotation.angle() + rotation);
        }
    }

    fn process_collisions(&mut self) {
        for animal in &mut self.world.animals {
            for food in &mut self.world.foods {
                let distance = na::distance(&animal.position, &food.position);

                if distance <= 0.01 {
                    animal.satiation += 1;
                    food.position = self.rng.r#gen();
                }
            }
        }
    }

    fn id(&self) -> Uuid {
        self.id
    }

    fn logistics(&self) -> Logistic {
        // score is the avrage fitness of animals.
        let mut max_fitness: usize = 0;
        let mut total_fitness: usize = 0;
        for animal in &self.world.animals {
            total_fitness += animal.satiation();
            max_fitness = max_fitness.max(animal.satiation());
        }

        Logistic {
            sim_id: self.id,
            total_fitness,
            avg_fitness: total_fitness / self.world.animals.len(),
            apex_fitness: max_fitness,
        }
    }
}

#[derive(Debug)]
pub struct Logistic {
    sim_id: Uuid,
    total_fitness: usize,
    avg_fitness: usize,
    apex_fitness: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_parralel_engine() {
        let mut eng = ParallelEngine::new();
        for _ in 0..10 {
            eng.step_all();
        }
        let eval = eng.eval_all();
        println!("{:?}", eval[0])
    }
}
