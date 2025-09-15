use lib_simulation as sim;
use rand::prelude::*;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Config {
    pub fov_range: f32,
    pub fov_angle: f32,
    pub animal_count: usize,
    pub food_count: usize,
    pub generation_length: usize,
    pub num_cells: usize,
}

#[wasm_bindgen]
impl Config {
    #[wasm_bindgen(constructor)]
    pub fn new(
        fov_range: f32,
        fov_angle: f32,
        animal_count: usize,
        food_count: usize,
        generation_length: usize,
        num_cells: usize,
    ) -> Self {
        Self {
            fov_range,
            fov_angle,
            animal_count,
            food_count,
            generation_length,
            num_cells,
        }
    }
}

impl From<Config> for sim::Config {
    fn from(config: Config) -> Self {
        sim::Config {
            fov_range: config.fov_range,
            fov_angle: config.fov_angle,
            animal_count: config.animal_count,
            food_count: config.food_count,
            generation_length: config.generation_length,
            num_cells: config.num_cells,
        }
    }
}

#[wasm_bindgen]
pub struct ParallelEngine {
    rng: ThreadRng,
    eng: sim::ParallelEngine,
}

#[wasm_bindgen]
impl ParallelEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let rng = thread_rng();
        let eng = sim::ParallelEngine::new();

        Self { rng, eng }
    }

    pub fn step_all(&mut self) {
        self.eng.step_all();
    }

    pub fn train(&mut self, id: &str) {
        let id = Uuid::parse_str(id).unwrap();
        self.eng.train(id);
    }

    pub fn worlds(&mut self) -> Vec<World> {
        self.eng
            .worlds()
            .into_iter()
            .map(|(id, world)| World::from_with_id(world, id))
            .collect()
    }
}

#[wasm_bindgen]
pub struct Simulation {
    rng: ThreadRng,
    sim: sim::Simulation,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let rng = thread_rng();
        let sim = sim::Simulation::random();
        Self { rng, sim }
    }

    #[wasm_bindgen]
    pub fn new_with_config(config: &Config) -> Self {
        let rng = thread_rng();
        let sim_config = sim::Config::from(config.clone());
        let sim = sim::Simulation::from(sim_config);
        Self { rng, sim }
    }

    pub fn world(&self) -> World {
        World::from(self.sim.world())
    }

    pub fn step(&mut self) {
        self.sim.step();
    }

    pub fn train(&mut self) -> String {
        let stats = self.sim.train();
        format!(
            "min={:.2}, max={:.2}, avg={:.2}",
            stats.min_fitness, stats.max_fitness, stats.avg_fitness,
        )
    }
}

pub trait FromWithId<T, Id> {
    fn from_with_id(value: T, id: Id) -> Self;
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct World {
    #[wasm_bindgen(getter_with_clone)]
    pub id: String,

    #[wasm_bindgen(getter_with_clone)]
    pub animals: Vec<Animal>,

    #[wasm_bindgen(getter_with_clone)]
    pub foods: Vec<Food>,
}

impl From<&sim::World> for World {
    fn from(world: &sim::World) -> Self {
        let animals = world.animals().iter().map(Animal::from).collect();
        let foods = world.foods().iter().map(Food::from).collect();
        Self {
            id: "null".to_string(),
            animals,
            foods,
        }
    }
}

impl FromWithId<&sim::World, String> for World {
    fn from_with_id(world: &sim::World, id: String) -> Self {
        let mut base = World::from(world);
        base.id = id;
        base
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Animal {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub satiation: usize,
}

impl From<&sim::Animal> for Animal {
    fn from(animal: &sim::Animal) -> Self {
        Self {
            x: animal.position().x,
            y: animal.position().y,
            rotation: animal.rotation().angle(),
            satiation: animal.satiation(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Food {
    pub x: f32,
    pub y: f32,
}

impl From<&sim::Food> for Food {
    fn from(food: &sim::Food) -> Self {
        Self {
            x: food.position().x,
            y: food.position().y,
        }
    }
}
