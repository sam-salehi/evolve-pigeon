use crate::*;

#[derive(Debug)]
pub struct World {
    pub(crate) animals: Vec<Animal>,
    pub(crate) foods: Vec<Food>,
}

pub trait FromWithRng<T> {
    fn from_with_rng(cfg: &Config, rng: &mut dyn RngCore) -> T;
}

impl FromWithRng<World> for World {
    fn from_with_rng(cfg: &Config, rng: &mut dyn RngCore) -> Self {
        let animals = (0..cfg.animal_count)
            .map(|_| Animal::from_with_rng(cfg, rng))
            .collect();
        let foods = (0..cfg.food_count).map(|_| Food::random(rng)).collect();

        Self { animals, foods }
    }
}

impl World {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        // TODO: change to use Poisson disk sampling(Supersanpling)

        let animals = (0..40).map(|_| Animal::random(rng)).collect();
        let foods = (0..60).map(|_| Food::random(rng)).collect();

        Self { animals, foods }
    }

    pub fn animals(&self) -> &[Animal] {
        &self.animals
    }

    pub fn foods(&self) -> &[Food] {
        &self.foods
    }
}
