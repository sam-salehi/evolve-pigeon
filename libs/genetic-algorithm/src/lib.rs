use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};

pub trait Individual {
    fn create(chromosome: Chromosome) -> Self;
    fn fitness(&self) -> f32;
    fn chromosome(&self) -> &Chromosome;
}

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }
    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> (Vec<I>, Statistics)
    where
        I: Individual,
    {
        assert!(!population.is_empty());
        let new_population = (0..population.len())
            .map(|_| {
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();
                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);
                self.mutation_method.mutate(rng, &mut child);
                I::create(child)
            })
            .collect();
        let stats = Statistics::new(population);

        (new_population, stats)
    }
}

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}

pub struct RouletteWheelSelection;

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("Recieved empty population. Possibly called early population to early.")
    }
}

pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    // Takes the best creature and pushes on to next iteration.
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

impl SelectionMethod for TournamentSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        let tournament_size = self.tournament_size.min(population.len());
        let tournament: Vec<_> = population.choose_multiple(rng, tournament_size).collect();

        tournament
            .into_iter()
            .max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
            .expect("Tournament should not be empty")
    }
}

pub struct RankSelection;

impl SelectionMethod for RankSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        // Create indices and sort by fitness
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            population[a]
                .fitness()
                .partial_cmp(&population[b].fitness())
                .unwrap()
        });

        // Create rank-based weights (linear ranking)
        let weights: Vec<f32> = (1..=indices.len()).map(|rank| rank as f32).collect();

        // Select based on rank weights
        let selected_index = indices
            .choose_weighted(rng, |&index| {
                weights[indices.iter().position(|&x| x == index).unwrap()]
            })
            .expect("Population should not be empty");

        &population[*selected_index]
    }
}

use std::ops::Index;

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    // Allows for basic indexing.
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}
impl FromIterator<f32> for Chromosome {
    // Allows to collect into my type.
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    // Converts type into iterator.
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome;
}

#[derive(Clone, Debug)]
pub struct UniformCrossOver;

impl CrossoverMethod for UniformCrossOver {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        parent_a
            .iter()
            .zip(parent_b.iter())
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SinglePointCrossOver;

impl CrossoverMethod for SinglePointCrossOver {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        if parent_a.len() <= 1 {
            return parent_a.clone();
        }

        let crossover_point = rng.gen_range(1..parent_a.len());

        let mut child_genes = Vec::with_capacity(parent_a.len());
        child_genes.extend(parent_a.iter().take(crossover_point));
        child_genes.extend(parent_b.iter().skip(crossover_point));

        child_genes.into_iter().collect()
    }
}

#[derive(Clone, Debug)]
pub struct ArithmeticCrossOver {
    alpha: f32,
}

impl ArithmeticCrossOver {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl CrossoverMethod for ArithmeticCrossOver {
    fn crossover(
        &self,
        _rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        parent_a
            .iter()
            .zip(parent_b.iter())
            .map(|(&a, &b)| self.alpha * a + (1.0 - self.alpha) * b)
            .collect()
    }
}

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

pub struct GuassianMutation {
    chance: f32,
    coeff: f32,
}

impl GuassianMutation {
    pub fn new(chance: f32, coeff: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);
        Self { chance, coeff }
    }
}

impl MutationMethod for GuassianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
            if rng.gen_bool(self.chance as f64) {
                *gene += sign * self.coeff * rng.r#gen::<f32>();
            }
        }
    }
}

pub struct UniformMutation {
    chance: f32,
    min: f32,
    max: f32,
}

impl UniformMutation {
    pub fn new(chance: f32, min: f32, max: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);
        assert!(min <= max);
        Self { chance, min, max }
    }
}

impl MutationMethod for UniformMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            if rng.gen_bool(self.chance as f64) {
                *gene = rng.gen_range(self.min..=self.max);
            }
        }
    }
}

pub struct CauchyMutation {
    chance: f32,
    scale: f32,
}

impl CauchyMutation {
    pub fn new(chance: f32, scale: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);
        Self { chance, scale }
    }
}

impl MutationMethod for CauchyMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            if rng.gen_bool(self.chance as f64) {
                // Generate Cauchy-distributed random number
                let u = rng.r#gen::<f32>() - 0.5;
                let cauchy_value = self.scale * (u / (1.0 - u.abs())).tan();
                *gene += cauchy_value;
            }
        }
    }
}

pub struct Statistics {
    pub min_fitness: f32,
    pub max_fitness: f32,
    pub avg_fitness: f32,
}

impl Statistics {
    fn new<I>(population: &[I]) -> Self
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let mut min_fitness = population[0].fitness();
        let mut max_fitness = min_fitness;
        let mut sum_fitness = 0.0;

        for i in population {
            let fitness = i.fitness();
            min_fitness = min_fitness.min(fitness);
            max_fitness = max_fitness.max(fitness);
            sum_fitness += fitness;
        }

        Self {
            min_fitness,
            max_fitness,
            avg_fitness: sum_fitness / (population.len() as f32),
        }
    }
}
// TODO

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    #[derive(Clone, Debug, PartialEq)]

    pub enum TestIndividual {
        WithChromosome { chromosome: Chromosome },
        WithFitness { fitness: f32 },
    }

    impl TestIndividual {
        fn new(fitness: f32) -> Self {
            Self::WithFitness { fitness }
        }
    }

    impl PartialEq for Chromosome {
        fn eq(&self, other: &Self) -> bool {
            approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
        }
    }

    impl Individual for TestIndividual {
        fn create(chromosome: Chromosome) -> Self {
            Self::WithChromosome { chromosome }
        }

        fn fitness(&self) -> f32 {
            match self {
                Self::WithChromosome { chromosome } => chromosome.iter().sum(),
                Self::WithFitness { fitness } => *fitness,
            }
        }

        fn chromosome(&self) -> &Chromosome {
            match self {
                Self::WithChromosome { chromosome } => chromosome,
                Self::WithFitness { .. } => {
                    panic!("Not supported for TestIndividual::WithFitness")
                }
            }
        }
    }

    #[test]
    fn genetic_algorithm() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection,
            UniformCrossOver,
            GuassianMutation::new(0.5, 0.5),
        );

        fn individual(genes: &[f32]) -> TestIndividual {
            TestIndividual::create(genes.iter().cloned().collect())
        }

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]),
            individual(&[1.0, 1.0, 1.0]),
            individual(&[1.0, 2.0, 1.0]),
            individual(&[1.0, 2.0, 4.0]),
        ];

        for _ in 0..10 {
            let (new_population, _) = ga.evolve(&mut rng, &population);
            population = new_population
        }

        let expected_population = vec![
            individual(&[0.44769490, 2.0648358, 4.3058133]),
            individual(&[1.21268670, 1.5538777, 2.8869110]),
            individual(&[1.06176780, 2.2657390, 4.4287640]),
            individual(&[0.95909685, 2.4618788, 4.0247330]),
        ];

        assert_eq!(population, expected_population)
    }

    #[test]
    fn roulette_wheel_selection() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];

        let mut actual_histogram = BTreeMap::new();

        for _ in 0..100 {
            let fitness = RouletteWheelSelection
                .select(&mut rng, &population)
                .fitness() as i32;

            *actual_histogram.entry(fitness).or_insert(0) += 1
        }

        let expected_histogram = BTreeMap::from_iter([(1, 11), (2, 18), (3, 21), (4, 50)]);

        assert_eq!(actual_histogram, expected_histogram);
    }

    #[test]
    fn uniform_crossover() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let parent_a: Chromosome = (1..=100).map(|n| n as f32).collect();
        let parent_b: Chromosome = (1..=100).map(|n| -n as f32).collect();
        let child = UniformCrossOver.crossover(&mut rng, &parent_a, &parent_b);

        let diff_a = child.iter().zip(parent_a).filter(|(c, p)| *c != p).count();
        let diff_b = child.iter().zip(parent_b).filter(|(c, p)| *c != p).count();

        assert_eq!(diff_a, 49);
        assert_eq!(diff_b, 51);
    }

    mod guassian_mutation {
        use super::*;

        fn actual(chance: f32, coeff: f32) -> Vec<f32> {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

            GuassianMutation::new(chance, coeff).mutate(&mut rng, &mut child);
            child.into_iter().collect()
        }

        mod given_zero_chance {
            use approx::assert_relative_eq;

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(0.0, coeff)
            }

            mod and_zero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = super::actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }

        mod given_fifty_fifty_chance {
            use approx::assert_relative_eq;

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(0.5, coeff)
            }

            mod and_zero_coefficient {
                use super::*;
                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;
                #[test]
                fn slightly_changes_the_original_chromosome() {
                    let actual = actual(0.2);
                    let expected = vec![1.0, 1.91025, 3.0, 4.0638723, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }

        mod given_max_chance {
            use approx::{assert_relative_eq, assert_relative_ne};

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(1.0, coeff)
            }

            mod and_zero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;

                #[test]
                fn entirely_changes_the_original_chromosome() {
                    let actual = actual(1.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                    let _ = actual
                        .iter()
                        .zip(expected.iter())
                        .map(|(a, e)| assert_relative_ne!(a, e));
                }
            }
        }
    }
}
