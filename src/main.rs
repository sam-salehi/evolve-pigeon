use std::ops::Deref;

use lib_simulation::ParallelEngine;

fn main() {
    let mut eng = ParallelEngine::new();

    println!("Beginning simulation");
    let score = eng.test_train(1000);

    let output = score
        .iter()
        .map(|c| c.to_string()) // now valid, because Config: Display
        .collect::<Vec<_>>()
        .join("\n");

    println!("{}", output);
    println!("Finishing sm");
}
