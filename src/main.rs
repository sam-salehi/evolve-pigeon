use std::ops::Deref;

use lib_simulation::ParallelEngine;

fn main() {
    let mut eng = ParallelEngine::new();

    // TODO: let the front end be able to load the better configs, so that I can actually see them.
    // TODO: maybe move around the colors, to get something abit distinct.
    // TODO: just display one simulation in frotend. That should be the end of it. Hopefull.

    println!("Beginning simulation");
    let score = eng.test_train(10000);

    let output = score
        .iter()
        .map(|c| format!("Score: {}. {}", c.0.to_string(), c.1.to_string())) // now valid, because Config Display
        .collect::<Vec<_>>()
        .join("\n");

    println!("{}", output);
    println!("Finishing sm");
}
