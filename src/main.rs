#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::time::{Duration, Instant};

mod activation_functions;
mod cost_functions;
mod layer;
mod matrix;
mod network;

use activation_functions::*;
use cost_functions::*;
use layer::*;
use matrix::*;
use network::*;

fn main() {
    let n_train: usize = 100;
    let in_size: usize = 28 * 28;
    let hidden_size: usize = 32;
    let out_size: usize = 1;
    let batch_size: usize = 64;

    let start = Instant::now();
    let x_train: Vec<Matrix> = (0..n_train)
        .into_iter()
        .map(|i| Matrix::randn(batch_size, in_size, 0.0, 1.0))
        .collect();
    let y_train: Vec<Matrix> = (0..n_train)
        .into_iter()
        .map(|i| Matrix::new(batch_size, out_size, 0.0))
        .collect();

    let duration = start.elapsed();
    println!("Time elapsed in create dataset is: {:?}", duration);

    let config = (
        (batch_size, "cross_entropy"),
        vec![
            ("dense", in_size, hidden_size, "sigmoid"),
            ("dense", hidden_size, hidden_size, "sigmoid"),
            ("dense", hidden_size, out_size, "sigmoid"),
        ],
    );
    let mut network = Network::new(&config);

    let start = Instant::now();
    network.forward_pass(&x_train);
    let duration = start.elapsed();
    println!("Time elapsed in a forward pass is: {:?}", duration);

    let start = Instant::now();
    network.backward_pass(&x_train, &y_train);
    let duration = start.elapsed();
    println!("Time elapsed in a backward pass is: {:?}", duration);
}
