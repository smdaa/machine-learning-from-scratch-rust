#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]

use std::process::exit;
use std::time::{Duration, Instant};

mod cost_functions;
mod linear_layer;
mod matrix;
mod sigmoid_layer;

use cost_functions::*;
use linear_layer::*;
use matrix::*;
use sigmoid_layer::*;

fn main() {
    let x_train = Matrix::from_txt("./x_.txt");
    let y_train = Matrix::from_txt("./y_.txt");

    let n_train: usize = x_train.n_rows;
    let in_size: usize = x_train.n_columns;
    let batch_size: usize = n_train;
    let out_size: usize = 1;
    let learning_rate = 0.01;
    let n_epochs = 100_000;

    let mut linear_layer_0 = LinearLayer::new(in_size, out_size, batch_size);
    let mut sigmoid_layer_0 = SigmoidLayer::new(out_size, batch_size);

    let start = Instant::now();

    for i in 0..n_epochs {
        // forward pass
        linear_layer_0.forward(&x_train);
        sigmoid_layer_0.forward(&linear_layer_0.z);

        // compute precision
        let y_hat = &sigmoid_layer_0.a;
        let precision: f32 = ((y_train
            .data
            .iter()
            .zip(y_hat.data.iter())
            .map(|(&y_train_n, &y_hat_n)| (y_train_n == y_hat_n.round()) as i32)
            .sum::<i32>()) as f32)
            / (n_train as f32);


        // compute cost
        let (cost, grad) = binary_cross_entropy(&y_train, &sigmoid_layer_0.a);
        println!("epoch: {:?}, cost : {:?}, precision : {:?}", i, cost, precision);

        // backward pass
        sigmoid_layer_0.backward(&grad);
        linear_layer_0.backward(&sigmoid_layer_0.dz_);

        // update weights
        linear_layer_0.update_weights(learning_rate);
    }
}
