#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
mod activation_layers;
mod linear_layer;
mod loss_layers;
mod matrix;
mod vector;

use activation_layers::*;
use linear_layer::*;
use loss_layers::*;
use matrix::*;
use std::time::{Duration, Instant};
use vector::*;

fn benchmark_dot_matrix() {
    let (n, m) = (3000, 3000);
    let a: Matrix<f64> = Matrix::rand(n, m, 0.0, 100.0);
    let (m, p) = (3000, 3000);
    let b: Matrix<f64> = Matrix::rand(m, p, 0.0, 100.0);

    let start = Instant::now();
    let c: Matrix<f64> = a.dot_matrix(&b);
    let duration = start.elapsed();
    println!(
        "Time elapsed in dot_matrix {:?} x {:?} = {:?} is: {:?}",
        a.shape(),
        b.shape(),
        c.shape(),
        duration
    );
}
fn main() {
    let mut a = Matrix::new(100, 100, 1.0);
    a.svd();
}
