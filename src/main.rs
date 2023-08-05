#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
mod matrix;
mod loss_layers;

use std::time::{Duration, Instant};
use matrix::*;
use loss_layers::*;

fn benchmark_dot_matrix() {
    let (n, m) = (3000, 100);
    let a:Matrix<f64> = Matrix::rand(n, m, 0.0, 100.0);
    let (m, p) = (100, 3000);
    let b:Matrix<f64> = Matrix::rand(m, p, 0.0, 100.0);

    let start = Instant::now();
    let c:Matrix<f64> = a.dot_matrix(&b);
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
    benchmark_dot_matrix();
}
