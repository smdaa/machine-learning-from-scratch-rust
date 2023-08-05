#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
use std::time::{Duration, Instant};
mod matrix;
use matrix::*;

fn benchmark_dot_matrix() {
    let (n, m) = (2000, 2000);
    let a:Matrix<f64> = Matrix::rand(n, m, 0.0, 100.0);
    let (m, p) = (2000, 2000);
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
