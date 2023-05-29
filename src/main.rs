mod activation_function;
mod layer;
mod matrix;
mod vector;

use activation_function::*;
use layer::*;
use matrix::*;
use vector::*;

use std::time::{Duration, Instant};

pub fn create_batches(x: &Matrix, batch_size: usize) -> Vec<Matrix> {
    let n_rows = x.n_rows;
    let n_columns = x.n_columns;
    let n_batches = n_rows / batch_size;
    let mut batches: Vec<Matrix> = Vec::new();

    for batch_idx in 0..n_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = start_idx + batch_size;
        let batch_data = slice(&x, (start_idx, end_idx-1), (0, n_columns - 1));
        batches.push(batch_data);
    }

    if n_rows % batch_size != 0 {
        let start_idx = n_batches * batch_size;
        let batch_data = slice(&x, (start_idx, n_rows - 1), (0, n_columns - 1));
        batches.push(batch_data);
    }
    batches
}

pub fn unbatch(batchs: &Vec<Matrix>) -> Matrix {
    let mut acc = batchs[0].copy();
    for i in 1..batchs.len(){
        acc = concat(&acc, &batchs[i], true);
    }
    acc
}

fn main() {
    let in_size = 3;
    let hidden_size = 64;
    let out_size = 1;
    let batch_size = 32;

    let dense1 = Dense::new(in_size, hidden_size, batch_size);
    let dense2 = Dense::new(hidden_size, hidden_size, batch_size);
    let dense3 = Dense::new(hidden_size, hidden_size, batch_size);
    let dense4 = Dense::new(hidden_size, out_size, batch_size);

    let size_w = 512;
    let size_h = 512;

    // Create dataset
    let mut x = Matrix::new(size_w * size_h, in_size, 0.0);
    for i in 0..size_h {
        for j in 0..size_w {
            x.data[i * size_w + j] = vec![i as f64, j as f64, (i + j) as f64];
        }
    }

    let x_batched = create_batches(&x, batch_size);
    let mut y_batched : Vec<Matrix> = Vec::new();

    let start = Instant::now();
    for x_batch in x_batched{
        let mut y1 = forward_pass_dense(&dense1, &x_batch);
        y1 = activation_pass(&y1, tanh);

        let mut y2 = forward_pass_dense(&dense2, &y1);
        y2 = activation_pass(&y2, tanh);
        
        let mut y3 = forward_pass_dense(&dense3, &y2);
        y3 = activation_pass(&y3, tanh);
        
        let mut y4 = forward_pass_dense(&dense4, &y3);
        y4 = activation_pass(&y4, sigmoid);

        y_batched.push(y4);
    }

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

}
