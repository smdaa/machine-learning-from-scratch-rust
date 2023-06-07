#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod activation_function;
mod layer;
mod matrix;
mod network;
mod vector;

use activation_function::*;
use layer::*;
use matrix::*;
use network::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use vector::*;

use image::{ImageBuffer, RgbImage};
use std::{
    process::exit,
    time::{Duration, Instant},
};

pub fn batch(x: &Matrix, batch_size: usize) -> Vec<Matrix> {
    let n_rows = x.n_rows;
    let n_columns = x.n_columns;
    let n_batches = n_rows / batch_size;

    let batches = (0..n_batches)
        .map(|i| {
            let start_idx = i * batch_size;
            let end_idx = start_idx + batch_size;
            slice(&x, (start_idx, end_idx - 1), (0, n_columns - 1))
        })
        .collect();

    batches
}

pub fn unbatch(batches: &Vec<Matrix>) -> Matrix {
    let n_columns = batches[0].n_columns;
    let n_rows: usize = batches.iter().map(|batch| batch.n_rows).sum();
    let data: Vec<f32> = batches
        .iter()
        .flat_map(|batch| batch.data.clone())
        .collect();

    Matrix {
        n_rows,
        n_columns,
        data,
    }
}

fn main() {
    // Create Network
    let start = Instant::now();

    let in_size = 4;
    let hidden_size = 32;
    let out_size = 3;
    let batch_size = 128;
    let config = vec![
        (("dense", in_size, hidden_size, batch_size), "tanh"),
        (("dense", hidden_size, hidden_size, batch_size), "tanh"),
        (("dense", hidden_size, hidden_size, batch_size), "tanh"),
        (("dense", hidden_size, hidden_size, batch_size), "tanh"),
        (("dense", hidden_size, hidden_size, batch_size), "sigmoid"),
        (("dense", hidden_size, hidden_size, batch_size), "sigmoid"),
        (("dense", hidden_size, hidden_size, batch_size), "tanh"),
        (("dense", hidden_size, hidden_size, batch_size), "tanh"),
        (("dense", hidden_size, out_size, batch_size), "sigmoid"),
    ];
    let mut network = Network::new(&config);
    let duration = start.elapsed();
    println!("Time elapsed in create network is: {:?}", duration);

    // Change initial weights to truncated normal to have something pretty
    for i in 0..network.layers.len()-1 {
        network.layers[i].w = Matrix::randn_truncated(
            network.layers[i].w.n_rows,
            network.layers[i].w.n_columns,
            0.0,
            1.0,
            -3.0,
            3.0,
        );
    }

    let size_w: usize = 3840;
    let size_h: usize = 2160;
    let batch_size = 128;

    // Create dataset
    let start = Instant::now();
    let mut x = Matrix::new(size_w * size_h, in_size, 0.0);
    for i in 0..size_h {
        for j in 0..size_w {
            x.data[(i * size_w + j) * x.n_columns] = (i as f32 / size_h as f32).cos();
            x.data[(i * size_w + j) * x.n_columns + 1] = (j as f32 / size_w as f32).sin();
            x.data[(i * size_w + j) * x.n_columns + 2] = (i as f32 / size_h as f32).powf(2.0);
            x.data[(i * size_w + j) * x.n_columns + 3] = (j as f32 / size_w as f32).powf(2.0);
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed in create dataset is: {:?}", duration);

    // Create batches
    let start = Instant::now();
    let x_batches = batch(&x, batch_size);
    let duration = start.elapsed();
    println!("Time elapsed in create batches is: {:?}", duration);

    // Forward passes
    let start = Instant::now();

    let y_batches: Vec<Matrix> = x_batches
        .par_iter()
        .map(|x_batch| forward_pass_network(&network, &x_batch))
        .collect();
    let duration = start.elapsed();
    println!("Time elapsed in forward passes is: {:?}", duration);

    // Unbatch
    let start = Instant::now();
    let y = unbatch(&y_batches);

    let duration = start.elapsed();
    println!("Time elapsed in unbatch is: {:?}", duration);

    println!("{:?}", x.shape());
    println!("{:?}", y.shape());

    // Create image
    let width: u32 = size_w.try_into().unwrap();
    let height: u32 = size_h.try_into().unwrap();
    let mut image: RgbImage = ImageBuffer::new(width, height);

    for i in 0..size_h {
        for j in 0..size_w {
            let val0 = y.data[(i * size_w + j) * y.n_columns + 0];
            let val1 = y.data[(i * size_w + j) * y.n_columns + 1];
            let val2 = y.data[(i * size_w + j) * y.n_columns + 2];

            let red = (255.0*val0) as u8;
            let green =(255.0*val1) as u8;
            let blue = (255.0*val2) as u8;


            *image.get_pixel_mut(j.try_into().unwrap(), i.try_into().unwrap()) =
                image::Rgb([red, green, blue]);
        }
    }
    image.save("output.png").unwrap();
}
