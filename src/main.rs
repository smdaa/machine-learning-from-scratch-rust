#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod activation_function;
mod layer;
mod matrix;
mod vector;

use activation_function::*;
use layer::*;
use matrix::*;
use vector::*;

use image::{ImageBuffer, RgbImage};
use std::{
    process::exit,
    time::{Duration, Instant},
};

fn main() {
    let in_size = 3;
    let hidden_size = 64;
    let out_size = 3;

    let size_w: usize = 1024;
    let size_h: usize = 1024;

    let batch_size = size_h * size_w;

    // Create Layers
    let mut dense1 = Dense::new(in_size, hidden_size, batch_size);
    let mut dense2 = Dense::new(hidden_size, hidden_size, batch_size);
    let mut dense3 = Dense::new(hidden_size, hidden_size, batch_size);
    let mut dense4 = Dense::new(hidden_size, hidden_size, batch_size);
    let mut dense5 = Dense::new(hidden_size, hidden_size, batch_size);
    let mut dense6 = Dense::new(hidden_size, out_size, batch_size);

    // Change initial weights to truncated normal to have something pretty
    dense1.w = Matrix::randn_truncated(dense1.w.n_rows, dense1.w.n_columns, 0.0, 1.0, -2.0, 2.0);
    dense2.w = Matrix::randn_truncated(dense2.w.n_rows, dense2.w.n_columns, 0.0, 1.0, -2.0, 2.0);
    dense3.w = Matrix::randn_truncated(dense3.w.n_rows, dense3.w.n_columns, 0.0, 1.0, -2.0, 2.0);
    dense4.w = Matrix::randn_truncated(dense4.w.n_rows, dense4.w.n_columns, 0.0, 1.0, -2.0, 2.0);
    dense5.w = Matrix::randn_truncated(dense5.w.n_rows, dense5.w.n_columns, 0.0, 1.0, -2.0, 2.0);
    dense6.w = Matrix::randn_truncated(dense6.w.n_rows, dense6.w.n_columns, 0.0, 1.0, -2.0, 2.0);

    // Create dataset
    let start = Instant::now();
    let mut x = Matrix::new(size_w * size_h, in_size, 0.0);
    for i in 0..size_h {
        for j in 0..size_w {
            x.data[(i * size_w + j) * x.n_columns + 0] = (i as f64 / size_h as f64) - 0.5;
            x.data[(i * size_w + j) * x.n_columns + 1] = (j as f64 / size_w as f64) - 0.5;
            x.data[(i * size_w + j) * x.n_columns + 2] = (((i as f64 / size_h as f64) - 0.5)
                .powf(2.0)
                + ((j as f64 / size_w as f64) - 0.5).powf(2.0))
            .sqrt();
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed in create dataset is: {:?}", duration);

    // Forward passes
    let start = Instant::now();

    let mut y1 = forward_pass_dense(&dense1, &x);
    y1 = activation_pass(&y1, tanh);

    let mut y2 = forward_pass_dense(&dense2, &y1);
    y2 = activation_pass(&y2, tanh);

    let mut y3 = forward_pass_dense(&dense3, &y2);
    y3 = activation_pass(&y3, tanh);

    let mut y4 = forward_pass_dense(&dense4, &y3);
    y4 = activation_pass(&y4, tanh);

    let mut y5 = forward_pass_dense(&dense5, &y4);
    y5 = activation_pass(&y5, tanh);

    let mut y = forward_pass_dense(&dense6, &y5);
    y = activation_pass(&y, sigmoid);

    let duration = start.elapsed();
    println!("Time elapsed in forward loops is: {:?}", duration);

    println!("{:?}", x.shape());
    println!("{:?}", y.shape());

    // Create image
    let width: u32 = size_w.try_into().unwrap();
    let height: u32 = size_h.try_into().unwrap();
    let mut image: RgbImage = ImageBuffer::new(width, height);

    for i in 0..size_h {
        for j in 0..size_w {
            let red = (255.0 * y.data[(i * size_w + j) * y.n_columns + 0]) as u8;
            let green = (255.0 * y.data[(i * size_w + j) * y.n_columns + 1]) as u8;
            let blue = (255.0 * y.data[(i * size_w + j) * y.n_columns + 2]) as u8;
            *image.get_pixel_mut(j.try_into().unwrap(), i.try_into().unwrap()) =
                image::Rgb([red, green, blue]);
        }
    }
    image.save("output.png").unwrap();
}
