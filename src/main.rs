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
use std::time::{Duration, Instant};

pub fn create_batches(x: &Matrix, batch_size: usize) -> Vec<Matrix> {
    let n_rows = x.n_rows;
    let n_columns = x.n_columns;
    let n_batches = n_rows / batch_size;
    let mut batches: Vec<Matrix> = Vec::new();

    for batch_idx in 0..n_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = start_idx + batch_size;
        let batch_data = slice(&x, (start_idx, end_idx - 1), (0, n_columns - 1));
        batches.push(batch_data);
    }

    if n_rows % batch_size != 0 {
        let start_idx = n_batches * batch_size;
        let batch_data = slice(&x, (start_idx, n_rows - 1), (0, n_columns - 1));
        batches.push(batch_data);
    }
    batches
}

pub fn un_batch(batches: &Vec<Matrix>) -> Matrix {
    let n_columns = batches[0].n_columns;
    let n_rows = batches.iter().map(|mat| mat.n_rows).sum();

    let mut data = Vec::new();
    for mat in batches {
        for row in &mat.data {
            data.push(row.to_vec());
        }
    }

    Matrix {
        n_rows: n_rows,
        n_columns: n_columns,
        data: data,
    }
}


fn main() {
    let in_size = 3;
    let hidden_size = 64;
    let out_size = 3;
    let batch_size = 256;

    let size_w: usize = 1920;
    let size_h: usize = 1080;

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
            x.data[i * size_w + j] = vec![
                (i as f64 / size_h as f64) - 0.5,
                (j as f64 / size_w as f64) - 0.5,
                (((i as f64 / size_h as f64) - 0.5).powf(2.0) + ((j as f64 / size_w as f64)- 0.5).powf(2.0))
                    .sqrt(),
            ];
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed in create dataset is: {:?}", duration);

    // Create batches
    let start = Instant::now();
    let x_batched = create_batches(&x, batch_size);
    let duration = start.elapsed();
    println!("Time elapsed in create batches is: {:?}", duration);

    // Forward loops
    let start = Instant::now();
    let mut y_batched: Vec<Matrix> = Vec::new();
    for x_batch in x_batched {
        let mut y1 = forward_pass_dense(&dense1, &x_batch);
        y1 = activation_pass(&y1, tanh);

        let mut y2 = forward_pass_dense(&dense2, &y1);
        y2 = activation_pass(&y2, tanh);

        let mut y3 = forward_pass_dense(&dense3, &y2);
        y3 = activation_pass(&y3, tanh);

        let mut y4 = forward_pass_dense(&dense4, &y3);
        y4 = activation_pass(&y4, tanh);

        let mut y5 = forward_pass_dense(&dense5, &y4);
        y5 = activation_pass(&y5, tanh);

        let mut y6 = forward_pass_dense(&dense6, &y5);
        y6 = activation_pass(&y6, sigmoid);

        y_batched.push(y6);
    }
    let duration = start.elapsed();
    println!("Time elapsed in forward loops is: {:?}", duration);

    // Unbatch
    let start = Instant::now();
    let y = un_batch(&y_batched);
    let duration = start.elapsed();
    println!("Time elapsed in Unbatch is: {:?}", duration);

    println!("{:?}", x.shape());
    println!("{:?}", y.shape());

    // Create image
    let width: u32 = size_w.try_into().unwrap();
    let height: u32 = size_h.try_into().unwrap();
    let mut image: RgbImage = ImageBuffer::new(width, height);

    for i in 0..size_h {
        for j in 0..size_w {
            let red = (255.0 * y.data[i * size_w + j][0]) as u8;
            let green = (255.0 * y.data[i * size_w + j][1]) as u8;
            let blue = (255.0 * y.data[i * size_w + j][2]) as u8;          
            *image.get_pixel_mut(j.try_into().unwrap(), i.try_into().unwrap()) =
                image::Rgb([red, green, blue]);                
        }
    }
    image.save("output.png").unwrap();

}
