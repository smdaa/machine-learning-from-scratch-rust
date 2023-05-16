mod vector;
mod matrix;
mod layer;

use vector::Vector;
use matrix::Matrix;
use layer::Dense;

fn main() {

    let dense = Dense::new(32, 16);
    let limit = 1.0 / (dense.in_size as f64).sqrt();
    println!("{}", limit);
    for i in 0..dense.w.n_rows{
        for j in 0..dense.w.n_columns{
            if dense.w.data[i][j] > limit {
                println!("{}", dense.w.data[i][j]);
            }
            if dense.w.data[i][j] < -limit {
                println!("{}", dense.w.data[i][j]);
            }
        }
    }
}
