mod vector;
mod matrix;
mod layer;

use vector::*;
use matrix::*;
use layer::*;

fn main() {
    let in_size = 32;
    let out_size = 16;
    let dense = Dense::new(in_size, out_size);
    let x = Vector::rand(in_size, 0.0, 1.0);
    let y = forward_pass_dense(&dense, |v| 2.0*v+1.0, &x);

    x.print();
    y.print();
}
