use crate::matrix::*;

pub struct Dense {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix,
    pub b: Matrix,
}
impl Dense {
    pub fn new(in_size: usize, out_size: usize, batch_size: usize) -> Self {
        let std_dev = (2.0 / (in_size + out_size) as f32).sqrt();
        Self {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::randn(in_size, out_size, 0.0, std_dev),
            b: Matrix::new(batch_size, out_size, 0.0),
        }
    }
}

pub fn forward_pass_dense(dense: &Dense, x: &Matrix) -> Matrix {
    add_matrices(&dot_matrix_matrix(x, &dense.w), &dense.b)
}

pub fn activation_pass(y: &Matrix, activation: impl Fn(&Matrix) -> Matrix) -> Matrix {
    activation(y)
}

/*
pub fn backward_pass_dense(dense: &Dense, x:&Matrix, grad_output: &Matrix) -> (Vector, Matrix, Vector){
    
}
*/
