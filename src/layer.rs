use crate::matrix::*;
use crate::vector::*;

pub struct Dense {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix,
    pub b: Matrix,
}

impl Dense {
    pub fn new(in_size: usize, out_size: usize, batch_size: usize) -> Self {
        let limit = 1.0 / (in_size as f64).sqrt();
        Dense {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::rand(in_size, out_size, -limit, limit),
            b: Matrix::rand(batch_size,out_size, -limit, limit),
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

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_new_dense() {
        let in_size = 32;
        let out_size = 16;
        let batch_size = 8;
        let mut dense = Dense::new(in_size, out_size, batch_size);
        assert_eq!(dense.w.shape(), (32, 16));
        assert_eq!(dense.b.shape(), (8, 16));
        let limit = 1.0 / (dense.in_size as f64).sqrt();
        assert!(dense
            .w
            .data
            .iter()
            .all(|row| row.iter().all(|x| *x > -limit && *x < limit)));
        assert!(dense
            .b
            .data
            .iter()
            .all(|row| row.iter().all(|x| *x > -limit && *x < limit)));
    }

    #[test]
    fn test_forward_pass_dense() {
        let in_size = 32;
        let out_size = 16;
        let batch_size = 8;
        let mut dense = Dense::new(in_size, out_size, batch_size);
        dense.w = Matrix::new(in_size, out_size, 1.0);
        dense.b = Matrix::new(batch_size, out_size, 3.0);
        let x = Matrix::new(batch_size, in_size, 10.0);
        let y = forward_pass_dense(&dense, &x);
        let sum_x = 10.0 * (in_size as f64);
        assert!(y.data.iter().all(|row| row.iter().all(|v| sum_x == v - 3.0)))
    }
}
