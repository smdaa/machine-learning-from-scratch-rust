use crate::matrix::*;
use crate::vector::*;

pub struct Dense {
    pub in_size: usize,
    pub out_size: usize,
    pub w: Matrix,
    pub b: Vector,
}

impl Dense {
    pub fn new(
        in_size: usize,
        out_size: usize,
    ) -> Self {
        let limit = 1.0 / (in_size as f64).sqrt();
        Dense {
            in_size: in_size,
            out_size: out_size,
            w: Matrix::rand(out_size, in_size, -limit, limit),
            b: Vector::rand(out_size, -limit, limit),
        }
    }
}

pub fn forward_pass_dense(dense: &Dense, activation: impl Fn(f64) -> f64, x: &Vector) -> Vector {
    element_wise_operation_vector(
        &add_vectors(&dot_matrix_vector(&dense.w, x), &dense.b),
        activation,
    )
}

pub fn backward_pass_dense(dense: &Dense, activation: impl Fn(f64) -> f64, x: &Vector, grad_accum: &Vector) {

}

#[cfg(test)]
mod tests {
    use crate::{matrix, vector::sum_vector};

    use super::*;
    #[test]
    fn test_new_dense() {
        let dense = Dense::new(32, 16);
        assert_eq!(dense.w.shape(), (16, 32));
        assert_eq!(dense.b.size, 16);
        let limit = 1.0 / (dense.in_size as f64).sqrt();
        assert!(dense.w
            .data
            .iter()
            .all(|row| row.iter().all(|x| *x > -limit && *x < limit)));
        assert!(dense.b.data
            .iter()
            .all(|x| *x > -limit && *x < limit));
    }

    #[test]
    fn test_forward_pass_dense() {
        let in_size = 32;
        let out_size = 16;
        let mut dense = Dense::new(in_size, out_size);
        dense.w = Matrix::new(out_size, in_size, 1.0);
        dense.b = Vector::new(out_size, 3.0);
        let x = Vector::new(in_size, 10.0);
        let y = forward_pass_dense(&dense, |v| 2.0*v+1.0, &x);
        let sum_x = 10.0 * (in_size as f64);
        assert!(y.data.iter().all(|v| sum_x == (v-1.0)/2.0 -3.0));
    }

}
