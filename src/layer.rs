use crate::matrix::{dot_matrix_vector, Matrix};
use crate::vector::{add_vectors, element_wise_operation_vector, Vector};

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
            w: Matrix::rand(in_size, out_size, -limit, limit),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_new_dense() {
        let dense = Dense::new(32, 16);
        assert_eq!(dense.w.shape(), (32, 16));
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
        let mut dense = Dense::new(32, 32);
        dense.w = Matrix::eye(32);
        dense.b = Vector::new(32, 2.0);
        let x = Vector::new(32, 1.0);
        let new_x = forward_pass_dense(&dense, |v|  2.0*v +1.0, &x);

    }
}
