use crate::matrix::*;

pub struct SigmoidLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub a: Matrix,
    pub dz_: Matrix,
}

impl SigmoidLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            a: Matrix::new(batch_size, in_size, 0.0),
            dz_: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, z_: &Matrix) {
        self.a = element_wise_operation_matrix(z_, |x| 1.0 / (1.0 + (-x).exp()));
    }

    pub fn backward(&mut self, _grad: &Matrix) {
        self.dz_ = multiply_matrices(
            _grad,
            &multiply_matrices(
                &(self.a),
                &element_wise_operation_matrix(&(self.a), |x| 1.0 - x),
            ),
        );
    }
}
