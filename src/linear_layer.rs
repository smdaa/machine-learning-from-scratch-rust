use crate::matrix::*;

pub struct LinearLayer {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix,
    pub b: Matrix,
    pub z: Matrix,
    pub a_: Matrix,
    pub dw: Matrix,
    pub db: Matrix,
    pub da_: Matrix,
}

impl LinearLayer {
    pub fn new(in_size: usize, out_size: usize, batch_size: usize) -> Self {
        let std_dev = (2.0 / (in_size + out_size) as f32).sqrt();
        Self {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::randn(in_size, out_size, 0.0, std_dev),
            b: Matrix::new(batch_size, out_size, 0.0),
            z: Matrix::new(batch_size, out_size, 0.0),
            a_: Matrix::new(batch_size, out_size, 0.0),
            dw: Matrix::new(in_size, out_size, 0.0),
            db: Matrix::new(batch_size, out_size, 0.0),
            da_: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, a_: &Matrix) {
        self.a_ = a_.copy();
        self.z = add_matrices(&dot_matrix_matrix(a_, &self.w), &self.b);
    }

    pub fn backward(&mut self, _grad: &Matrix) {
        self.dw = dot_matrix_matrix(&transpose(&self.a_), _grad);
        self.db = dot_matrix_matrix(_grad, &Matrix::new(_grad.n_columns, self.out_size, 1.0));
        self.da_ = dot_matrix_matrix(_grad, &transpose(&self.w));
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.w = subtract_matrices(&(self.w), &multiply_scalar_matrix(learning_rate, &(self.dw)));
        self.b = subtract_matrices(&(self.b), &multiply_scalar_matrix(learning_rate, &(self.db)));
    }
}
