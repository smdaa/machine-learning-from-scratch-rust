use crate::matrix::*;

pub struct Dense {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix,
    pub b: Matrix,
    pub z: Matrix,
    pub a: Matrix,
    pub db: Matrix,
    pub dw: Matrix,
    pub activation_function: fn(&Matrix) -> Matrix,
    pub d_activation_function: fn(&Matrix) -> Matrix,
}

impl Dense {
    pub fn new(
        in_size: usize,
        out_size: usize,
        batch_size: usize,
        activation_function: fn(&Matrix) -> Matrix,
        d_activation_function: fn(&Matrix) -> Matrix,
    ) -> Self {
        let std_dev = (2.0 / (in_size + out_size) as f32).sqrt();
        Self {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::randn(in_size, out_size, 0.0, std_dev),
            b: Matrix::new(batch_size, out_size, 0.0),
            z: Matrix::new(batch_size, out_size, 0.0),
            a: Matrix::new(batch_size, out_size, 0.0),
            dw: Matrix::new(in_size, out_size, 0.0),
            db: Matrix::new(batch_size, out_size, 0.0),
            activation_function: activation_function,
            d_activation_function: d_activation_function,
        }
    }

    pub fn forward_pass(&mut self, x_batch: &Matrix) -> Matrix {
        self.z = add_matrices(&dot_matrix_matrix(x_batch, &self.w), &self.b);
        self.a = (self.activation_function)(&self.z);

        self.a.copy()
    }

    pub fn backward_pass(&mut self, right_grad_output: &Matrix, left_activation_output: &Matrix) {
        self.db = multiply_matrices(&(self.d_activation_function)(&self.z), right_grad_output);
        self.dw = dot_matrix_matrix(&left_activation_output, &transpose(&self.db));
    }
}
