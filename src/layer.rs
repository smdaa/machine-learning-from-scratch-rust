use crate::matrix::*;

pub struct LinearLayer {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix,
    pub b: Matrix,
    pub z: Matrix,
    pub x: Matrix,
    pub dw: Matrix,
    pub db: Matrix,
    pub grad: Matrix,
}

impl LinearLayer {
    pub fn new(in_size: usize, out_size: usize, batch_size: usize) -> Self {
        let std_dev = 1.0 / (in_size as f32).sqrt();
        Self {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::randn(in_size, out_size, -std_dev, std_dev),
            b: Matrix::randn(batch_size, out_size, -std_dev, std_dev),
            z: Matrix::new(batch_size, out_size, 0.0),
            x: Matrix::new(batch_size, out_size, 0.0),
            dw: Matrix::new(in_size, out_size, 0.0),
            db: Matrix::new(batch_size, out_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, x: &Matrix) {
        self.x = x.copy();
        self.z = add_matrices(&dot_matrix_matrix(x, &self.w), &self.b);
    }

    pub fn backward(&mut self, upstream_grad: &Matrix) {
        self.dw = dot_matrix_matrix(&transpose(&self.x), upstream_grad);
        self.db = dot_matrix_matrix(upstream_grad, &Matrix::new(upstream_grad.n_columns, self.out_size, 1.0));
        self.grad = dot_matrix_matrix(upstream_grad, &transpose(&self.w));
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.w = add_matrices(
            &(self.w),
            &multiply_scalar_matrix(-learning_rate, &(self.dw)),
        );
        self.b = add_matrices(
            &(self.b),
            &multiply_scalar_matrix(-learning_rate, &(self.db)),
        );

    }
}

pub struct SigmoidLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub a: Matrix,
    pub grad: Matrix,
}

impl SigmoidLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix) {
        self.a = element_wise_operation_matrix(z, |x| 1.0 / (1.0 + (-x).exp()));
    }

    pub fn backward(&mut self) {
        self.grad = multiply_matrices(
            &(self.a),
            &element_wise_operation_matrix(&(self.a), |x| 1.0 - x),
        )
    }
}

pub struct ReluLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub a: Matrix,
    pub grad: Matrix,
}

impl ReluLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix) {
        self.a = element_wise_operation_matrix(z, |x| if x > 0.0 { x } else { 0.0 });
    }

    pub fn backward(&mut self) {
        self.grad = element_wise_operation_matrix(&(self.a), |x| if x > 0.0 { 1.0 } else { 0.0 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_forward() {
        let in_size = 3;
        let out_size = 4;
        let batch_size = 5;
        let mut layer = LinearLayer::new(in_size, out_size, batch_size);
        layer.w = transpose(&Matrix::from_str("1 2 3, 4 5 6, 7 8 9, 10 11 12"));
        layer.b = Matrix::from_str("1 2 3 4, 1 2 3 4, 1 2 3 4, 1 2 3 4, 1 2 3 4");
        let x = Matrix::from_str("10 11 12, 13 14 15, 16 17 18, 19 20 21, 22 23 24");
        layer.forward(&x);
        assert!((layer.z).is_equal(&Matrix::from_str(
            "69 169 269 369, 87 214 341 468, 105 259 413 567, 123 304 485 666, 141 349 557 765"
        )))
    }

    #[test]
    fn test_sigmoid_layer_forward() {
        let in_size = 4;
        let batch_size = 5;
        let mut layer = SigmoidLayer::new(in_size, batch_size);
        let x = multiply_scalar_matrix(
            1.0 / 250.0,
            &Matrix::from_str("10 11 12, 13 14 15, 16 17 18, 19 20 21, 22 23 24"),
        );
        layer.forward(&x);
        assert!((layer.a).is_equal(&Matrix::from_str("0.5100 0.5110 0.5120, 0.5130 0.5140 0.5150, 0.5160 0.5170 0.5180, 0.5190 0.5200 0.5210, 0.5220 0.5230 0.5240")));
    }
}
