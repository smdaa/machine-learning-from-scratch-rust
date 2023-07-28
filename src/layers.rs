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
            x: Matrix::new(batch_size, in_size, 0.0),
            dw: Matrix::new(in_size, out_size, 0.0),
            db: Matrix::new(batch_size, out_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, x: &Matrix) {
        self.x.copy_from(x);
        self.z.copy_from(&(x.dot_matrix(&self.w)));
        self.z.add_matrix(&self.b);
    }

    pub fn backward(&mut self, upstream_grad: &Matrix) {
        self.dw
            .copy_from(&((self.x.transpose()).dot_matrix(upstream_grad)));

        self.db.copy_from(upstream_grad);
        let upstream_grad_sum_rows = upstream_grad.copy();
        upstream_grad_sum_rows.sum(1);
        self.db.add_to_rows(&upstream_grad_sum_rows);

        self.grad
            .copy_from(&(&upstream_grad.dot_matrix(&(self.w.transpose()))));
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.dw.multiply_scalar(-learning_rate);
        self.w.add_matrix(&(self.dw));

        self.db.multiply_scalar(-learning_rate);
        self.b.add_matrix(&(self.db));
    }
}

pub struct BCELossLayer {
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix,
    pub grad: Matrix,
}

impl BCELossLayer {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size,
            loss: 0.0,
            a: Matrix::new(batch_size, 1, 0.0),
            grad: Matrix::new(batch_size, 1, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix, y: &Matrix) {
        self.a.copy_from(z);
        self.a.element_wise_operation(|x| 1.0 / (1.0 + (-x).exp()));
        self.loss = z
            .data
            .iter()
            .zip(y.data.iter())
            .map(|(z_n, y_n)| z_n.max(0.0) - z_n * y_n + (1.0 + (-z_n.abs()).exp()).ln())
            .sum::<f32>()
            / (self.batch_size as f32);
    }

    pub fn backward(&mut self, y: &Matrix) {
        self.grad.copy_from(&(self.a));
        self.grad.subtract_matrix(y);
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
        self.a.copy_from(z);
        self.a
            .element_wise_operation(|x| if x > 0.0 { x } else { 0.0 });
    }

    pub fn backward(&mut self, upstream_grad: &Matrix) {
        self.grad.copy_from(&(self.a));
        self.grad
            .element_wise_operation(|x| if x > 0.0 { 1.0 } else { 0.0 });
        self.grad.multiply_matrix(upstream_grad);
    }
}

pub struct CELossLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix,
    pub grad: Matrix,
}

impl CELossLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            loss: 0.0,
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix, y: &Matrix) {}

    pub fn backward(&mut self, y: &Matrix) {}
}
