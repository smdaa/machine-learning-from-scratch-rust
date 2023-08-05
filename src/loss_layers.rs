use crate::matrix::*;

pub trait LossLayer {
    fn forward(&mut self, z: &Matrix<f32>, y: &Matrix<f32>);
    fn backward(&mut self, y: &Matrix<f32>);
}

pub struct BCELossLayer {
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix<f32>,
    pub grad: Matrix<f32>,
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
}

impl LossLayer for BCELossLayer {
    fn forward(&mut self, z: &Matrix<f32>, y: &Matrix<f32>) {
        self.a.copy_content_from(z);
        self.a.element_wise_operation(|x| 1.0 / (1.0 + (-x).exp()));
        self.loss = z
            .data
            .iter()
            .zip(y.data.iter())
            .map(|(&z_n, &y_n)| z_n.max(0.0) - z_n * y_n + (1.0 + (-z_n.abs()).exp()).ln())
            .sum::<f32>();
        self.loss = self.loss / (self.batch_size as f32);
    }
    fn backward(&mut self, y: &Matrix<f32>) {
        self.grad.copy_content_from(&(self.a));
        self.grad.subtract_matrix(y);
        self.grad.multiply_scalar(1.0 / (self.batch_size as f32));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bce_loss_layer() {
        let batch_size = 4;
        let y: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: 1,
            data: vec![0.0, 1.0, 0.0, 0.0],
        };
        let z: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: 1,
            data: vec![-18.6, 0.51, 2.94, -12.8],
        };
        let mut bce_loss_layer: BCELossLayer = BCELossLayer::new(batch_size);
        bce_loss_layer.forward(&z, &y);
        bce_loss_layer.backward(&y);
        assert!((bce_loss_layer.loss - 0.865458).abs() < std::f32::EPSILON);
        assert_eq!(bce_loss_layer.a.n_rows, batch_size);
        assert_eq!(bce_loss_layer.a.n_columns, 1);
        let truth: Vec<f32> = vec![8.3583869e-09, 6.2480646e-01, 9.497887e-01, 2.7607643e-06];
        assert!(bce_loss_layer
            .a
            .data
            .iter()
            .zip(truth.iter())
            .all(|(&x, &y)| (x - y).abs() < std::f32::EPSILON));
        assert_eq!(bce_loss_layer.grad.n_rows, batch_size);
        assert_eq!(bce_loss_layer.grad.n_columns, 1);
        let truth: Vec<f32> = vec![2.0895967e-09, -9.3798377e-02, 2.3744719e-01, 6.9019114e-07];
        assert!(bce_loss_layer
            .grad
            .data
            .iter()
            .zip(truth.iter())
            .all(|(&x, &y)| (x - y).abs() < std::f32::EPSILON));
    }
}
