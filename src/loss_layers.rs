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

pub struct CCELossLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix<f32>,
    pub grad: Matrix<f32>,
}

impl CCELossLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            loss: 0.0,
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }
}

impl LossLayer for CCELossLayer {
    fn forward(&mut self, z: &Matrix<f32>, y: &Matrix<f32>) {
        self.a.copy_content_from(z);
        self.a.subtract_column(&(self.a.max(1)));
        self.a.element_wise_operation(|x| x.exp());
        self.a.divide_column(&(self.a.sum(1)));

        self.loss = y
            .data
            .iter()
            .zip(self.a.data.iter())
            .map(|(&y_n, a_n)| if y_n > 0.0 { -a_n.ln() } else { 0.0 })
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

    fn almost_equal_scalar(scalar_1: f32, scalar_2: f32) -> bool {
        (scalar_1 - scalar_2).abs() < std::f32::EPSILON
    }
    fn almost_equal_vec(vec_1: &Vec<f32>, vec_2: &Vec<f32>) -> bool {
        vec_1
            .iter()
            .zip(vec_2.iter())
            .all(|(&x, &y)| (x - y).abs() < std::f32::EPSILON)
    }

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

        assert!(almost_equal_scalar(bce_loss_layer.loss, 0.865458));

        assert_eq!(bce_loss_layer.a.n_rows, batch_size);
        assert_eq!(bce_loss_layer.a.n_columns, 1);
        let truth: Vec<f32> = vec![8.3583869e-09, 6.2480646e-01, 9.497887e-01, 2.7607643e-06];
        assert!(almost_equal_vec(&bce_loss_layer.a.data, &truth));

        assert_eq!(bce_loss_layer.grad.n_rows, batch_size);
        assert_eq!(bce_loss_layer.grad.n_columns, 1);
        let truth: Vec<f32> = vec![2.0895967e-09, -9.3798377e-02, 2.3744719e-01, 6.9019114e-07];
        assert!(almost_equal_vec(&bce_loss_layer.grad.data, &truth));
    }

    #[test]
    fn test_cce_loss_layer() {
        let batch_size = 2;
        let in_size = 3;
        let y: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: in_size,
            data: vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let z: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: in_size,
            data: vec![-18.6, 0.51, 2.94, -12.8, 0.40, 3.95],
        };
        let mut cce_loss_layer: CCELossLayer = CCELossLayer::new(in_size, batch_size);
        cce_loss_layer.forward(&z, &y);
        cce_loss_layer.backward(&y);

        assert!(almost_equal_scalar(cce_loss_layer.loss, 1.2713474));

        assert_eq!(cce_loss_layer.a.n_rows, batch_size);
        assert_eq!(cce_loss_layer.a.n_columns, in_size);
        let truth: Vec<f32> = vec![
            4.0611861e-10,
            8.0913469e-02,
            9.1908658e-01,
            5.1673545e-08,
            2.7922573e-02,
            9.7207737e-01,
        ];
        assert!(almost_equal_vec(&cce_loss_layer.a.data, &truth));

        assert_eq!(cce_loss_layer.grad.n_rows, batch_size);
        assert_eq!(cce_loss_layer.grad.n_columns, in_size);
        let truth: Vec<f32> = vec![
            2.0305931e-10,
            -4.5954326e-01,
            4.5954329e-01,
            2.5836773e-08,
            1.3961287e-02,
            -1.3961315e-02,
        ];
        assert!(almost_equal_vec(&cce_loss_layer.grad.data, &truth));
    }
}
