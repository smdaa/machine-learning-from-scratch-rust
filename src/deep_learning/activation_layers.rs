use crate::common::matrix::*;

pub trait ActivationLayer {
    fn new(in_size: usize, batch_size: usize) -> Self;
    fn forward(&mut self, z: &Matrix<f32>);
    fn backward(&mut self, upstream_grad: &Matrix<f32>);
}

pub struct ReluLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub z: Matrix<f32>,
    pub a: Matrix<f32>,
    pub grad: Matrix<f32>,
}

impl ActivationLayer for ReluLayer {
    fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            z: Matrix::new(batch_size, in_size, 0.0),
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }
    fn forward(&mut self, z: &Matrix<f32>) {
        self.z.copy_content_from(z);
        self.a.copy_content_from(z);
        self.a
            .element_wise_operation(|x| if x > 0.0 { x } else { 0.0 });
    }

    fn backward(&mut self, upstream_grad: &Matrix<f32>) {
        self.grad.copy_content_from(&(self.z));
        self.grad
            .element_wise_operation(|x| if x > 0.0 { 1.0 } else { 0.0 });
        self.grad.multiply_matrix(upstream_grad);
    }
}

pub struct SigmoidLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub z: Matrix<f32>,
    pub a: Matrix<f32>,
    pub grad: Matrix<f32>,
}

impl ActivationLayer for SigmoidLayer {
    fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            z: Matrix::new(batch_size, in_size, 0.0),
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }
    fn forward(&mut self, z: &Matrix<f32>) {
        self.z.copy_content_from(z);
        self.a.copy_content_from(z);
        self.a.element_wise_operation(|x| 1.0 / (1.0 + (-x).exp()));
    }

    fn backward(&mut self, upstream_grad: &Matrix<f32>) {
        self.grad.copy_content_from(&(self.z));
        self.grad
            .element_wise_operation(|x| ((-x).exp()) / (1.0 + (-x).exp()).powf(2.0));
        self.grad.multiply_matrix(upstream_grad);
    }
}

pub struct SoftmaxLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub z: Matrix<f32>,
    pub a: Matrix<f32>,
    pub grad: Matrix<f32>,
}

impl ActivationLayer for SoftmaxLayer {
    fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            z: Matrix::new(batch_size, in_size, 0.0),
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }
    fn forward(&mut self, z: &Matrix<f32>) {
        self.z.copy_content_from(z);
        self.a.copy_content_from(z);
        self.a.subtract_column(&(self.a.max(1)));
        self.a.element_wise_operation(|x| x.exp());
        self.a.divide_column(&(self.a.sum(1)));
    }

    fn backward(&mut self, upstream_grad: &Matrix<f32>) {
        self.grad.copy_content_from(&(self.z));
        self.grad
            .element_wise_operation(|x| ((-x).exp()) / (1.0 + (-x).exp()).powf(2.0));
        self.grad.multiply_matrix(upstream_grad);
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
    fn test_relu_activation_layer() {
        let batch_size = 4;
        let in_size = 2;
        let z: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: in_size,
            data: vec![-3.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -3.0],
        };
        let mut relu_layer = ReluLayer::new(in_size, batch_size);
        relu_layer.forward(&z);
        relu_layer.backward(&Matrix::new(batch_size, in_size, 1.0));

        assert_eq!(relu_layer.a.n_rows, batch_size);
        assert_eq!(relu_layer.a.n_columns, in_size);
        let truth: Vec<f32> = vec![0., 2., 0., 0., 0., 0., 2., 0.];
        assert!(almost_equal_vec(&relu_layer.a.data, &truth));

        assert_eq!(relu_layer.grad.n_rows, batch_size);
        assert_eq!(relu_layer.grad.n_columns, in_size);
        let truth: Vec<f32> = vec![0., 1., 0., 0., 0., 0., 1., 0.];
        assert!(almost_equal_vec(&relu_layer.grad.data, &truth));
    }

    #[test]
    fn test_sigmoid_activation_layer() {
        let batch_size = 2;
        let in_size = 5;
        let z: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: in_size,
            data: vec![-20., -1., 0., 1., 20., 20., 1., 0., -1., -20.],
        };
        let mut sigmoid_layer = SigmoidLayer::new(in_size, batch_size);
        sigmoid_layer.forward(&z);
        sigmoid_layer.backward(&Matrix::new(batch_size, in_size, 1.0));

        assert_eq!(sigmoid_layer.a.n_rows, batch_size);
        assert_eq!(sigmoid_layer.a.n_columns, in_size);
        let truth: Vec<f32> = vec![
            2.0611537e-09,
            2.6894143e-01,
            5.0000000e-01,
            7.3105860e-01,
            1.0000000e+00,
            1.0000000e+00,
            7.3105860e-01,
            5.0000000e-01,
            2.6894143e-01,
            2.0611537e-09,
        ];
        assert!(almost_equal_vec(&sigmoid_layer.a.data, &truth));

        assert_eq!(sigmoid_layer.grad.n_rows, batch_size);
        assert_eq!(sigmoid_layer.grad.n_columns, in_size);
        let truth: Vec<f32> = vec![
            2.0611537e-09,
            1.9661194e-01,
            2.5000000e-01,
            1.9661193e-01,
            0.0000000e+00,
            0.0000000e+00,
            1.9661193e-01,
            2.5000000e-01,
            1.9661194e-01,
            2.0611537e-09,
        ];
        assert!(almost_equal_vec(&sigmoid_layer.grad.data, &truth));
    }

    #[test]
    fn test_softmax_activation_layer() {
        let batch_size = 2;
        let in_size = 3;
        let z: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: in_size,
            data: vec![1., 2., 1., 0.5, 1., 0.5],
        };
        let mut softmax_layer = SoftmaxLayer::new(in_size, batch_size);
        softmax_layer.forward(&z);
        softmax_layer.backward(&Matrix::new(batch_size, in_size, 1.0));

        assert_eq!(softmax_layer.a.n_rows, batch_size);
        assert_eq!(softmax_layer.a.n_columns, in_size);
        let truth: Vec<f32> = vec![
            0.21194157, 0.5761169, 0.21194157, 0.27406862, 0.45186275, 0.27406862,
        ];
        assert!(almost_equal_vec(&softmax_layer.a.data, &truth));

        assert_eq!(softmax_layer.grad.n_rows, batch_size);
        assert_eq!(softmax_layer.grad.n_columns, in_size);
        let truth: Vec<f32> = vec![
            -2.5265404e-08,
            -6.8678489e-08,
            -2.5265404e-08,
            0.0000000e+00,
            0.0000000e+00,
            0.0000000e+00,
        ];
        assert!(almost_equal_vec(&softmax_layer.grad.data, &truth));
    }
}
