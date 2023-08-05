use crate::matrix::*;

pub trait ActivationLayer {
    fn forward(&mut self, z: &Matrix<f32>);
    fn backward(&mut self, upstream_grad: &Matrix<f32>);
}

pub struct ReluLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub a: Matrix<f32>,
    pub grad: Matrix<f32>,
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
}

impl ActivationLayer for ReluLayer {
    fn forward(&mut self, z: &Matrix<f32>) {
        self.a.copy_content_from(z);
        self.a
            .element_wise_operation(|x| if x > 0.0 { x } else { 0.0 });
    }

    fn backward(&mut self, upstream_grad: &Matrix<f32>) {
        self.grad.copy_content_from(&(self.a));
        self.grad
            .element_wise_operation(|x| if x > 0.0 { 1.0 } else { 0.0 });
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
}
