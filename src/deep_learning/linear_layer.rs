use crate::common::matrix::*;
use crate::common::vector::*;

pub struct LinearLayer {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix<f32>,
    pub b: Vector<f32>,
    pub z: Matrix<f32>,
    pub x: Matrix<f32>,
    pub dw: Matrix<f32>,
    pub db: Vector<f32>,
    pub grad: Matrix<f32>,
}

impl LinearLayer {
    pub fn new(in_size: usize, out_size: usize, batch_size: usize) -> Self {
        let std_dev = (6.0_f32).sqrt() / ((in_size + out_size) as f32).sqrt();
        Self {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::rand(in_size, out_size, -std_dev, std_dev),
            b: Vector::rand(out_size, -std_dev, std_dev),
            z: Matrix::new(batch_size, out_size, 0.0),
            x: Matrix::new(batch_size, in_size, 0.0),
            dw: Matrix::new(in_size, out_size, 0.0),
            db: Vector::new(out_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, x: &Matrix<f32>) {
        self.x.copy_content_from(x);
        self.z.copy_content_from(&(x.dot_matrix(&self.w)));
        self.z.add_row(&self.b);
    }

    pub fn backward(&mut self, upstream_grad: &Matrix<f32>) {
        self.dw
            .copy_content_from(&((self.x.transpose()).dot_matrix(upstream_grad)));
        self.db.copy_content_from(&(upstream_grad.sum(0)));
        self.grad
            .copy_content_from(&(&upstream_grad.dot_matrix(&(self.w.transpose()))));
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.dw.multiply_scalar(-learning_rate);
        self.w.add_matrix(&(self.dw));
        self.db.multiply_scalar(-learning_rate);
        self.b.add_vector(&(self.db));
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
    fn test_linear_layer() {
        let batch_size = 2;
        let in_size = 3;
        let out_size = 4;
        let x: Matrix<f32> = Matrix {
            n_rows: batch_size,
            n_columns: in_size,
            data: vec![1., 2., 1., 0.5, 1., 0.5],
        };
        let mut linear_layer = LinearLayer::new(in_size, out_size, batch_size);
        linear_layer.w = Matrix {
            n_rows: in_size,
            n_columns: out_size,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0],
        };
        linear_layer.b = Vector {
            n: out_size,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        linear_layer.forward(&x);
        linear_layer.backward(&Matrix::new(batch_size, out_size, 1.0));

        assert_eq!(linear_layer.z.n_rows, batch_size);
        assert_eq!(linear_layer.z.n_columns, out_size);
        let truth: Vec<f32> = vec![20., 25., 30., 35., 10.5, 13.5, 16.5, 19.5];
        assert!(almost_equal_vec(&linear_layer.z.data, &truth));

        assert_eq!(linear_layer.grad.n_rows, batch_size);
        assert_eq!(linear_layer.grad.n_columns, in_size);
        let truth: Vec<f32> = vec![10., 26., 38., 10., 26., 38.];
        assert!(almost_equal_vec(&linear_layer.grad.data, &truth));

        assert_eq!(linear_layer.dw.n_rows, in_size);
        assert_eq!(linear_layer.dw.n_columns, out_size);
        let truth: Vec<f32> = vec![1.5, 1.5, 1.5, 1.5, 3., 3., 3., 3., 1.5, 1.5, 1.5, 1.5];
        assert!(almost_equal_vec(&linear_layer.dw.data, &truth));

        assert_eq!(linear_layer.db.n, out_size);
        let truth: Vec<f32> = vec![2., 2., 2., 2.];
        assert!(almost_equal_vec(&linear_layer.db.data, &truth));
    }
}
