use std::fmt::Display;
use std::str::FromStr;

use num_traits::float::Float;
use rand_distr::uniform::SampleUniform;

use crate::common::linear_algebra::*;
use crate::common::matrix::*;
use crate::common::vector::*;

pub fn linear_regression<T: Float + SampleUniform + FromStr + Display + Send + Sync>(
    x: &Matrix<T>,
    y: &Vector<T>,
) -> Vector<T> {
    let n_samples = x.n_rows;
    assert_eq!(y.n, n_samples);
    let x_ = x.insert_column(&Vector::new(n_samples, T::one()), 0);
    let (_, q, r) = qr_decomposition(&x_);
    let d = q.transpose().dot_vector(&y);
    back_substitution(&r, &d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        let n_samples = 100;
        let n_features = 5;
        let x: Matrix<f32> = Matrix::rand(n_samples, n_features, -10.0, 10.0);
        let beta: Vector<f32> = Vector { n: n_features, data: vec![2.0, 3.0, 4.0, 5.0, 6.0] };
        let mut y: Vector<f32> = x.dot_vector(&beta);
        let intercept: f32 = 10.0;
        y.add_scalar(intercept);
        let beta_hat: Vector<f32> = linear_regression(&x, &y);
        assert_eq!(beta_hat.n, n_features + 1);
        assert!((beta_hat.data[0] - intercept).abs() < f32::epsilon().sqrt());
        assert!(beta_hat.data.iter().skip(1).zip(beta.data.iter()).all(|(&a, &b)| (a - b).abs() < f32::epsilon().sqrt()))
    }
}