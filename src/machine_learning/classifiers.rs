use std::fmt::Display;
use std::str::FromStr;

use num_traits::float::Float;
use rand_distr::uniform::SampleUniform;

use crate::common::matrix::*;
use crate::common::vector::*;

pub fn logistic_regression<T: Float + SampleUniform + FromStr + Display + Send + Sync>(
    x: &Matrix<T>,
    y: &Vector<T>,
    epochs: usize,
    lr: T,
) -> (Vector<T>, Vector<T>, T) {
    let n_samples = x.n_rows;
    let n_features = x.n_columns;
    assert_eq!(y.n, n_samples);
    let mut w: Vector<T> = Vector::zeros(n_features);
    let mut b = T::zero();
    for i in 0..epochs {
        let mut a = x.dot_vector(&w);
        a.add_scalar(b);
        a.element_wise_operation(|x| T::one() / (T::one() + (-x).exp()));
        a.subtract_vector(&y);
        let db = a.mean();
        let mut dw = x.transpose().dot_vector(&a);
        dw.multiply_scalar(T::one() / T::from(n_features).unwrap());
        b = b - lr * db;
        dw.multiply_scalar(lr);
        w.subtract_vector(&dw);
    }
    let mut y_pred = x.dot_vector(&w);
    y_pred.add_scalar(b);
    y_pred.element_wise_operation(|x| T::one() / (T::one() + (-x).exp()));
    y_pred.data.iter_mut().for_each(|x| {
        if (*x > T::from(0.5).unwrap()) {
            *x = T::one();
        } else {
            *x = T::zero();
        }
    }
    );
    (y_pred, w, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression() {
        let n_samples = 100;
        let n_features = 5;
        let w: Vector<f32> = Vector { n: n_features, data: vec![2.0, 3.0, 4.0, 5.0, 6.0] };
        let b: f32 = 10.0;
        let x: Matrix<f32> = Matrix::rand(n_samples, n_features, 0.0, 1.0);
        let mut y: Vector<f32> = x.dot_vector(&w);
        y.add_scalar(b);
        y.element_wise_operation(|x| 1.0 / (1.0 + (-x).exp()));
        y.data.iter_mut().for_each(|x| {
            if (*x > 0.5) {
                *x = 1.0;
            } else {
                *x = 0.0;
            }
        });
        let epochs = 2;
        let lr = 0.1;
        let (y_pred, w_pred, b_pred) = logistic_regression(&x, &y, epochs, lr);
        assert_eq!(y_pred.n, n_samples);
        assert_eq!(w_pred.n, n_features);
        let acc: i32 = y.data.iter().zip(y_pred.data.iter()).map(|(a, b)| (a == b) as i32).sum();
        assert_eq!(acc, n_samples.try_into().unwrap());
    }
}