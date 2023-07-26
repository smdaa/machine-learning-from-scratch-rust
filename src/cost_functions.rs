use crate::matrix::*;
use std::cmp::max;
use std::cmp::min;

pub fn binary_cross_entropy(y: &Matrix, y_hat: &Matrix) -> (f32, Matrix) {
    let lower_bound: f32 = 1e-07;
    let upper_bound: f32 = 1.0 - lower_bound;
    let loss: f32 = y
        .data
        .iter()
        .zip(y_hat.data.iter())
        .map(|(&y_n, &y_hat_n)| {
            let y_hat_n_clipped = lower_bound.max(upper_bound.min(y_hat_n));
            if y_n == 1.0 {
                -y_hat_n_clipped.ln()
            } else {
                -(1.0 - y_hat_n_clipped).ln()
            }
        })
        .sum::<f32>()
        / (y.n_rows as f32);

    let mut grad = y_hat.copy();
    grad.element_wise_operation_matrix(&y, |y_hat_n, y_n| {
        (y_hat_n - y_n) / (y_hat_n * (1.0 - y_hat_n))
    });

    (loss, grad)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_cross_entropy() {
        let y = Matrix::from_str("0, 0");
        let y_hat = Matrix::from_str("0.6, 0.2");
        let (loss, grad) = binary_cross_entropy(&y, &y_hat);
        assert!((loss - 0.569).abs() <= 1e-3);
        assert!(grad.is_equal(&Matrix::from_str("2.5, 1.25")))
    }
}
