use crate::matrix::*;

pub fn binary_cross_entropy(y: &Matrix, y_hat: &Matrix) -> (f32, Matrix) {
    let loss: f32 = y
        .data
        .iter()
        .zip(y_hat.data.iter())
        .map(|(&y_n, &y_hat_n)| -(y_n * y_hat_n.ln() + (1.0 - y_n) * (1.0 - y_hat_n).ln()))
        .sum::<f32>()
        / (y.n_rows as f32);

    let grad = divide_matrices(
        &subtract_matrices(&y_hat, &y),
        &multiply_matrices(&y_hat, &element_wise_operation_matrix(y_hat, |x| 1.0 - x)),
    );


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
