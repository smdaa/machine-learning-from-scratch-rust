use crate::matrix::*;

pub fn cross_entropy(y_hat: &Matrix, y: &Matrix) -> f32 {
    assert_eq!(
        (y_hat.n_rows, y_hat.n_columns),
        (y.n_rows, y.n_columns),
        "Matrix shapes must match"
    );
    assert!(y.data.iter().all(|&x| x == 1.0 || x == 0.0));
    assert!(y_hat.data.iter().all(|&x| x >= 0.0 || x <= 1.0));

    y.data
        .iter()
        .zip(y_hat.data.iter())
        .map(|(&y_n, &y_hat_n)| -(y_n * y_hat_n.ln() + (1.0 - y_n) * (1.0 - y_hat_n).ln()))
        .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy() {
        let y_hat = Matrix::from_str("0.07, 0.91, 0.74, 0.23, 0.85, 0.17, 0.94");
        let y = Matrix::from_str("0, 1, 1, 0, 0, 1, 1");
        let res = cross_entropy(&y_hat, &y);
        assert_eq!(res, 4.4603033);

        let y_hat = Matrix::from_str("0.6, 0.51, 0.94, 0.8");
        let y = Matrix::from_str("0, 1, 0, 0");
        let res = cross_entropy(&y_hat, &y);
        assert_eq!(res, 6.012484);

    }
}
