use crate::matrix::*;

pub fn cross_entropy(y_hat: &Matrix, y: &Matrix) -> Matrix {
    assert_eq!(
        (y_hat.n_rows, y_hat.n_columns),
        (y.n_rows, y.n_columns),
        "Matrix shapes must match"
    );
    assert!(y.data.iter().all(|&x| x == 1.0 || x == 0.0));
    assert!(y_hat.data.iter().all(|&x| x >= 0.0 || x <= 1.0));

    let data = y
        .data
        .iter()
        .zip(y_hat.data.iter())
        .map(|(&y_n, &y_hat_n)| -(y_n * y_hat_n.ln() + (1.0 - y_n) * (1.0 - y_hat_n).ln()))
        .collect();

    Matrix {
        n_rows: y_hat.n_rows,
        n_columns: 1,
        data: data,
    }
}