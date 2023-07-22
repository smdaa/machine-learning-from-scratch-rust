use crate::matrix::*;

pub fn binary_cross_entropy(y: &Matrix, y_hat: &Matrix) -> (f32, Matrix) {
    let cost: f32 = y
        .data
        .iter()
        .zip(y_hat.data.iter())
        .map(|(&y_n, &y_hat_n)| -(y_n * y_hat_n.ln() + (1.0 - y_n) * (1.0 - y_hat_n).ln()))
        .sum();

    let mut grad = multiply_scalar_matrix(-1.0 / (y.n_rows as f32), &divide_matrices(y, y_hat));
    grad = add_matrices(
        &grad,
        &divide_matrices(
            &element_wise_operation_matrix(&y, |x| 1.0 - x),
            &element_wise_operation_matrix(&y_hat, |x| 1.0 - x),
        ),
    );

    (cost, grad)
}
