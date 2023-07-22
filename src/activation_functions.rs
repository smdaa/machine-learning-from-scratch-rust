use crate::matrix::*;

pub fn sigmoid_(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn d_sigmoid_(x: f32) -> f32 {
    sigmoid_(x) * (1.0 - sigmoid_(x))
}

pub fn sigmoid(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, sigmoid_)
}

pub fn d_sigmoid(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, d_sigmoid_)
}

pub fn tanh_(x: f32) -> f32 {
    2.0 / (1.0 + (-2.0 * x).exp()) - 1.0
}

pub fn tanh(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, tanh_)
}

pub fn softmax(x: &Matrix) -> Matrix {
    let mut exp_x = element_wise_operation_matrix(&x, |v| v.exp());
    let exp_x_sum_columns: Vec<f32> = (0..x.n_rows)
        .map(|i| {
            exp_x
                .data
                .iter()
                .skip(i * x.n_columns)
                .take(x.n_columns)
                .sum()
        })
        .collect();

    for i in 0..exp_x.n_rows {
        for j in 0..exp_x.n_columns {
            exp_x.data[i * exp_x.n_columns + j] /= exp_x_sum_columns[i];
        }
    }

    exp_x
}
