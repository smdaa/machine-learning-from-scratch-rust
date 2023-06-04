use crate::matrix::*;

pub fn sigmoid_(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, sigmoid_)
}

pub fn tanh_(x: f32) -> f32 {
    2.0 / (1.0 + (-2.0 * x).exp()) - 1.0
}

pub fn tanh(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, tanh_)
}

