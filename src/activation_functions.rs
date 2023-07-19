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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mat = Matrix::from_str("0.0, 1.0, -2.0, 3.0, 3.0 -2.0 1.0 0.0");
        let res = sigmoid(&mat);
        assert!(res.is_equal(&Matrix::from_str(
            "0.5, 0.7310586, 0.11920292, 0.95257413, 0.95257413 0.11920292 0.7310586 0.5"
        )));
    }

    #[test]
    fn test_tanh() {
        let mat = Matrix::from_str("0.0 1.0 -2.0 3.0, 3.0 -2.0 1.0 0.0");
        let res = tanh(&mat);
        assert!(res.is_equal(&Matrix::from_str(
            "0 0.76159406 -0.9640276 0.99505484, 0.99505484 -0.9640276 0.76159406 0"
        )))
    }

    #[test]
    fn test_softmax() {
        let mat = Matrix::from_str("1 3 2.5 5 4 2, 2 4 5 2.5 3 1");
        let res = softmax(&mat);
        assert!(res.is_equal(&Matrix::from_str(
            "0.011077545 0.0818526 0.049646113 0.60481346 0.22249843 0.030111888, 0.030111888 0.22249843 0.60481346 0.049646113 0.0818526 0.011077545"
        )))
    }
}
