use crate::{matrix::*, vector::Vector};

pub fn sigmoid_(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn d_sigmoid_(x: f64) -> f64 {
    sigmoid_(x) * (1.0 - sigmoid_(x))
}

pub fn sigmoid(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, sigmoid_)
}

pub fn d_sigmoid(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, d_sigmoid_)
}

pub fn softmax(x: &Matrix) -> Matrix {
    let exp_x = element_wise_operation_matrix(&x, |v| v.exp());
    let sum_exp = dot_matrix_vector(&exp_x, &Vector::new(exp_x.n_columns, 1.0));
    element_wise_operation_vector_matrix(&exp_x, &sum_exp, |a, b| 1.0 / b * a, false)
}

pub fn tanh_(x: f64) -> f64 {
    2.0 / (1.0 + (-2.0 * x).exp()) - 1.0
}
pub fn d_tanh_(x: f64) -> f64 {
    let tanh = tanh_(x);
    tanh.powi(2)
}

pub fn tanh(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, tanh_)
}

pub fn d_tanh(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, d_tanh_)
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_sigmoid() {
        let input1 = Matrix::new(3, 1, 1.0);
        let expected_output1 = Matrix::new(3, 1, 0.7310585786300049);
        let output1 = sigmoid(&input1);
        assert_eq!(output1.data, expected_output1.data);
        assert_eq!(output1.shape(), expected_output1.shape());

        let input2 = Matrix::new(3, 1, -1.0);
        let expected_output2 = Matrix::new(3, 1, 0.2689414213699951);
        let output2 = sigmoid(&input2);
        assert_eq!(output2.data, expected_output2.data);
        assert_eq!(output2.shape(), expected_output2.shape());

        let input3 = Matrix::from_str("0.5 1.0 2.0, -0.5 -1.0 -2.0");
        let expected_output3 =
        Matrix::from_str("0.6224593312018546 0.7310585786300049 0.8807970779778823, 0.3775406687981454 0.2689414213699951 0.11920292202211755");
        let output3 = sigmoid(&input3);
        assert_eq!(output3.data, expected_output3.data);
        assert_eq!(output3.shape(), expected_output3.shape());
    }

    #[test]
    fn test_d_sigmoid() {
        let input1 = Matrix::new(3, 1, 1.0);
        let expected_output1 = Matrix::new(3, 1, 0.19661193324148185);
        let output1 = d_sigmoid(&input1);
        assert_eq!(output1.data, expected_output1.data);
        assert_eq!(output1.shape(), expected_output1.shape());

        let input2 = Matrix::new(3, 1, -1.0);
        let expected_output2 = Matrix::new(3, 1, 0.19661193324148185);
        let output2 = d_sigmoid(&input2);
        assert_eq!(output2.data, expected_output2.data);
        assert_eq!(output2.shape(), expected_output2.shape());

        let input3 = Matrix::from_str("0.5 1.0 2.0, -0.5 -1.0 -2.0");
        let expected_output3 =
        Matrix::from_str("0.2350037122015945 0.19661193324148185 0.10499358540350662, 0.2350037122015945 0.19661193324148185 0.1049935854035065");
        let output3 = d_sigmoid(&input3);
        assert_eq!(output3.data, expected_output3.data);
        assert_eq!(output3.shape(), expected_output3.shape());
    }

    #[test]
    fn test_softmax() {
        let input1 = Matrix::new(3, 1, 1.0);
        let expected_output1 = Matrix::new(3, 1, 1.0);
        let output1 = softmax(&input1);
        assert_eq!(output1.data, expected_output1.data);
        assert_eq!(output1.shape(), expected_output1.shape());

        let input2 = Matrix::new(3, 1, -1.0);
        let expected_output2 = Matrix::new(3, 1, 1.0);
        let output2 = softmax(&input2);
        assert_eq!(output2.data, expected_output2.data);
        assert_eq!(output2.shape(), expected_output2.shape());

        let input3 = Matrix::from_str("0.5 1.0 2.0, -0.5 -1.0 -2.0");
        let expected_output3 =
        Matrix::from_str("0.14024438316608848 0.23122389762214904 0.6285317192117624, 0.5465493872661796 0.33149896042409155 0.12195165230972888");
        let output3 = softmax(&input3);
        assert_eq!(output3.data, expected_output3.data);
        assert_eq!(output3.shape(), expected_output3.shape());
    }

    #[test]
    fn test_tanh() {
        let input1 = Matrix::new(3, 1, 1.0);
        let expected_output1 = Matrix::new(3, 1, 0.7615941559557646);
        let output1 = tanh(&input1);
        assert_eq!(output1.data, expected_output1.data);
        assert_eq!(output1.shape(), expected_output1.shape());

        let input2 = Matrix::new(3, 1, -1.0);
        let expected_output2 = Matrix::new(3, 1, -0.7615941559557649);
        let output2 = tanh(&input2);
        assert_eq!(output2.data, expected_output2.data);
        assert_eq!(output2.shape(), expected_output2.shape());

        let input3 = Matrix::from_str("0.5 1.0 2.0, -0.5 -1.0 -2.0");
        let expected_output3 =
        Matrix::from_str("0.4621171572600098 0.7615941559557646 0.9640275800758169, -0.4621171572600098 -0.7615941559557649 -0.9640275800758169");
        let output3 = tanh(&input3);
        assert_eq!(output3.data, expected_output3.data);
        assert_eq!(output3.shape(), expected_output3.shape());
    }

    #[test]
    fn test_d_tanh() {
        let input1 = Matrix::new(3, 1, 1.0);
        let expected_output1 = Matrix::new(3, 1, 0.5800256583859735);
        let output1 = d_tanh(&input1);
        assert_eq!(output1.data, expected_output1.data);
        assert_eq!(output1.shape(), expected_output1.shape());

        let input2 = Matrix::new(3, 1, -1.0);
        let expected_output2 = Matrix::new(3, 1, 0.5800256583859739);
        let output2 = d_tanh(&input2);
        assert_eq!(output2.data, expected_output2.data);
        assert_eq!(output2.shape(), expected_output2.shape());

        let input3 = Matrix::from_str("0.5 1.0 2.0, -0.5 -1.0 -2.0");
        let expected_output3 =
        Matrix::from_str("0.21355226703407262 0.5800256583859735 0.9293491751468356, 0.21355226703407262 0.5800256583859739 0.9293491751468356");
        let output3 = d_tanh(&input3);
        assert_eq!(output3.data, expected_output3.data);
        assert_eq!(output3.shape(), expected_output3.shape());
    }
}
