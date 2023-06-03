use crate::{matrix::*};

pub fn sigmoid_(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, sigmoid_)
}

pub fn tanh_(x: f64) -> f64 {
    2.0 / (1.0 + (-2.0 * x).exp()) - 1.0
}

pub fn tanh(x: &Matrix) -> Matrix {
    element_wise_operation_matrix(x, tanh_)
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
}
