use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::Sync;
use std::vec;

pub struct Matrix {
    pub n_rows: usize,
    pub n_columns: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(n_rows: usize, n_columns: usize, value: f64) -> Self {
        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: vec![value; n_rows * n_columns],
        }
    }

    pub fn eye(size: usize) -> Self {
        let data = (0..size * size)
            .into_par_iter()
            .map(|i| if i % (size + 1) == 0 { 1.0 } else { 0.0 })
            .collect();

        Self {
            n_rows: size,
            n_columns: size,
            data: data,
        }
    }

    pub fn randn(n_rows: usize, n_columns: usize, mean: f64, std_dev: f64) -> Self {
        let normal = Normal::new(mean, std_dev).unwrap();
        let data = (0..n_columns * n_rows)
            .into_par_iter()
            .map(|_| normal.sample(&mut rand::thread_rng()))
            .collect();

        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn randn_truncated(
        n_rows: usize,
        n_columns: usize,
        mean: f64,
        std_dev: f64,
        lo: f64,
        hi: f64,
    ) -> Self {
        let normal = Normal::new(mean, std_dev).unwrap();
        let data: Vec<f64> = normal
            .sample_iter(&mut rand::thread_rng())
            .filter(|&value| lo <= value && value <= hi)
            .take(n_rows * n_columns)
            .collect();

        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn from_str(string: &str) -> Self {
        let n_rows = string.split(",").count();
        let n_columns = string.split(',').next().unwrap().split_whitespace().count();
        let data: Vec<f64> = string
            .replace(",", "")
            .split(" ")
            .filter_map(|v| v.trim().parse::<f64>().ok())
            .collect();

        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_columns)
    }

    pub fn size(&self) -> usize {
        self.n_rows * self.n_columns
    }

    pub fn print(&self) {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                print!("{} ", self.data[i * self.n_columns + j]);
            }
            println!();
        }
    }

    pub fn copy(&self) -> Self {
        let new_data: Vec<f64> = self.data.clone();

        Self {
            n_rows: self.n_rows,
            n_columns: self.n_columns,
            data: new_data,
        }
    }

    pub fn is_equal(&self, mat: &Matrix) -> bool {
        self.shape() == mat.shape() && self.data.iter().zip(mat.data.iter()).all(|(&a, &b)| a == b)
    }
}

pub fn transpose(mat: &Matrix) -> Matrix {
    let data = (0..mat.n_columns)
        .flat_map(|j| mat.data.iter().skip(j).step_by(mat.n_columns).copied())
        .collect();

    Matrix {
        n_rows: mat.n_columns,
        n_columns: mat.n_rows,
        data: data,
    }
}

pub fn slice(
    mat: &Matrix,
    (start_row, end_row): (usize, usize),
    (start_column, end_column): (usize, usize),
) -> Matrix {
    assert!(end_row < mat.n_rows);
    assert!(end_column < mat.n_columns);
    let new_n_rows = end_row - start_row + 1;
    let new_n_columns = end_column - start_column + 1;

    let data = mat
        .data
        .chunks(mat.n_columns)
        .skip(start_row)
        .take(new_n_rows)
        .flat_map(|row| row.iter().copied().skip(start_column).take(new_n_columns))
        .collect();

    Matrix {
        n_rows: new_n_rows,
        n_columns: new_n_columns,
        data: data,
    }
}

pub fn dot_matrix_matrix_naive(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    assert_eq!(mat1.n_columns, mat2.n_rows);
    let n_rows = mat1.n_rows;
    let n_columns = mat2.n_columns;
    let mut mat = Matrix::new(n_rows, n_columns, 0.0);

    for i in 0..n_rows {
        for j in 0..n_columns {
            let mut acc = 0.0;
            for k in 0..mat1.n_columns {
                acc += mat1.data[i * mat1.n_columns + k] * mat2.data[k * mat2.n_columns + j];
            }
            mat.data[i * mat.n_columns + j] = acc;
        }
    }

    mat
}

pub fn dot_matrix_matrix(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    fn dot_row_col(row: &Vec<f64>, col: &Vec<f64>) -> f64 {
        row.iter().zip(col.iter()).map(|(&a, &b)| a * b).sum()
    }
    assert_eq!(mat1.n_columns, mat2.n_rows);

    let n_rows = mat1.n_rows;
    let n_columns = mat2.n_columns;

    let data: Vec<f64> = (0..mat1.n_rows)
        .into_par_iter()
        .map(|i| {
            let row = &mat1
                .data
                .iter()
                .copied()
                .skip(mat1.n_columns * i)
                .take(mat1.n_columns)
                .collect();
            let mut result_row = Vec::with_capacity(mat2.n_columns);

            for j in 0..mat2.n_columns {
                let col = &mat2
                    .data
                    .iter()
                    .copied()
                    .skip(j)
                    .step_by(mat2.n_columns)
                    .take(mat2.n_rows)
                    .collect();
                let dot_product = dot_row_col(row, col);
                result_row.push(dot_product);
            }

            result_row
        })
        .flatten()
        .collect();

    Matrix {
        n_rows: n_rows,
        n_columns: n_columns,
        data: data,
    }
}

pub fn element_wise_operation_matrix(mat: &Matrix, op: impl Fn(f64) -> f64 + Sync) -> Matrix {
    let data = mat.data.par_iter().map(|&x| op(x)).collect();
    Matrix {
        n_rows: mat.n_rows,
        n_columns: mat.n_columns,
        data: data,
    }
}

pub fn add_scalar_matrix(scalar: f64, mat: &Matrix) -> Matrix {
    element_wise_operation_matrix(mat, |x| scalar + x)
}

pub fn subtract_scalar_matrix(scalar: f64, mat: &Matrix) -> Matrix {
    element_wise_operation_matrix(mat, |x| x - scalar)
}

pub fn multiply_scalar_matrix(scalar: f64, mat: &Matrix) -> Matrix {
    element_wise_operation_matrix(mat, |x| scalar * x)
}

pub fn element_wise_operation_matrices(
    mat1: &Matrix,
    mat2: &Matrix,
    op: impl Fn(f64, f64) -> f64 + Sync,
) -> Matrix {
    assert_eq!(
        (mat1.n_rows, mat1.n_columns),
        (mat2.n_rows, mat2.n_columns),
        "Matrix shapes must match"
    );
    let data = mat1
        .data
        .par_iter()
        .zip(mat2.data.par_iter())
        .map(|(&a, &b)| op(a, b))
        .collect();
    Matrix {
        n_rows: mat1.n_rows,
        n_columns: mat1.n_columns,
        data: data,
    }
}

pub fn add_matrices(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    element_wise_operation_matrices(mat1, mat2, |a, b| a + b)
}

pub fn subtract_matrices(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    element_wise_operation_matrices(mat1, mat2, |a, b| a - b)
}

pub fn multiply_matrices(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    element_wise_operation_matrices(mat1, mat2, |a, b| a * b)
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_new_matrix() {
        let mat = Matrix::new(100, 100, 1.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        assert!(mat.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_eye() {
        let mat = Matrix::eye(100);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        for i in 0..mat.n_rows {
            for j in 0..mat.n_columns {
                if j == i {
                    assert_eq!(mat.data[i * mat.n_columns + j], 1.0);
                } else {
                    assert_eq!(mat.data[i * mat.n_columns + j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_randn_matrix() {
        let mat = Matrix::randn(1000, 1000, 0.0, 1.0);
        assert_eq!(mat.n_rows, 1000);
        assert_eq!(mat.n_columns, 1000);
        let n = mat.n_rows * mat.n_columns;
        let mean = mat.data.iter().sum::<f64>() / n as f64;
        let mut std_dev: f64 = mat.data.iter().map(|&x| (x - mean).abs().powf(2.0)).sum();
        std_dev = (std_dev / n as f64).sqrt();
        assert!(mean < 10f64.powi(-(2 as i32)));
        assert!((std_dev - 1.0).abs() < 10f64.powi(-(2 as i32)));
    }

    #[test]
    fn test_randn_truncated_matrix() {
        let mat = Matrix::randn_truncated(1000, 1000, 0.0, 1.0, -2.0, 2.0);
        assert_eq!(mat.n_rows, 1000);
        assert_eq!(mat.n_columns, 1000);
        assert!(mat.data.iter().all(|&x| -2.0 <= x && x <= 2.0));
    }

    #[test]
    fn test_from_str_matrix() {
        let mat = Matrix::from_str("1.0 1.0 1.0, 1.0 1.0 1.0, 1.0 1.0 1.0, 1.0 1.0 1.0");
        assert_eq!(mat.n_rows, 4);
        assert_eq!(mat.n_columns, 3);
        assert!(mat.data.iter().all(|&x| x == 1.0));
        let mat2 = Matrix::from_str("1.0 1.0 1.0, 2.0 2.0 2.0, 3.0 3.0 3.0, 4.0 4.0 4.0");
        assert_eq!(mat2.n_rows, 4);
        assert_eq!(mat2.n_columns, 3);
        assert_eq!(
            mat2.data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn test_shape() {
        let mat = Matrix::randn(100, 200, 0.0, 1.0);
        assert_eq!(mat.shape(), (100, 200));
    }

    #[test]
    fn test_size() {
        let mat = Matrix::randn(100, 200, 0.0, 1.0);
        assert_eq!(mat.size(), 20000);
    }

    #[test]
    fn test_copy_matrix() {
        let mat = Matrix::randn(100, 200, 0.0, 1.0);
        let mat_copy = mat.copy();
        assert!(mat.is_equal(&mat_copy));
    }

    #[test]
    fn test_is_equal() {
        let mat1 = Matrix::from_str("1.0 1.0 1.0, 2.0 2.0 2.0, 3.0 3.0 3.0, 4.0 4.0 4.0");
        let mat2 = Matrix::from_str("1.0 1.0 1.0, 2.0 2.0 2.0, 3.0 3.0 3.0, 4.0 4.0 4.0");
        let mat3 = Matrix::from_str("1.0 1.0 1.0, 2.0 2.0 2.0, 3.0 3.0 3.0, 5.0 4.0 4.0");
        assert_eq!(mat1.is_equal(&mat2), true);
        assert_eq!(mat1.is_equal(&mat3), false);
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix::from_str("1 2 3, 4 5 6");
        let mat_transposed = transpose(&mat);
        assert!(mat_transposed.is_equal(&Matrix::from_str("1 4, 2 5, 3 6")));
    }

    #[test]
    fn test_slice() {
        let mat = Matrix::eye(5);
        let mat_slice1 = slice(&mat, (0, 4), (0, 4));
        assert!(mat.is_equal(&mat_slice1));
        let mat_slice2 = slice(&mat, (0, 2), (0, 2));
        let expected_mat_slice2 = Matrix::eye(3);
        assert!(mat_slice2.is_equal(&expected_mat_slice2));
        let mat_slice3 = slice(&mat, (0, 4), (0, 0));
        let expected_mat_slice3 = Matrix::from_str("1, 0, 0, 0, 0");
        assert!(mat_slice3.is_equal(&expected_mat_slice3));
    }

    #[test]
    fn test_dot_matrix_matrix() {
        let mat = Matrix::randn(100, 200, 0.0, 1.0);
        let res = dot_matrix_matrix(&mat, &Matrix::eye(200));
        assert!(mat.is_equal(&res));
        let mat1 = Matrix::from_str("1 0 1, 2 1 1, 0 1 1, 1 1 2");
        let mat2 = Matrix::from_str("1 2 1, 2 3 1, 4 2 2");
        let mat3 = dot_matrix_matrix(&mat1, &mat2);
        assert!(mat3.is_equal(&Matrix::from_str("5 4 3, 8 9 5, 6 5 3, 11 9 6")));
    }

    #[test]
    fn test_element_wise_operation_matrix() {
        let mat = Matrix::from_str("1 2 3, 4 5 6");
        let res = element_wise_operation_matrix(&mat, |x| x - 1.0);
        assert!(res.is_equal(&Matrix::from_str("0 1 2, 3 4 5")));
    }

    #[test]
    fn test_add_scalar_matrix() {
        let mat = Matrix::from_str("1 2 3, 4 5 6");
        let res = add_scalar_matrix(1.0, &mat);
        assert!(res.is_equal(&Matrix::from_str("2 3 4, 5 6 7")));
    }

    #[test]
    fn test_subtract_scalar_matrix() {
        let mat = Matrix::from_str("1 2 3, 4 5 6");
        let res = subtract_scalar_matrix(1.0, &mat);
        assert!(res.is_equal(&Matrix::from_str("0 1 2, 3 4 5")));
    }

    #[test]
    fn test_multiply_scalar_matrix() {
        let mat = Matrix::from_str("1 2 3, 4 5 6");
        let res = multiply_scalar_matrix(2.0, &mat);
        assert!(res.is_equal(&Matrix::from_str("2 4 6, 8 10 12")));
    }

    #[test]
    fn test_element_wise_operation_matrices() {
        let mat1 = Matrix::from_str("1 2 3, 4 5 6");
        let mat2 = Matrix::from_str("6 5 4, 3 2 1");
        let mat3 = element_wise_operation_matrices(&mat1, &mat2, |x, y| (x + y) * 2.0);
        assert!(mat3.is_equal(&Matrix::from_str("14 14 14, 14 14 14")));
    }

    #[test]
    fn test_add_matrices() {
        let mat1 = Matrix::from_str("1 2 3, 4 5 6");
        let mat2 = Matrix::from_str("6 5 4, 3 2 1");
        let mat3 = add_matrices(&mat1, &mat2);
        assert!(mat3.is_equal(&Matrix::from_str("7 7 7, 7 7 7")));
    }

    #[test]
    fn test_subtract_matrices() {
        let mat1 = Matrix::from_str("1 2 3, 4 5 6");
        let mat2 = Matrix::from_str("6 5 4, 3 2 1");
        let mat3 = subtract_matrices(&mat1, &mat2);
        assert!(mat3.is_equal(&Matrix::from_str("-5 -3 -1, 1 3 5")));
    }

    #[test]
    fn test_multiply_matrices() {
        let mat1 = Matrix::from_str("1 2 3, 4 5 6");
        let mat2 = Matrix::from_str("6 5 4, 3 2 1");
        let mat3 = multiply_matrices(&mat1, &mat2);
        assert!(mat3.is_equal(&Matrix::from_str("6 10 12, 12 10 6")));
    }
}
