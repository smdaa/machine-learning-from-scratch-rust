use rand::distributions::{Distribution, Uniform};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::vec;

use crate::vector::Vector;

pub struct Matrix {
    pub n_rows: usize,
    pub n_columns: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(n_rows: usize, n_columns: usize, value: f64) -> Self {
        Matrix {
            n_rows: n_rows,
            n_columns: n_columns,
            data: vec![vec![value; n_columns]; n_rows],
        }
    }

    pub fn eye(size: usize) -> Self {
        let mut data = vec![vec![0.0; size]; size];
        for i in 0..size {
            data[i][i] = 1.0;
        }
        Self {
            n_rows: size,
            n_columns: size,
            data: data,
        }
    }

    pub fn rand(n_rows: usize, n_columns: usize, low: f64, high: f64) -> Self {
        let mut rng = rand::thread_rng();
        let uniform_dist = Uniform::new(low, high);
        let mut data = Vec::new();
        for _ in 0..n_rows {
            data.push(
                (0..n_columns)
                    .map(|_| uniform_dist.sample(&mut rng))
                    .collect(),
            );
        }
        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn from_txt(path: &str) -> Self {
        let file = File::open(path).expect("Could not open file");
        let reader = BufReader::new(file);
        let mut data: Vec<Vec<f64>> = Vec::new();
        for line in reader.lines() {
            let line = line.expect("Could not read line");
            let entries: Vec<&str> = line.split_whitespace().collect();
            let row: Vec<f64> = entries
                .iter()
                .map(|s| s.parse::<f64>().expect("Could not parse number"))
                .collect();
            data.push(row);
        }
        Self {
            n_rows: data.len(),
            n_columns: data[0].len(),
            data: data,
        }
    }

    pub fn from_str(string: &str) -> Self {
        let mut data: Vec<Vec<f64>> = Vec::new();
        let rows_str: Vec<&str> = string.split(",").collect();
        for row_str in rows_str {
            let entries: Vec<&str> = row_str.split_whitespace().collect();
            let row: Vec<f64> = entries
                .iter()
                .map(|s| s.parse::<f64>().expect("Could not parse number"))
                .collect();
            data.push(row);
        }

        Self {
            n_rows: data.len(),
            n_columns: data[0].len(),
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
        self.data.iter().for_each(|row| println!("{:?}", row))
    }

    pub fn copy(&self) -> Self {
        let new_data: Vec<Vec<f64>> = self.data.clone();

        Self {
            n_rows: self.data.len(),
            n_columns: self.data[0].len(),
            data: new_data,
        }
    }

    pub fn is_equal(&self, mat: &Matrix) -> bool {
        self.shape() == mat.shape() && self.data.iter().zip(mat.data.iter()).all(|(x, y)| *x == *y)
    }
}

pub fn transpose(mat: &Matrix) -> Matrix {
    let mut new_mat: Matrix = Matrix::new(mat.n_columns, mat.n_rows, 0.0);
    for i in 0..mat.n_rows {
        for j in 0..mat.n_columns {
            new_mat.data[j][i] = mat.data[i][j]
        }
    }
    new_mat
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
    let mut new_mat = Matrix::new(new_n_rows, new_n_columns, 0.0);

    for i in 0..new_mat.n_rows {
        for j in 0..new_mat.n_columns {
            new_mat.data[i][j] = mat.data[i + start_row][j + start_column];
        }
    }

    new_mat
}

pub fn concat(mat1: &Matrix, mat2: &Matrix, row: bool) -> Matrix {
    if row {
        assert_eq!(mat1.n_columns, mat2.n_columns);
        let n_rows = mat1.n_rows + mat2.n_rows;
        let n_columns = mat1.n_columns;
        let mut mat = mat1.copy();
        mat.n_rows = n_rows;
        mat.n_columns = n_columns;
        mat2.data.iter().for_each(|row| mat.data.push(row.clone()));
        mat
    } else {
        assert_eq!(mat1.n_rows, mat2.n_rows);
        let n_rows = mat1.n_rows;
        let n_columns = mat1.n_columns + mat2.n_columns;
        let mut mat = mat1.copy();
        mat.n_rows = n_rows;
        mat.n_columns = n_columns;
        mat2.data
            .iter()
            .zip(mat.data.iter_mut())
            .for_each(|(row2, row1)| row2.iter().for_each(|v| row1.push(*v)));

        mat
    }
}

pub fn det(mat: &Matrix) -> f64 {
    assert_eq!(mat.n_rows, mat.n_columns, "Matrix must be square");
    let mut mat_copy = mat.copy();
    let mut acc = 1.0;

    for j in 0..mat_copy.n_columns {
        let mut pivot = mat_copy.data[j][j];
        let mut pivot_row = j;

        for i in j + 1..mat_copy.n_rows {
            if mat_copy.data[i][j].abs() > pivot.abs() {
                pivot = mat_copy.data[i][j];
                pivot_row = i;
            }
        }
        if pivot == 0.0 {
            return 0.0;
        }
        if pivot_row != j {
            mat_copy.data.swap(j, pivot_row);
            acc *= -1.0;
        }

        acc *= pivot;

        for i in j + 1..mat_copy.n_rows {
            for c in j + 1..mat_copy.n_columns {
                mat_copy.data[i][c] -= mat_copy.data[i][j] * mat_copy.data[j][c] / pivot;
            }
        }
    }

    acc
}

pub fn dot_matrix_matrix(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    assert_eq!(mat1.n_columns, mat2.n_rows);
    let mut mat = Matrix::new(mat1.n_rows, mat2.n_columns, 0.0);

    for i in 0..mat.n_rows {
        for j in 0..mat.n_columns {
            let mut acc = 0.0;
            for k in 0..mat1.n_columns {
                acc += mat1.data[i][k] * mat2.data[k][j];
            }
            mat.data[i][j] = acc;
        }
    }

    mat
}

pub fn element_wise_operation_matrix(mat: &Matrix, op: impl Fn(f64) -> f64) -> Matrix {
    let mut new_mat = mat.copy();
    new_mat
        .data
        .iter_mut()
        .for_each(|row| row.iter_mut().for_each(|x| *x = op(*x)));
    new_mat
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
    op: impl Fn(f64, f64) -> f64,
) -> Matrix {
    assert_eq!(
        (mat1.n_rows, mat1.n_columns),
        (mat2.n_rows, mat2.n_columns),
        "Matrix shapes must match"
    );
    let mut mat = mat1.copy();

    mat.data
        .iter_mut()
        .zip(mat2.data.iter())
        .for_each(|(row1, row2)| row1.iter_mut().zip(row2).for_each(|(a, b)| *a = op(*a, *b)));

    mat
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

pub fn dot_matrix_vector(mat: &Matrix, vec: &Vector) -> Vector {
    assert_eq!(mat.n_columns, vec.size);
    let mut new_vec = Vector::new(mat.n_rows, 0.0);
    new_vec
        .data
        .iter_mut()
        .zip(mat.data.iter())
        .for_each(|(x, row)| *x = row.iter().zip(vec.data.iter()).map(|(x, y)| x * y).sum());

    new_vec
}

pub fn element_wise_operation_vector_matrix(
    mat: &Matrix,
    vec: &Vector,
    op: impl Fn(f64, f64) -> f64,
    row: bool,
) -> Matrix {
    if row {
        assert_eq!(vec.size, mat.n_columns);
    } else {
        assert_eq!(vec.size, mat.n_rows);
    }
    let mut new_mat = mat.copy();
    for i in 0..new_mat.n_rows {
        for j in 0..new_mat.n_columns {
            if row {
                new_mat.data[i][j] = op(new_mat.data[i][j], vec.data[j]);
            } else {
                new_mat.data[i][j] = op(new_mat.data[i][j], vec.data[i]);
            }
        }
    }
    new_mat
}

pub fn add_vector_matrix(mat: &Matrix, vec: &Vector, row: bool) -> Matrix {
    element_wise_operation_vector_matrix(mat, vec, |a, b| a + b, row)
}

pub fn subtract_vector_matrix(mat: &Matrix, vec: &Vector, row: bool) -> Matrix {
    element_wise_operation_vector_matrix(mat, vec, |a, b| a - b, row)
}

pub fn multiply_vector_matrix(mat: &Matrix, vec: &Vector, row: bool) -> Matrix {
    element_wise_operation_vector_matrix(mat, vec, |a, b| a * b, row)
}

pub fn dot_vector_t_vector(vec1: &Vector, vec2: &Vector) -> Matrix {
    let n_rows = vec1.size;
    let n_columns = vec2.size;
    let mut mat = Matrix::new(n_rows, n_columns, 0.0);

    for i in 0..mat.n_rows {
        for j in 0..mat.n_columns {
            mat.data[i][j] = vec1.data[i] * vec2.data[j];
        }
    }

    mat
}

#[cfg(test)]
mod tests {
    use crate::vector::dot_vector_vector;

    use super::*;
    #[test]
    fn test_new_matrix() {
        let mat = Matrix::new(100, 100, 1.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        assert!(mat.data.iter().all(|row| row.iter().all(|x| *x == 1.0)));
    }

    #[test]
    fn test_eye() {
        let mat = Matrix::eye(100);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        for i in 0..mat.n_rows {
            for j in 0..mat.n_columns {
                if j == i {
                    assert_eq!(mat.data[i][j], 1.0);
                } else {
                    assert_eq!(mat.data[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_rand_matrix() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 200);
        assert!(mat
            .data
            .iter()
            .all(|row| row.iter().all(|x| *x > 0.0 && *x < 1.0)));
    }

    #[test]
    fn test_from_str_matrix() {
        let mat = Matrix::from_str("1.0 1.0 1.0, 1.0 1.0 1.0, 1.0 1.0 1.0, 1.0 1.0 1.0");
        assert_eq!(mat.n_rows, 4);
        assert_eq!(mat.n_columns, 3);
        assert!(mat.data.iter().all(|row| row.iter().all(|x| *x == 1.0)));
    }

    #[test]
    fn test_shape() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        assert_eq!(mat.shape(), (100, 200));
    }

    #[test]
    fn test_size() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        assert_eq!(mat.size(), 20000);
    }

    #[test]
    fn test_copy_matrix() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        let mat_copy = mat.copy();
        assert!(mat.is_equal(&mat_copy));
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        let mat_transposed = transpose(&mat);
        assert_eq!(mat.n_rows, mat_transposed.n_columns);
        assert_eq!(mat.n_columns, mat_transposed.n_rows);
        for i in 0..mat.n_rows {
            for j in 0..mat.n_columns {
                assert_eq!(mat.data[i][j], mat_transposed.data[j][i]);
            }
        }
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
    fn test_concat() {
        let mat1 = Matrix::new(2, 3, 0.0);
        let mat2 = Matrix::new(3, 3, 2.0);
        let mat = concat(&mat1, &mat2, true);
        let expected_mat = Matrix::from_str("0 0 0, 0 0 0, 2 2 2, 2 2 2, 2 2 2");
        assert!(mat.is_equal(&expected_mat));
        let mat1 = Matrix::new(2, 2, 0.0);
        let mat2 = Matrix::new(2, 3, 2.0);
        let mat = concat(&mat1, &mat2, false);
        let expected_mat = Matrix::from_str("0 0 2 2 2, 0 0 2 2 2");
        assert!(mat.is_equal(&expected_mat));
    }

    #[test]
    fn test_det_0() {
        let mat = Matrix::eye(100);
        assert_eq!(det(&mat), 1.0);
    }

    #[test]
    fn test_det_1() {
        let mat = Matrix::from_str("6 1 1, 4 -2 5, 2 8 7");
        assert_eq!(det(&mat), -306.0);
    }

    #[test]
    fn test_det_2() {
        let mut mat = Matrix::eye(100);
        mat.data[50][50] = 0.0;
        assert_eq!(det(&mat), 0.0);
    }

    #[test]
    fn test_dot_matrix_matrix_0() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        let res = dot_matrix_matrix(&mat, &Matrix::eye(200));
        assert_eq!(res.shape(), (100, 200));
        assert!(mat.is_equal(&res));
    }

    #[test]
    fn test_dot_matrix_matrix_1() {
        let mat1 = Matrix::from_str("1 0 1, 2 1 1, 0 1 1, 1 1 2");
        let mat2 = Matrix::from_str("1 2 1, 2 3 1, 4 2 2");
        let mat3 = dot_matrix_matrix(&mat1, &mat2);
        assert!(mat3.is_equal(&Matrix::from_str("5 4 3, 8 9 5, 6 5 3, 11 9 6")));
    }

    #[test]
    fn test_dot_matrix_vector() {
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1 0");
        assert!(dot_matrix_vector(&mat, &vec).is_equal(&Vector::from_str("1 -3")));
    }

    #[test]
    fn test_element_wise_operation_matrix() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = element_wise_operation_matrix(&mat1, |x| 2.0 * x + 1.0);
        assert_eq!(mat1.shape(), mat2.shape());
        assert!(mat1
            .data
            .iter()
            .zip(mat2.data.iter())
            .all(|(row1, row2)| row1.iter().zip(row2).all(|(x, y)| *y == *x * 2.0 + 1.0)));
    }

    #[test]
    fn test_add_scalar_matrix() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = add_scalar_matrix(2.0, &mat1);
        assert_eq!(mat1.shape(), mat2.shape());
        assert!(mat1
            .data
            .iter()
            .zip(mat2.data.iter())
            .all(|(row1, row2)| row1.iter().zip(row2).all(|(x, y)| *y == *x + 2.0)));
    }

    #[test]
    fn test_subtract_scalar_matrix() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = subtract_scalar_matrix(2.0, &mat1);
        assert_eq!(mat1.shape(), mat2.shape());
        assert!(mat1
            .data
            .iter()
            .zip(mat2.data.iter())
            .all(|(row1, row2)| row1.iter().zip(row2).all(|(x, y)| *y == *x - 2.0)));
    }

    #[test]
    fn test_multiply_scalar_matrix() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = multiply_scalar_matrix(2.0, &mat1);
        assert_eq!(mat1.shape(), mat2.shape());
        assert!(mat1
            .data
            .iter()
            .zip(mat2.data.iter())
            .all(|(row1, row2)| row1.iter().zip(row2).all(|(x, y)| *y == *x * 2.0)));
    }

    #[test]
    fn test_element_wise_operation_matrices() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat3 = element_wise_operation_matrices(&mat1, &mat2, |x, y| x * 2.0 + y);
        assert!(mat3
            .data
            .iter()
            .zip(mat1.data.iter().zip(mat2.data.iter()))
            .all(|(row3, (row1, row2))| row3
                .iter()
                .zip(row1.iter().zip(row2.iter()))
                .all(|(z, (x, y))| *z == *y + *x * 2.0)));
    }

    #[test]
    fn test_add_matrices() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat3 = add_matrices(&mat1, &mat2);
        assert!(mat3
            .data
            .iter()
            .zip(mat1.data.iter().zip(mat2.data.iter()))
            .all(|(row3, (row1, row2))| row3
                .iter()
                .zip(row1.iter().zip(row2.iter()))
                .all(|(z, (x, y))| *z == *y + *x)));
    }

    #[test]
    fn test_subtract_matrices() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat3 = subtract_matrices(&mat1, &mat2);
        assert!(mat3
            .data
            .iter()
            .zip(mat1.data.iter().zip(mat2.data.iter()))
            .all(|(row3, (row1, row2))| row3
                .iter()
                .zip(row1.iter().zip(row2.iter()))
                .all(|(z, (x, y))| *z == *x - *y)));
    }

    #[test]
    fn test_multiply_matrices() {
        let mat1 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat2 = Matrix::rand(100, 200, 0.0, 1.0);
        let mat3 = multiply_matrices(&mat1, &mat2);
        assert!(mat3
            .data
            .iter()
            .zip(mat1.data.iter().zip(mat2.data.iter()))
            .all(|(row3, (row1, row2))| row3
                .iter()
                .zip(row1.iter().zip(row2.iter()))
                .all(|(z, (x, y))| *z == *x * *y)));
    }

    #[test]
    fn test_add_vector_matrix() {
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1 0");
        assert!(add_vector_matrix(&mat, &vec, true).is_equal(&Matrix::from_str("3 0 2, 2 -2 1")));
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1");
        assert!(add_vector_matrix(&mat, &vec, false).is_equal(&Matrix::from_str("3 1 4, 1 -2 2")));
    }

    #[test]
    fn test_subtract_vector_matrix() {
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1 0");
        assert!(subtract_vector_matrix(&mat, &vec, true)
            .is_equal(&Matrix::from_str("-1 -2 2, -2 -4 1")));
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1");
        assert!(subtract_vector_matrix(&mat, &vec, false)
            .is_equal(&Matrix::from_str("-1 -3 0, -1 -4 0")));
    }

    #[test]
    fn test_multiply_vector_matrix() {
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1 0");
        assert!(
            multiply_vector_matrix(&mat, &vec, true).is_equal(&Matrix::from_str("2 -1 0, 0 -3 0"))
        );
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1");
        assert!(
            multiply_vector_matrix(&mat, &vec, false).is_equal(&Matrix::from_str("2 -2 4, 0 -3 1"))
        )
    }

    #[test]
    fn test_dot_vector_t_vector() {
        let vec1 = Vector::new(32, 2.0);
        let vec2 = Vector::new(16, 2.0);
        let mat = dot_vector_t_vector(&vec1, &vec2);
        assert!(mat.data.iter().all(|row| row.iter().all(|v| *v == 4.0)));
    }
}
