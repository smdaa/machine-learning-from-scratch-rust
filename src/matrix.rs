use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::vec;

pub struct Matrix {
    pub n_rows: usize,
    pub n_columns: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(n_rows: usize, n_columns: usize, value: f32) -> Self {
        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: vec![value; n_rows * n_columns],
        }
    }

    pub fn eye(size: usize) -> Self {
        let data = (0..size * size)
            .map(|i| if i % (size + 1) == 0 { 1.0 } else { 0.0 })
            .collect();

        Self {
            n_rows: size,
            n_columns: size,
            data: data,
        }
    }

    pub fn randn(n_rows: usize, n_columns: usize, low: f32, high: f32) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..n_columns * n_rows)
            .map(|_| rng.gen_range(low..=high))
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
        mean: f32,
        std_dev: f32,
        lo: f32,
        hi: f32,
    ) -> Self {
        let normal = Normal::new(mean, std_dev).unwrap();
        let data: Vec<f32> = normal
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
        let data: Vec<f32> = string
            .replace(",", "")
            .split(" ")
            .filter_map(|v| v.trim().parse::<f32>().ok())
            .collect();

        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn from_txt(path: &str) -> Self {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(err) => {
                panic!("Error opening file: {}", err);
            }
        };

        let lines: Vec<String> = BufReader::new(file)
            .lines()
            .map(|line| line.expect("Error reading line"))
            .collect();

        let n_rows = lines.len();
        let n_columns = lines[0].split_whitespace().count();

        let mut data = Vec::new();
        for line in lines {
            let values: Vec<f32> = line
                .split_whitespace()
                .map(|value| value.parse().expect("Error parsing float"))
                .collect();
            data.extend(values);
        }

        if data.len() != n_rows * n_columns {
            panic!("Inconsistent number of columns in the file.");
        }

        Self {
            n_rows,
            n_columns,
            data,
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
        let new_data: Vec<f32> = self.data.clone();

        Self {
            n_rows: self.n_rows,
            n_columns: self.n_columns,
            data: new_data,
        }
    }

    pub fn copy_from(&mut self, other: &Matrix) {
        assert_eq!(
            (self.n_rows, self.n_columns),
            (other.n_rows, other.n_columns),
            "Matrix shapes must match"
        );
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x = *y);
    }

    pub fn is_equal(&self, mat: &Matrix) -> bool {
        self.shape() == mat.shape()
            && self
                .data
                .iter()
                .zip(mat.data.iter())
                .all(|(&a, &b)| (a - b).abs() < f32::EPSILON.sqrt())
    }

    pub fn transpose(&self) -> Self {
        let data = (0..self.n_columns)
            .flat_map(|j| self.data.iter().skip(j).step_by(self.n_columns).copied())
            .collect();

        Self {
            n_rows: self.n_columns,
            n_columns: self.n_rows,
            data: data,
        }
    }

    pub fn slice(
        &self,
        (start_row, end_row): (usize, usize),
        (start_column, end_column): (usize, usize),
    ) -> Self {
        assert!(end_row < self.n_rows);
        assert!(end_column < self.n_columns);
        let new_n_rows = end_row - start_row + 1;
        let new_n_columns = end_column - start_column + 1;

        let data = self
            .data
            .chunks(self.n_columns)
            .skip(start_row)
            .take(new_n_rows)
            .flat_map(|row| row.iter().copied().skip(start_column).take(new_n_columns))
            .collect();

        Self {
            n_rows: new_n_rows,
            n_columns: new_n_columns,
            data: data,
        }
    }

    pub fn element_wise_operation(&mut self, op: impl Fn(f32) -> f32) {
        self.data.iter_mut().for_each(|x| *x = op(*x));
    }

    pub fn add_scalar(&mut self, scalar: f32) {
        self.element_wise_operation(|x| scalar + x);
    }

    pub fn subtract_scalar(&mut self, scalar: f32) {
        self.element_wise_operation(|x| x - scalar);
    }

    pub fn multiply_scalar(&mut self, scalar: f32) {
        self.element_wise_operation(|x| x * scalar);
    }

    pub fn element_wise_operation_matrix(&mut self, other: &Self, op: impl Fn(f32, f32) -> f32) {
        assert_eq!(
            (self.n_rows, self.n_columns),
            (other.n_rows, other.n_columns),
            "Matrix shapes must match"
        );
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x = op(*x, *y));
    }

    pub fn add_matrix(&mut self, other: &Self) {
        self.element_wise_operation_matrix(other, |a, b| a + b);
    }

    pub fn subtract_matrix(&mut self, other: &Self) {
        self.element_wise_operation_matrix(other, |a, b| a - b);
    }

    pub fn multiply_matrix(&mut self, other: &Self) {
        self.element_wise_operation_matrix(other, |a, b| a * b);
    }

    pub fn divide_matrix(&mut self, other: &Self) {
        self.element_wise_operation_matrix(other, |a, b| a / b);
    }

    pub fn dot_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.n_columns, other.n_rows);
        let n_rows = self.n_rows;
        let n_columns = other.n_columns;
        let mut mat = Self::new(n_rows, n_columns, 0.0);

        for i in 0..n_rows {
            for j in 0..n_columns {
                let mut acc = 0.0;
                for k in 0..self.n_columns {
                    acc += self.data[i * self.n_columns + k] * other.data[k * other.n_columns + j];
                }
                mat.data[i * mat.n_columns + j] = acc;
            }
        }

        mat
    }

    pub fn sum(&self, axis: i32) -> Self {
        match axis {
            0 => Self {
                n_rows: 1,
                n_columns: self.n_columns,
                data: (0..self.n_columns)
                    .map(|col_idx| self.data.iter().skip(col_idx).step_by(self.n_columns).sum())
                    .collect(),
            },
            1 => Self {
                n_rows: self.n_rows,
                n_columns: 1,
                data: self
                    .data
                    .chunks(self.n_columns)
                    .map(|row| row.iter().sum())
                    .collect(),
            },
            _ => panic!("Invalid option"),
        }
    }

    pub fn add_to_rows(&mut self, row: &Matrix) {
        assert_eq!(self.n_columns, row.n_columns);
        assert_eq!(1, row.n_rows);
        self.data.chunks_mut(self.n_columns).for_each(|row_| {
            row_.iter_mut()
                .zip(row.data.iter())
                .for_each(|(x, y)| *x = *x + *y)
        });
    }

    pub fn add_to_columns(&mut self, column: &Matrix) {
        assert_eq!(self.n_rows, column.n_rows);
        assert_eq!(1, column.n_columns);
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                self.data[i * self.n_columns + j] += column.data[i]
            }
        }
    }
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
    fn test_randn() {
        let mat = Matrix::randn(100, 200, -5.0, 5.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 200);
        assert!(mat.data.iter().all(|&x| -5.0 <= x && x <= 5.0));
    }

    #[test]
    fn test_randn_truncated() {
        let mat = Matrix::randn_truncated(100, 200, 0.0, 1.0, -2.0, 2.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 200);
        assert!(mat.data.iter().all(|&x| -2.0 <= x && x <= 2.0));
    }

    #[test]
    fn test_from_str() {
        let mat = Matrix::from_str("1.0 1.0 1.0, 2.0 2.0 2.0, 3.0 3.0 3.0, 4.0 4.0 4.0");
        assert_eq!(mat.n_rows, 4);
        assert_eq!(mat.n_columns, 3);
        assert_eq!(
            mat.data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn test_from_txt() {
        let x = Matrix::from_txt("./test_data/test_from_txt/x.txt");
        let y = Matrix::from_txt("./test_data/test_from_txt/y.txt");
        assert_eq!(x.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(x.n_rows, 3);
        assert_eq!(x.n_columns, 3);
        assert_eq!(
            y.data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
        assert_eq!(y.n_rows, 10);
        assert_eq!(y.n_columns, 1);
    }

    #[test]
    fn test_shape() {
        let mat = Matrix::new(10, 20, 0.0);
        assert_eq!(mat.shape(), (10, 20));
    }

    #[test]
    fn test_size() {
        let mat = Matrix::new(12, 12, 0.0);
        assert_eq!(mat.size(), 144);
    }

    #[test]
    fn test_copy() {
        let mat = Matrix::new(12, 13, 2.0);
        let mat_copy = mat.copy();
        assert_eq!(
            (mat.n_rows, mat.n_columns),
            (mat_copy.n_rows, mat_copy.n_columns)
        );
        assert!(mat_copy.data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_copy_from() {
        let mut mat = Matrix::new(12, 13, 2.0);
        mat.copy_from(&Matrix::new(12, 13, 1.0));
        assert_eq!((mat.n_rows, mat.n_columns), (12, 13));
        assert!(mat.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_is_equal() {
        let mat1 = Matrix::new(3, 4, 2.0);
        let mat2 = Matrix::new(3, 4, 2.0);
        let mut mat3 = Matrix::new(3, 4, 2.0);
        mat3.data[5] = 1.0;
        assert_eq!(mat1.is_equal(&mat2), true);
        assert_eq!(mat1.is_equal(&mat3), false);
    }

    #[test]
    fn test_transpose() {
        let mut mat = Matrix::new(4, 3, 0.0);
        let mut mat_t = Matrix::new(3, 4, 0.0);
        let mut k = 0.0;
        for i in 0..4 {
            for j in 0..3 {
                mat.data[i * 3 + j] = k;
                mat_t.data[j * 4 + i] = k;
                k = k + 1.0;
            }
        }

        let mat_t_hat = mat.transpose();
        assert_eq!(
            (mat_t_hat.n_rows, mat_t_hat.n_columns),
            (mat_t.n_rows, mat_t.n_columns)
        );
        assert!(mat_t_hat
            .data
            .iter()
            .zip(mat_t.data.iter())
            .all(|(&x, &y)| x == y));
    }

    #[test]
    fn test_slice() {
        let mat = Matrix::eye(5);
        let mat_slice1 = mat.slice((0, 4), (0, 4));
        assert_eq!(
            (mat.n_rows, mat.n_columns),
            (mat_slice1.n_rows, mat_slice1.n_columns)
        );
        assert!(mat
            .data
            .iter()
            .zip(mat_slice1.data.iter())
            .all(|(&x, &y)| x == y));

        let mat_slice2 = mat.slice((0, 2), (0, 2));
        let expected_mat_slice2 = Matrix::eye(3);
        assert_eq!(
            (mat_slice2.n_rows, mat_slice2.n_columns),
            (expected_mat_slice2.n_rows, expected_mat_slice2.n_columns)
        );
        assert!(mat_slice2
            .data
            .iter()
            .zip(expected_mat_slice2.data.iter())
            .all(|(&x, &y)| x == y));
    }

    #[test]
    fn test_dot_matrix() {
        let mat = Matrix::randn(100, 200, 0.0, 1.0);
        let res = mat.dot_matrix(&Matrix::eye(200));
        assert_eq!((mat.n_rows, mat.n_columns), (res.n_rows, res.n_columns));
        assert!(mat.data.iter().zip(res.data.iter()).all(|(&x, &y)| x == y));
        let mat1 = Matrix::new(2, 3, 2.0);
        let mat2 = Matrix::new(3, 2, 3.0);
        let mat3 = mat1.dot_matrix(&mat2);
        assert_eq!((mat3.n_rows, mat3.n_columns), (2, 2));
        assert!(mat3.data.iter().all(|&x| x == 18.0));
    }

    #[test]
    fn test_element_wise_operation() {
        let mut mat = Matrix::new(3, 3, 2.0);
        mat.element_wise_operation(|x| x - 1.0);
        assert_eq!((mat.n_rows, mat.n_columns), (3, 3));
        assert!(mat.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_add_scalar() {
        let mut mat = Matrix::new(3, 3, 2.0);
        mat.add_scalar(2.0);
        assert_eq!((mat.n_rows, mat.n_columns), (3, 3));
        assert!(mat.data.iter().all(|&x| x == 4.0));
    }

    #[test]
    fn test_subtract_scalar() {
        let mut mat = Matrix::new(3, 3, 2.0);
        mat.subtract_scalar(2.0);
        assert_eq!((mat.n_rows, mat.n_columns), (3, 3));
        assert!(mat.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multiply_scalar() {
        let mut mat = Matrix::new(3, 3, 2.0);
        mat.multiply_scalar(10.0);
        assert_eq!((mat.n_rows, mat.n_columns), (3, 3));
        assert!(mat.data.iter().all(|&x| x == 20.0));
    }

    #[test]
    fn test_element_wise_operation_matrix() {
        let mut mat1 = Matrix::new(3, 3, 2.0);
        let mat2 = Matrix::new(3, 3, 3.0);
        mat1.element_wise_operation_matrix(&mat2, |x, y| (x + y) * 2.0);
        assert_eq!((mat1.n_rows, mat1.n_columns), (3, 3));
        assert!(mat1.data.iter().all(|&x| x == 10.0));
    }

    #[test]
    fn test_add_matrix() {
        let mut mat1 = Matrix::new(3, 3, 2.0);
        let mat2 = Matrix::new(3, 3, 3.0);
        mat1.add_matrix(&mat2);
        assert_eq!((mat1.n_rows, mat1.n_columns), (3, 3));
        assert!(mat1.data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_subtract_matrix() {
        let mut mat1 = Matrix::new(3, 3, 2.0);
        let mat2 = Matrix::new(3, 3, 3.0);
        mat1.subtract_matrix(&mat2);
        assert_eq!((mat1.n_rows, mat1.n_columns), (3, 3));
        assert!(mat1.data.iter().all(|&x| x == -1.0));
    }

    #[test]
    fn test_multiply_matrix() {
        let mut mat1 = Matrix::new(3, 3, 2.0);
        let mat2 = Matrix::new(3, 3, 3.0);
        mat1.multiply_matrix(&mat2);
        assert_eq!((mat1.n_rows, mat1.n_columns), (3, 3));
        assert!(mat1.data.iter().all(|&x| x == 6.0));
    }

    #[test]
    fn test_divide_matrix() {
        let mut mat1 = Matrix::new(3, 3, 2.0);
        let mat2 = Matrix::new(3, 3, 3.0);
        mat1.divide_matrix(&mat2);
        assert_eq!((mat1.n_rows, mat1.n_columns), (3, 3));
        assert!(mat1.data.iter().all(|&x| x == 2.0 / 3.0));
    }

    #[test]
    fn test_sum() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let sum_1 = x.sum(1);
        assert_eq!((sum_1.n_rows, sum_1.n_columns), (2, 1));
        assert_eq!(sum_1.data, vec![6.0, 15.0]);

        let sum_0 = x.sum(0);
        assert_eq!((sum_0.n_rows, sum_0.n_columns), (1, 3));
        assert_eq!(sum_0.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_to_rows() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let row = Matrix {
            n_rows: 1,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        x.add_to_rows(&row);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_to_columns() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let column = Matrix {
            n_rows: 2,
            n_columns: 1,
            data: vec![2.0, 3.0],
        };
        x.add_to_columns(&column);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![3.0, 4.0, 5.0, 7.0, 8.0, 9.0]);
    }
}
