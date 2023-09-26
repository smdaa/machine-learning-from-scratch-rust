use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use std::vec;

use num_traits::float::Float;
use rand::Rng;
use rand_distr::uniform::SampleUniform;

use crate::common::vector::*;

pub struct Matrix<T> {
    pub n_rows: usize,
    pub n_columns: usize,
    pub data: Vec<T>,
}

impl<T: Float + SampleUniform + FromStr + Display + Send + Sync> Matrix<T> {
    pub fn new(n_rows: usize, n_columns: usize, value: T) -> Self {
        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: vec![value; n_rows * n_columns],
        }
    }

    pub fn zeros(n_rows: usize, n_columns: usize) -> Self {
        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: vec![T::zero(); n_rows * n_columns],
        }
    }

    pub fn eye(size: usize) -> Self {
        let data = (0..size * size)
            .map(|i| {
                if i % (size + 1) == 0 {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();

        Self {
            n_rows: size,
            n_columns: size,
            data: data,
        }
    }

    pub fn rand(n_rows: usize, n_columns: usize, low: T, high: T) -> Self {
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

    pub fn from_str(string: &str) -> Self {
        let n_rows = string.split(",").count();
        let n_columns = string.split(',').next().unwrap().split_whitespace().count();
        let data: Vec<T> = string
            .replace(",", "")
            .split(" ")
            .filter_map(|v| v.trim().parse::<T>().ok())
            .collect();

        Self {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn from_txt(path: &str) -> Self
        where
            <T as FromStr>::Err: Debug,
    {
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
            let values: Vec<T> = line
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

    pub fn clone(&self) -> Self {
        Self {
            n_rows: self.n_rows,
            n_columns: self.n_columns,
            data: self.data.clone(),
        }
    }

    pub fn copy_content_from(&mut self, other: &Self) {
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

    pub fn is_equal(&self, mat: &Self) -> bool {
        self.shape() == mat.shape()
            && self
            .data
            .iter()
            .zip(mat.data.iter())
            .all(|(&a, &b)| (a - b).abs() < T::epsilon())
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

    pub fn element_wise_operation(&mut self, op: impl Fn(T) -> T) {
        self.data.iter_mut().for_each(|x| *x = op(*x));
    }

    pub fn add_scalar(&mut self, scalar: T) {
        self.element_wise_operation(|x| scalar + x);
    }

    pub fn subtract_scalar(&mut self, scalar: T) {
        self.element_wise_operation(|x| x - scalar);
    }

    pub fn multiply_scalar(&mut self, scalar: T) {
        self.element_wise_operation(|x| x * scalar);
    }

    pub fn element_wise_operation_matrix(&mut self, other: &Self, op: impl Fn(T, T) -> T) {
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
        let mut mat: Matrix<T> = Matrix::new(n_rows, n_columns, T::zero());

        let n_inner = self.n_columns;
        mat.data
            .chunks_mut(n_columns)
            .enumerate()
            .for_each(|(i, chunk)| {
                for k in 0..n_inner {
                    let self_val = self.data[i * self.n_columns + k];
                    for j in 0..n_columns {
                        chunk[j] = chunk[j] + self_val * other.data[k * other.n_columns + j];
                    }
                }
            });

        mat
    }

    pub fn dot_vector(&self, vec: &Vector<T>) -> Vector<T> {
        assert_eq!(self.n_columns, vec.n);
        let data: Vec<T> = self
            .data
            .chunks(self.n_columns)
            .map(|row| {
                row.iter()
                    .zip(vec.data.iter())
                    .fold(T::zero(), |a, (&b, &c)| a + b * c)
            })
            .collect();
        Vector {
            n: self.n_rows,
            data: data,
        }
    }

    pub fn reduce_over_axis(&self, axis: i32, mut op: impl FnMut(T, T) -> T, init: T) -> Vector<T> {
        match axis {
            0 => Vector {
                n: self.n_columns,
                data: (0..self.n_columns)
                    .map(|col_idx| {
                        self.data
                            .iter()
                            .skip(col_idx)
                            .step_by(self.n_columns)
                            .fold(init, |a, &b| op(a, b))
                    })
                    .collect(),
            },
            1 => Vector {
                n: self.n_rows,
                data: self
                    .data
                    .chunks(self.n_columns)
                    .map(|row| row.iter().fold(init, |a, &b| op(a, b)))
                    .collect(),
            },
            _ => panic!("Invalid option"),
        }
    }

    pub fn reduce_all(&self, mut op: impl FnMut(T, T) -> T, init: T) -> T {
        self.data.iter().fold(init, |a, &b| op(a, b))
    }
    pub fn sum(&self, axis: i32) -> Vector<T> {
        self.reduce_over_axis(axis, |a, b| a + b, T::zero())
    }

    pub fn sum_all(&self) -> T {
        self.reduce_all(|a, b| a + b, T::zero())
    }

    pub fn max(&self, axis: i32) -> Vector<T> {
        self.reduce_over_axis(axis, |a, b| a.max(b), T::min_value())
    }

    pub fn max_all(&self) -> T {
        self.reduce_all(|a, b| a.max(b), T::min_value())
    }

    pub fn min(&self, axis: i32) -> Vector<T> {
        self.reduce_over_axis(axis, |a, b| a.min(b), T::max_value())
    }

    pub fn min_all(&self) -> T {
        self.reduce_all(|a, b| a.min(b), T::max_value())
    }

    pub fn element_wise_operation_row(&mut self, row: &Vector<T>, op: impl Fn(T, T) -> T) {
        assert_eq!(self.n_columns, row.n);
        self.data.chunks_mut(self.n_columns).for_each(|self_row| {
            self_row
                .iter_mut()
                .zip(row.data.iter())
                .for_each(|(x, y)| *x = op(*x, *y))
        })
    }

    pub fn add_row(&mut self, row: &Vector<T>) {
        self.element_wise_operation_row(row, |x, y| x + y);
    }

    pub fn subtract_row(&mut self, row: &Vector<T>) {
        self.element_wise_operation_row(row, |x, y| x - y);
    }

    pub fn multiply_row(&mut self, row: &Vector<T>) {
        self.element_wise_operation_row(row, |x, y| x * y);
    }

    pub fn divide_row(&mut self, row: &Vector<T>) {
        self.element_wise_operation_row(row, |x, y| x / y);
    }

    pub fn element_wise_operation_column(&mut self, column: &Vector<T>, op: impl Fn(T, T) -> T) {
        assert_eq!(self.n_rows, column.n);
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                self.data[i * self.n_columns + j] =
                    op(self.data[i * self.n_columns + j], column.data[i]);
            }
        }
    }

    pub fn add_column(&mut self, column: &Vector<T>) {
        self.element_wise_operation_column(column, |x, y| x + y);
    }

    pub fn subtract_column(&mut self, column: &Vector<T>) {
        self.element_wise_operation_column(column, |x, y| x - y);
    }

    pub fn multiply_column(&mut self, column: &Vector<T>) {
        self.element_wise_operation_column(column, |x, y| x * y);
    }

    pub fn divide_column(&mut self, column: &Vector<T>) {
        self.element_wise_operation_column(column, |x, y| x / y);
    }

    pub fn repeat(&self, n: usize, axis: i32) -> Self {
        match axis {
            0 => Self {
                n_rows: n * self.n_rows,
                n_columns: self.n_columns,
                data: self
                    .data
                    .iter()
                    .cloned()
                    .cycle()
                    .take(n * self.n_columns * self.n_rows)
                    .collect(),
            },
            1 => Self {
                n_rows: self.n_rows,
                n_columns: n * self.n_columns,
                data: self
                    .data
                    .chunks(self.n_columns)
                    .flat_map(|row| {
                        row.iter()
                            .cloned()
                            .cycle()
                            .take(n * self.n_columns)
                            .collect::<Vec<T>>()
                    })
                    .collect(),
            },
            _ => panic!("Invalid option"),
        }
    }

    pub fn insert_column(&self, column: &Vector<T>, j: usize) -> Self {
        let n_rows = self.n_rows;
        let n_columns = self.n_columns;
        assert_eq!(column.n, n_rows);
        assert!(j < n_columns + 1);
        let new_data = self
            .data
            .chunks(n_columns)
            .zip(column.data.iter())
            .flat_map(|(row, &value)| {
                let mut new_row: Vec<T> = row.into();
                new_row.insert(j, value);
                new_row
            })
            .collect::<Vec<T>>();

        Self {
            n_rows: n_rows,
            n_columns: n_columns + 1,
            data: new_data,
        }
    }

    pub fn insert_row(&self, row: &Vector<T>, i: usize) -> Self {
        let n_rows = self.n_rows;
        let n_columns = self.n_columns;
        assert_eq!(row.n, n_columns);
        assert!(i < n_rows + 1);
        //let new_data = self.data.iter().chain(row.data.iter()).cloned().collect();
        let mut new_data = Vec::with_capacity(n_columns * (n_rows + 1));
        if i == 0 {
            new_data.extend(&row.data);
            new_data.extend(&self.data);
        } else {
            new_data.extend(&self.data[0..=((i - 1) * n_columns + 1)]);
            new_data.extend(&row.data);
            new_data.extend(&self.data[(i * n_columns)..]);
        }

        Self {
            n_rows: n_rows + 1,
            n_columns: n_columns,
            data: new_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let mat: Matrix<f32> = Matrix::new(100, 100, 1.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        assert!(mat.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_zeros() {
        let mat: Matrix<f32> = Matrix::zeros(100, 100);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        assert!(mat.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_eye() {
        let mat: Matrix<f32> = Matrix::eye(100);
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
    fn test_rand() {
        let mat: Matrix<f32> = Matrix::rand(100, 200, -5.0, 5.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 200);
        assert!(mat.data.iter().all(|&x| -5.0 <= x && x <= 5.0));
    }

    #[test]
    fn test_from_str() {
        let mat: Matrix<f32> =
            Matrix::from_str("1.0 1.0 1.0, 2.0 2.0 2.0, 3.0 3.0 3.0, 4.0 4.0 4.0");
        assert_eq!(mat.n_rows, 4);
        assert_eq!(mat.n_columns, 3);
        assert_eq!(
            mat.data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn test_from_txt() {
        let x: Matrix<f32> = Matrix::from_txt("./test_data/test_from_txt/x.txt");
        let y: Matrix<f32> = Matrix::from_txt("./test_data/test_from_txt/y.txt");
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
        let mat: Matrix<f32> = Matrix::new(10, 20, 0.0);
        assert_eq!(mat.shape(), (10, 20));
    }

    #[test]
    fn test_size() {
        let mat: Matrix<f32> = Matrix::new(12, 12, 0.0);
        assert_eq!(mat.size(), 144);
    }

    #[test]
    fn test_clone() {
        let mat: Matrix<f32> = Matrix::new(12, 13, 2.0);
        let mat_copy: Matrix<f32> = mat.clone();
        assert_eq!(
            (mat.n_rows, mat.n_columns),
            (mat_copy.n_rows, mat_copy.n_columns)
        );
        assert!(mat_copy.data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_copy_content_from() {
        let mut mat = Matrix::new(12, 13, 2.0);
        mat.copy_content_from(&Matrix::new(12, 13, 1.0));
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
        let mat: Matrix<f32> = Matrix::eye(5);
        let mat_slice1: Matrix<f32> = mat.slice((0, 4), (0, 4));
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
    fn test_dot_matrix() {
        let mat = Matrix::rand(100, 200, 0.0, 1.0);
        let res = mat.dot_matrix(&Matrix::eye(200));
        assert_eq!((mat.n_rows, mat.n_columns), (res.n_rows, res.n_columns));
        assert!(mat.data.iter().zip(res.data.iter()).all(|(&x, &y)| x == y));
        let mat1 = Matrix::new(2, 3, 2.0);
        let mat2 = Matrix::new(3, 2, 3.0);
        let mat3 = mat1.dot_matrix(&mat2);
        assert_eq!((mat3.n_rows, mat3.n_columns), (2, 2));
        assert!(mat3.data.iter().all(|&x| x == 18.0));
        let mat1 = Matrix {
            n_rows: 3,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let mat2 = Matrix {
            n_rows: 3,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let mat3 = mat1.dot_matrix(&mat2);
        assert_eq!((mat3.n_rows, mat3.n_columns), (3, 3));
        assert_eq!(
            mat3.data,
            vec![30., 36., 42., 66., 81., 96., 102., 126., 150.]
        );
    }

    #[test]
    fn test_dot_vector() {
        let a = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, -1.0, 2.0, 0.0, -3.0, 1.0],
        };
        let x = Vector {
            n: 3,
            data: vec![2.0, 1.0, 0.0],
        };
        let y = a.dot_vector(&x);
        assert_eq!(y.n, 2);
        assert_eq!(y.data, vec![1.0, -3.0]);
    }

    #[test]
    fn test_sum() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let sum_1 = x.sum(1);
        assert_eq!(sum_1.n, 2);
        assert_eq!(sum_1.data, vec![6.0, 15.0]);
        let sum_0 = x.sum(0);
        assert_eq!(sum_0.n, 3);
        assert_eq!(sum_0.data, vec![5.0, 7.0, 9.0]);

        let y = Matrix {
            n_rows: 1,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let sum_1 = y.sum(1);
        assert_eq!(sum_1.n, 1);
        assert_eq!(sum_1.data, vec![6.0]);
        let sum_0 = y.sum(0);
        assert_eq!(sum_0.n, 3);
        assert_eq!(sum_0.data, vec![1.0, 2.0, 3.0]);

        let y = Matrix {
            n_rows: 3,
            n_columns: 1,
            data: vec![1.0, 2.0, 3.0],
        };
        let sum_1 = y.sum(1);
        assert_eq!(sum_1.n, 3);
        assert_eq!(sum_1.data, vec![1.0, 2.0, 3.0]);
        let sum_0 = y.sum(0);
        assert_eq!(sum_0.n, 1);
        assert_eq!(sum_0.data, vec![6.0]);
    }

    #[test]
    fn test_sum_all() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let sum_all = x.sum_all();
        assert_eq!(sum_all, 21.0);
    }

    #[test]
    fn test_max() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let max_1 = x.max(1);
        assert_eq!(max_1.n, 2);
        assert_eq!(max_1.data, vec![3.0, 6.0]);
        let max_0 = x.max(0);
        assert_eq!(max_0.n, 3);
        assert_eq!(max_0.data, vec![4.0, 5.0, 6.0]);

        let y = Matrix {
            n_rows: 1,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let max_1 = y.max(1);
        assert_eq!(max_1.n, 1);
        assert_eq!(max_1.data, vec![3.0]);
        let max_0 = y.max(0);
        assert_eq!(max_0.n, 3);
        assert_eq!(max_0.data, vec![1.0, 2.0, 3.0]);

        let y = Matrix {
            n_rows: 3,
            n_columns: 1,
            data: vec![1.0, 2.0, 3.0],
        };
        let max_1 = y.max(1);
        assert_eq!(max_1.n, 3);
        assert_eq!(max_1.data, vec![1.0, 2.0, 3.0]);
        let max_0 = y.max(0);
        assert_eq!(max_0.n, 1);
        assert_eq!(max_0.data, vec![3.0]);
    }

    #[test]
    fn test_max_all() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let max_all = x.max_all();
        assert_eq!(max_all, 6.0);
    }

    #[test]
    fn test_min() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let min_1 = x.min(1);
        assert_eq!(min_1.n, 2);
        assert_eq!(min_1.data, vec![1.0, 4.0]);
        let min_0 = x.min(0);
        assert_eq!(min_0.n, 3);
        assert_eq!(min_0.data, vec![1.0, 2.0, 3.0]);

        let y = Matrix {
            n_rows: 1,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let min_1 = y.min(1);
        assert_eq!(min_1.n, 1);
        assert_eq!(min_1.data, vec![1.0]);
        let min_0 = y.min(0);
        assert_eq!(min_0.n, 3);
        assert_eq!(min_0.data, vec![1.0, 2.0, 3.0]);

        let y = Matrix {
            n_rows: 3,
            n_columns: 1,
            data: vec![1.0, 2.0, 3.0],
        };
        let min_1 = y.min(1);
        assert_eq!(min_1.n, 3);
        assert_eq!(min_1.data, vec![1.0, 2.0, 3.0]);
        let min_0 = y.min(0);
        assert_eq!(min_0.n, 1);
        assert_eq!(min_0.data, vec![1.0]);
    }

    #[test]
    fn test_min_all() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let min_all = x.min_all();
        assert_eq!(min_all, 1.0);
    }

    #[test]
    fn test_add_row() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let row = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        x.add_row(&row);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subtract_row() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let row = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        x.subtract_row(&row);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![0.0, 0.0, 0.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_multiply_row() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let row = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        x.multiply_row(&row);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![1.0, 4.0, 9.0, 4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_divide_row() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let row = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        x.divide_row(&row);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![1.0, 1.0, 1.0, 4.0, 2.5, 2.0]);
    }

    #[test]
    fn test_add_column() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let column = Vector {
            n: 2,
            data: vec![1.0, 2.0],
        };
        x.add_column(&column);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_subtract_column() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let column = Vector {
            n: 2,
            data: vec![1.0, 2.0],
        };
        x.subtract_column(&column);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![0.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_multiply_column() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let column = Vector {
            n: 2,
            data: vec![1.0, 2.0],
        };
        x.multiply_column(&column);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![1.0, 2.0, 3.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_divide_column() {
        let mut x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let column = Vector {
            n: 2,
            data: vec![1.0, 2.0],
        };
        x.divide_column(&column);
        assert_eq!((x.n_rows, x.n_columns), (2, 3));
        assert_eq!(x.data, vec![1.0, 2.0, 3.0, 2.0, 2.5, 3.0]);
    }

    #[test]
    fn test_repeat() {
        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let y = x.repeat(3, 0);
        assert_eq!((y.n_rows, y.n_columns), (6, 3));
        assert_eq!(
            y.data,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0,
                5.0, 6.0,
            ]
        );

        let x = Matrix {
            n_rows: 1,
            n_columns: 1,
            data: vec![1.0],
        };
        let y = x.repeat(5, 0);
        assert_eq!((y.n_rows, y.n_columns), (5, 1));
        assert_eq!(y.data, vec![1.0, 1.0, 1.0, 1.0, 1.0]);

        let x = Matrix {
            n_rows: 2,
            n_columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let y = x.repeat(2, 1);
        assert_eq!((y.n_rows, y.n_columns), (2, 6));
        assert_eq!(
            y.data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );

        let x = Matrix {
            n_rows: 1,
            n_columns: 1,
            data: vec![1.0],
        };
        let y = x.repeat(5, 1);
        assert_eq!((y.n_rows, y.n_columns), (1, 5));
        assert_eq!(y.data, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_insert_column() {
        let mat = Matrix {
            n_rows: 3,
            n_columns: 2,
            data: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let column = Vector {
            n: 3,
            data: vec![1.0, 1.0, 1.0],
        };
        let new_mat = mat.insert_column(&column, 0);
        assert_eq!(new_mat.n_rows, 3);
        assert_eq!(new_mat.n_columns, 3);
        assert_eq!(
            new_mat.data,
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        let new_mat = mat.insert_column(&column, 1);
        assert_eq!(new_mat.n_rows, 3);
        assert_eq!(new_mat.n_columns, 3);
        assert_eq!(
            new_mat.data,
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );
        let new_mat = mat.insert_column(&column, 2);
        assert_eq!(new_mat.n_rows, 3);
        assert_eq!(new_mat.n_columns, 3);
        assert_eq!(
            new_mat.data,
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_insert_row() {
        let mat = Matrix {
            n_rows: 3,
            n_columns: 2,
            data: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let column = Vector {
            n: 2,
            data: vec![1.0, 1.0],
        };
        let new_mat = mat.insert_row(&column, 0);
        assert_eq!(new_mat.n_rows, 4);
        assert_eq!(new_mat.n_columns, 2);
        assert_eq!(new_mat.data, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let new_mat = mat.insert_row(&column, 3);
        assert_eq!(new_mat.n_rows, 4);
        assert_eq!(new_mat.n_columns, 2);
        assert_eq!(new_mat.data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        let new_mat = mat.insert_row(&column, 2);
        assert_eq!(new_mat.n_rows, 4);
        assert_eq!(new_mat.n_columns, 2);
        assert_eq!(new_mat.data, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }
    /*

    */
}
