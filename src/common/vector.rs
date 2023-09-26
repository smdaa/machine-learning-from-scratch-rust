use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use std::vec;

use num_traits::float::Float;
use rand::Rng;
use rand_distr::uniform::SampleUniform;

use crate::common::matrix::*;

pub struct Vector<T> {
    pub n: usize,
    pub data: Vec<T>,
}

impl<T: Float + SampleUniform + FromStr + Display + Send + Sync> Vector<T> {
    pub fn new(n: usize, value: T) -> Self {
        Self {
            n: n,
            data: vec![value; n],
        }
    }

    pub fn zeros(n: usize) -> Self {
        Self {
            n: n,
            data: vec![T::zero(); n],
        }
    }

    pub fn rand(n: usize, low: T, high: T) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..n).map(|_| rng.gen_range(low..=high)).collect();

        Self { n: n, data: data }
    }

    pub fn from_str(string: &str) -> Self {
        let mut data = Vec::new();
        for item in string.split(" ") {
            if let Ok(num) = item.trim().parse::<T>() {
                data.push(num);
            }
        }
        Self {
            n: data.len(),
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
        let reader = BufReader::new(file);
        let mut data: Vec<T> = Vec::new();
        for line in reader.lines() {
            let line = line.expect("Could not read line");
            let num = line.trim().parse::<T>().expect("Could not parse number");
            data.push(num);
        }
        Vector {
            n: data.len(),
            data: data,
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn print(&self) {
        for i in 0..self.n {
            println!("{}", self.data[i]);
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            n: self.n,
            data: self.data.clone(),
        }
    }

    pub fn copy_content_from(&mut self, other: &Self) {
        assert_eq!(self.n, other.n, "Vector sizes should be the same");
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x = *y);
    }

    pub fn is_equal(&self, other: &Self) -> bool {
        self.n == other.n
            && self
            .data
            .iter()
            .zip(other.data.iter())
            .all(|(&a, &b)| (a - b).abs() < T::epsilon())
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(end < self.n);
        let new_n = end - start + 1;

        let data = self.data.iter().copied().skip(start).take(new_n).collect();

        Self {
            n: new_n,
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

    pub fn element_wise_operation_vector(&mut self, other: &Self, op: impl Fn(T, T) -> T) {
        assert_eq!(self.n, other.n, "Vector sizes should be the same");
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x = op(*x, *y));
    }

    pub fn add_vector(&mut self, other: &Self) {
        self.element_wise_operation_vector(other, |a, b| a + b);
    }

    pub fn subtract_vector(&mut self, other: &Self) {
        self.element_wise_operation_vector(other, |a, b| a - b);
    }

    pub fn multiply_vector(&mut self, other: &Self) {
        self.element_wise_operation_vector(other, |a, b| a * b);
    }

    pub fn divide_vector(&mut self, other: &Self) {
        self.element_wise_operation_vector(other, |a, b| a / b);
    }

    pub fn outer(&self, other: &Self) -> Matrix<T> {
        let data: Vec<T> = self
            .data
            .iter()
            .flat_map(|&a_i| other.data.iter().map(move |&b_i| a_i * b_i))
            .collect();

        Matrix {
            n_rows: self.n,
            n_columns: other.n,
            data: data,
        }
    }

    pub fn reduce(&self, mut op: impl FnMut(T, T) -> T, init: T) -> T {
        self.data.iter().fold(init, |a, &b| op(a, b))
    }

    pub fn sum(&self) -> T {
        self.reduce(|a, b| a + b, T::zero())
    }

    pub fn mean(&self) -> T {
        self.sum() / T::from(self.n).unwrap()
    }

    pub fn max(&self) -> T {
        self.reduce(|a, b| a.max(b), T::min_value())
    }

    pub fn min(&self) -> T {
        self.reduce(|a, b| a.min(b), T::max_value())
    }

    pub fn norm2(&self) -> T {
        self.data
            .iter()
            .fold(T::zero(), |a, &b| a + b.powi(2))
            .sqrt()
    }

    pub fn h_stack(&self, m: usize) -> Matrix<T> {
        let n_rows = self.n;
        let n_columns = m;

        let data = self
            .data
            .iter()
            .flat_map(|&x| std::iter::repeat(x).take(m))
            .collect();

        Matrix {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }

    pub fn v_stack(&self, m: usize) -> Matrix<T> {
        let n_rows = m;
        let n_columns = self.n;

        let data = self
            .data
            .clone()
            .into_iter()
            .cycle()
            .take(n_rows * n_columns)
            .collect();

        Matrix {
            n_rows: n_rows,
            n_columns: n_columns,
            data: data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let vec: Vector<f32> = Vector::new(100, 1.0);
        assert_eq!(vec.n, 100);
        assert!(vec.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_zeros() {
        let vec: Vector<f32> = Vector::zeros(100);
        assert_eq!(vec.n, 100);
        assert!(vec.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_rand() {
        let vec: Vector<f32> = Vector::rand(100, 2.0, 3.0);
        assert_eq!(vec.n, 100);
        assert!(vec.data.iter().all(|&x| 2.0 <= x && x <= 3.0));
    }

    #[test]
    fn test_from_str() {
        let vec: Vector<f32> = Vector::from_str("1 2 3 4 5 6 7");
        assert_eq!(vec.n, 7);
        assert_eq!(vec.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_from_txt() {
        let vec: Vector<f32> = Vector::from_txt("./test_data/test_from_txt/y.txt");
        assert_eq!(vec.n, 10);
        assert_eq!(
            vec.data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn test_clone() {
        let vec = Vector::rand(10, 2.0, 3.0);
        let vec_copy = vec.clone();
        assert_eq!(vec.n, vec_copy.n);
        assert!(vec
            .data
            .iter()
            .zip(vec_copy.data.iter())
            .all(|(x, y)| x == y));
    }

    #[test]
    fn test_copy_content_from() {
        let mut vec = Vector::zeros(10);
        vec.copy_content_from(&Vector::new(10, 1.0));
        assert_eq!(vec.n, 10);
        assert!(vec.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_is_equal() {
        let vec1 = Vector::new(3, 2.0);
        let vec2 = Vector::new(3, 2.0);
        let mut vec3 = Vector::new(3, 2.0);
        vec3.data[2] = 1.0;
        assert_eq!(vec1.is_equal(&vec2), true);
        assert_eq!(vec1.is_equal(&vec3), false);
    }

    #[test]
    fn test_slice() {
        let vec = Vector::rand(10, -1.0, 1.0);
        let vec_slice = vec.slice(2, 5);
        assert_eq!(vec_slice.n, 4);
        assert!(vec_slice
            .data
            .iter()
            .zip(vec.data.iter().skip(2).take(4))
            .all(|(&x, &y)| x == y))
    }

    #[test]
    fn test_element_wise_operation() {
        let mut vec = Vector::new(10, 2.0);
        vec.element_wise_operation(|x| x - 1.0);
        assert_eq!(vec.n, 10);
        assert!(vec.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_add_scalar() {
        let mut vec = Vector::new(10, 2.0);
        vec.add_scalar(2.0);
        assert_eq!(vec.n, 10);
        assert!(vec.data.iter().all(|&x| x == 4.0));
    }

    #[test]
    fn test_subtract_scalar() {
        let mut vec = Vector::new(10, 2.0);
        vec.subtract_scalar(2.0);
        assert_eq!(vec.n, 10);
        assert!(vec.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multiply_scalar() {
        let mut vec = Vector::new(10, 2.0);
        vec.multiply_scalar(10.0);
        assert_eq!(vec.n, 10);
        assert!(vec.data.iter().all(|&x| x == 20.0));
    }

    #[test]
    fn test_element_wise_operation_vector() {
        let mut vec1 = Vector::new(3, 2.0);
        let vec2 = Vector::new(3, 3.0);
        vec1.element_wise_operation_vector(&vec2, |x, y| (x + y) * 2.0);
        assert_eq!(vec1.n, 3);
        assert!(vec1.data.iter().all(|&x| x == 10.0));
    }

    #[test]
    fn test_add_vector() {
        let mut vec1 = Vector::new(3, 2.0);
        let vec2 = Vector::new(3, 3.0);
        vec1.add_vector(&vec2);
        assert_eq!(vec1.n, 3);
        assert!(vec1.data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_subtract_vector() {
        let mut vec1 = Vector::new(3, 2.0);
        let vec2 = Vector::new(3, 3.0);
        vec1.subtract_vector(&vec2);
        assert_eq!(vec1.n, 3);
        assert!(vec1.data.iter().all(|&x| x == -1.0));
    }

    #[test]
    fn test_multiply_vector() {
        let mut vec1 = Vector::new(3, 2.0);
        let vec2 = Vector::new(3, 3.0);
        vec1.multiply_vector(&vec2);
        assert_eq!(vec1.n, 3);
        assert!(vec1.data.iter().all(|&x| x == 6.0));
    }

    #[test]
    fn test_divide_vector() {
        let mut vec1 = Vector::new(3, 2.0);
        let vec2 = Vector::new(3, 3.0);
        vec1.divide_vector(&vec2);
        assert_eq!(vec1.n, 3);
        assert!(vec1.data.iter().all(|&x| x == 2.0 / 3.0));
    }

    #[test]
    fn test_outer() {
        let u = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let v = Vector {
            n: 2,
            data: vec![4.0, 5.0],
        };
        let mat = u.outer(&v);
        assert_eq!((mat.n_rows, mat.n_columns), (3, 2));
        assert_eq!(mat.data, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_sum() {
        let x = Vector {
            n: 6,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let sum = x.sum();
        assert_eq!(sum, 21.0);
    }

    #[test]
    fn test_mean() {
        let x = Vector {
            n: 6,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let mean = x.mean();
        assert_eq!(mean, 3.5);
    }

    #[test]
    fn test_max() {
        let x = Vector {
            n: 6,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let max = x.max();
        assert_eq!(max, 6.0);
    }

    #[test]
    fn test_min() {
        let x = Vector {
            n: 6,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let min = x.min();
        assert_eq!(min, 1.0);
    }

    #[test]
    fn test_norm2() {
        let x = Vector {
            n: 2,
            data: vec![3.0, 4.0],
        };
        assert_eq!(x.norm2(), 5.0);
    }

    #[test]
    fn test_h_stack() {
        let x = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let mat = x.h_stack(2);
        assert_eq!((mat.n_rows, mat.n_columns), (3, 2));
        assert_eq!(mat.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_v_stack() {
        let x = Vector {
            n: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        let mat = x.v_stack(2);
        assert_eq!((mat.n_rows, mat.n_columns), (2, 3));
        assert_eq!(mat.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
}
