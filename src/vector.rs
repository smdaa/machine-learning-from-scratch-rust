use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::Sync;
use std::vec;
pub struct Vector {
    pub size: usize,
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize, value: f64) -> Self {
        Self {
            size: size,
            data: vec![value; size],
        }
    }

    pub fn randn(size: usize, mean: f64, std_dev: f64) -> Self {
        let normal = Normal::new(mean, std_dev).unwrap();
        let data = (0..size)
            .into_par_iter()
            .map(|_| normal.sample(&mut rand::thread_rng()))
            .collect();
        Self {
            size: size,
            data: data,
        }
    }

    pub fn from_str(string: &str) -> Self {
        let data: Vec<f64> = string
            .split(" ")
            .filter_map(|v| v.trim().parse::<f64>().ok())
            .collect();
        Self {
            size: data.len(),
            data,
        }
    }

    pub fn print(&self) {
        println!("{:?}", self.data)
    }

    pub fn copy(&self) -> Self {
        let new_data = self.data.clone();

        Self {
            size: self.size,
            data: new_data,
        }
    }

    pub fn is_equal(&self, vec: &Self) -> bool {
        self.size == vec.size && self.data.iter().zip(vec.data.iter()).all(|(&a, &b)| a == b)
    }
}

pub fn element_wise_operation_vector(vec: &Vector, op: impl Fn(f64) -> f64 + Sync) -> Vector {
    let data = vec.data.par_iter().map(|&x| op(x)).collect();
    Vector {
        size: vec.size,
        data: data,
    }
}

pub fn add_scalar_vector(scalar: f64, vec: &Vector) -> Vector {
    element_wise_operation_vector(vec, |x| scalar + x)
}

pub fn subtract_scalar_vector(scalar: f64, vec: &Vector) -> Vector {
    element_wise_operation_vector(vec, |x| x - scalar)
}

pub fn multiply_scalar_vector(scalar: f64, vec: &Vector) -> Vector {
    element_wise_operation_vector(vec, |x| scalar * x)
}

pub fn element_wise_operation_vectors(
    vec1: &Vector,
    vec2: &Vector,
    op: impl Fn(f64, f64) -> f64 +Sync,
) -> Vector {
    assert_eq!(vec1.size, vec2.size, "Matrix shapes must match");
    let data = vec1
        .data
        .par_iter()
        .zip(vec2.data.par_iter())
        .map(|(&a, &b)| op(a, b))
        .collect();

    Vector {
        size: vec1.size,
        data: data,
    }
}

pub fn add_vectors(vec1: &Vector, vec2: &Vector) -> Vector {
    element_wise_operation_vectors(vec1, vec2, |a, b| a + b)
}
pub fn subtract_vectors(vec1: &Vector, vec2: &Vector) -> Vector {
    element_wise_operation_vectors(vec1, vec2, |a, b| a - b)
}
pub fn multiply_vectors(vec1: &Vector, vec2: &Vector) -> Vector {
    element_wise_operation_vectors(vec1, vec2, |a, b| a * b)
}

pub fn dot_vector_vector(vec1: &Vector, vec2: &Vector) -> f64 {
    assert_eq!(vec1.size, vec2.size);
    vec1.data
        .par_iter()
        .zip(vec2.data.par_iter())
        .map(|(&x, &y)| x * y)
        .sum()
}

pub fn sum_vector(vec: &Vector) -> f64 {
    vec.data.par_iter().sum()
}

pub fn mean_vector(vec: &Vector) -> f64 {
    sum_vector(vec) / vec.size as f64
}

pub fn std_dev_vector(vec: &Vector) -> f64 {
    let mean = mean_vector(&vec);
    let n = vec.size as f64;
    let x: f64 = vec.data.par_iter().map(|&x| (x - mean).abs().powf(2.0)).sum();
    (x / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vector() {
        let vec = Vector::new(100, 1.0);
        assert_eq!(vec.size, 100);
        assert!(vec.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_randn_vector() {
        let vec = Vector::randn(100000, 0.0, 1.0);
        assert_eq!(vec.size, 100000);
        assert!(mean_vector(&vec) < 10f64.powi(-(2 as i32)));
        assert!((std_dev_vector(&vec) - 1.0).abs() < 10f64.powi(-(2 as i32)));
    }

    #[test]
    fn test_from_str_vector() {
        let vec = Vector::from_str("1.0 1.0 1.0 1.0 1.0 1.0");
        assert_eq!(vec.size, 6);
        assert!(vec.data.iter().all(|&x| x == 1.0));
        let vec = Vector::from_str("x y z");
        assert_eq!(vec.size, 0);
        let vec = Vector::from_str("x y z 2 t 3");
        assert_eq!(vec.size, 2);
        assert_eq!(vec.data[0], 2.0);
        assert_eq!(vec.data[1], 3.0);
    }

    #[test]
    fn test_copy_vector() {
        let vec = Vector::new(100, 1.0);
        let mut vec_copy = vec.copy();
        assert_eq!(vec.size, vec_copy.size);
        assert!(vec
            .data
            .iter()
            .zip(vec_copy.data.iter())
            .all(|(&x, &y)| x == y));
        vec_copy = add_scalar_vector(2.0, &vec_copy);
        assert!(vec
            .data
            .iter()
            .zip(vec_copy.data.iter())
            .all(|(&x, &y)| x == y - 2.0));
    }
    #[test]
    fn test_is_equal_vector() {
        let vec1 = Vector::from_str("1.0 1.0 1.0 1.0 1.0 1.0");
        let vec2 = Vector::from_str("1.0 1.0 1.0 1.0 2.0 1.0");
        let vec3 = Vector::from_str("1.0 1.0 1.0 1.0 1.0 1.0");
        let vec4 = Vector::from_str("1.0 1.0 1.0 1.0 1.0");
        assert_eq!(vec1.is_equal(&vec2), false);
        assert_eq!(vec1.is_equal(&vec3), true);
        assert_eq!(vec1.is_equal(&vec4), false);
    }

    #[test]
    fn test_element_wise_operation_vector() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = element_wise_operation_vector(&vec1, |x| 2.0 * x + 1.0);
        assert!(vec2.is_equal(&Vector::new(vec2.size, 5.0)));
    }

    #[test]
    fn test_add_scalar_vector() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = add_scalar_vector(3.0, &vec1);
        assert!(vec2.is_equal(&Vector::new(vec2.size, 5.0)));
    }

    #[test]
    fn test_subtract_scalar_vector() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = subtract_scalar_vector(2.0, &vec1);
        assert!(vec2.is_equal(&Vector::new(vec2.size, 0.0)));
    }

    #[test]
    fn test_multiply_scalar_vector() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = multiply_scalar_vector(2.0, &vec1);
        assert!(vec2.is_equal(&Vector::new(vec2.size, 4.0)));
    }

    #[test]
    fn test_element_wise_operation_vectors() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = Vector::new(100, 3.0);
        let vec3 = element_wise_operation_vectors(&vec1, &vec2, |x, y| x * 2.0 + y);
        assert!(vec3.is_equal(&Vector::new(vec3.size, 7.0)));
    }

    #[test]
    fn test_add_vectors() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = Vector::new(100, 3.0);
        let vec3 = add_vectors(&vec1, &vec2);
        assert!(vec3.is_equal(&Vector::new(vec3.size, 5.0)));
    }

    #[test]
    fn test_subtract_vectors() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = Vector::new(100, 3.0);
        let vec3 = subtract_vectors(&vec1, &vec2);
        assert!(vec3.is_equal(&Vector::new(vec3.size, -1.0)));
    }

    #[test]
    fn test_multiply_vectors() {
        let vec1 = Vector::new(100, 2.0);
        let vec2 = Vector::new(100, 3.0);
        let vec3 = multiply_vectors(&vec1, &vec2);
        assert!(vec3.is_equal(&Vector::new(vec3.size, 6.0)));
    }

    #[test]
    fn test_dot_vector_vector() {
        let vec1 = Vector::new(100, 1.0);
        let vec2 = Vector::new(100, 1.0);
        assert_eq!(100.0, dot_vector_vector(&vec1, &vec2));
    }

    #[test]
    fn test_sum_vector() {
        let vec = Vector::new(100, 1.0);
        assert_eq!(sum_vector(&vec), 100.0);
    }

    #[test]
    fn test_mean_vector() {
        let vec = Vector::new(100, 1.0);
        assert_eq!(mean_vector(&vec), 1.0);
    }

    #[test]
    fn test_std_dev_vector() {
        let vec = Vector::from_str("6 2 3 1");
        assert!((std_dev_vector(&vec) - 1.87).abs() < 10f64.powi(-(2 as i32)));
    }
}
