use rand_distr::{Normal, Distribution};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::vec;

pub struct Vector {
    pub size: usize,
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize, value: f64) -> Self {
        Vector {
            size: size,
            data: vec![value; size],
        }
    }

    pub fn randn(size: usize, mean:f64, std_dev:f64) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();
        let data = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Self {
            size: size,
            data: data,
        }
    }

    pub fn from_txt(path: &str) -> Self {
        let file = File::open(path).expect("Could not open file");
        let reader = BufReader::new(file);
        let mut data: Vec<f64> = Vec::new();
        for line in reader.lines() {
            let line = line.expect("Could not read line");
            let num = line.trim().parse::<f64>().expect("Could not parse number");
            data.push(num);
        }
        Vector {
            size: data.len(),
            data,
        }
    }

    pub fn from_str(string: &str) -> Self {
        let mut data = Vec::new();
        for item in string.split(" ") {
            if let Ok(num) = item.trim().parse::<f64>() {
                data.push(num);
            }
        }
        Vector {
            size: data.len(),
            data,
        }
    }

    pub fn print(&self) {
        println!("{:?}", self.data)
    }

    pub fn copy(&self) -> Self {
        let new_data = self.data.clone();

        Vector {
            size: self.size,
            data: new_data,
        }
    }

    pub fn is_equal(&self, vec: &Vector) -> bool {
        self.size == vec.size && self.data.iter().zip(vec.data.iter()).all(|(x, y)| *x == *y)
    }
}

pub fn element_wise_operation_vector(vec: &Vector, op: impl Fn(f64) -> f64) -> Vector {
    let mut new_vec = vec.copy();
    new_vec.data.iter_mut().for_each(|x| *x = op(*x));

    new_vec
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
    op: impl Fn(f64, f64) -> f64,
) -> Vector {
    assert_eq!(vec1.size, vec2.size, "Matrix shapes must match");
    let mut vec = vec1.copy();
    vec.data
        .iter_mut()
        .zip(vec2.data.iter())
        .for_each(|(a, b)| *a = op(*a, *b));
    vec
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
        .iter()
        .zip(vec2.data.iter())
        .map(|(x, y)| x * y)
        .sum()
}

pub fn sum_vector(vec: &Vector) -> f64 {
    vec.data.iter().sum()
}

pub fn mean_vector(vec: &Vector) -> f64 {
    sum_vector(vec) / vec.size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vector() {
        let vec = Vector::new(100, 1.0);
        assert_eq!(vec.size, 100);
        assert!(vec.data.iter().all(|x| *x == 1.0));
    }

    #[test]
    fn test_rand_vector() {
        let vec = Vector::randn(10000, 0.0, 1.0);
        assert_eq!(vec.size, 10000);
        assert_eq!(mean_vector(&vec).round(), 0.0);
    }

    #[test]
    fn test_from_str_vector() {
        let vec = Vector::from_str("1.0 1.0 1.0 1.0 1.0 1.0");
        assert_eq!(vec.size, 6);
        assert!(vec.data.iter().all(|x| *x == 1.0));
    }

    #[test]
    fn test_copy_vector() {
        let vec = Vector::randn(100, 0.0, 1.0);
        let vec_copy = vec.copy();
        assert_eq!(vec.size, vec_copy.size);
        assert!(vec
            .data
            .iter()
            .zip(vec_copy.data.iter())
            .all(|(x, y)| *x == *y));
    }

    #[test]
    fn test_element_wise_operation_vector() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = element_wise_operation_vector(&vec1, |x| 2.0 * x + 1.0);
        assert_eq!(vec1.size, vec2.size);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == *x * 2.0 + 1.0));
    }

    #[test]
    fn test_add_scalar_vector() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = add_scalar_vector(2.0, &vec1);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == 2.0 + *x));
    }

    #[test]
    fn test_subtract_scalar_vector() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = subtract_scalar_vector(2.0, &vec1);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == *x - 2.0));
    }

    #[test]
    fn test_multiply_scalar_vector() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = multiply_scalar_vector(2.0, &vec1);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == *x * 2.0));
    }

    #[test]
    fn test_element_wise_operation_vectors() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = Vector::randn(100, 0.0, 1.0);
        let vec3 = element_wise_operation_vectors(&vec1, &vec2, |x, y| x * 2.0 + y);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *y + *x * 2.0));
    }

    #[test]
    fn test_add_vectors() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = Vector::randn(100, 0.0, 1.0);
        let vec3 = add_vectors(&vec1, &vec2);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *y + *x));
    }

    #[test]
    fn test_subtract_vectors() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = Vector::randn(100, 0.0, 1.0);
        let vec3 = subtract_vectors(&vec1, &vec2);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *x - *y));
    }

    #[test]
    fn test_multiply_vectors() {
        let vec1 = Vector::randn(100, 0.0, 1.0);
        let vec2 = Vector::randn(100, 0.0, 1.0);
        let vec3 = multiply_vectors(&vec1, &vec2);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *x * *y));
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
}
