use rand::Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::vec;

pub struct Vector {
    pub size: usize,
    data: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize, value: f64) -> Self {
        Vector {
            size: size,
            data: vec![value; size],
        }
    }

    pub fn rand(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..size).map(|_| rng.gen_range(0.0..1.0)).collect();
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

pub struct Matrix {
    pub n_rows: usize,
    pub n_columns: usize,
    data: Vec<Vec<f64>>,
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

    pub fn rand(n_rows: usize, n_columns: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for _ in 0..n_rows {
            data.push((0..n_columns).map(|_| rng.gen_range(0.0..1.0)).collect());
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
) -> Matrix {
    assert_eq!(vec.size, mat.n_columns);
    let mut new_mat = mat.copy();
    new_mat.data.iter_mut().for_each(|row| {
        row.iter_mut()
            .zip(vec.data.iter())
            .for_each(|(x, y)| *x = op(*x, *y))
    });
    new_mat
}

pub fn add_vector_matrix(mat: &Matrix, vec: &Vector) -> Matrix {
    element_wise_operation_vector_matrix(mat, vec, |a, b| a + b)
}

pub fn subtract_vector_matrix(mat: &Matrix, vec: &Vector) -> Matrix {
    element_wise_operation_vector_matrix(mat, vec, |a, b| a - b)
}

pub fn multiply_vector_matrix(mat: &Matrix, vec: &Vector) -> Matrix {
    element_wise_operation_vector_matrix(mat, vec, |a, b| a * b)
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
        let vec = Vector::rand(100);
        assert_eq!(vec.size, 100);
        assert!(vec.data.iter().all(|x| *x > 0.0 && *x < 1.0));
    }

    #[test]
    fn test_from_str_vector() {
        let vec = Vector::from_str("1.0 1.0 1.0 1.0 1.0 1.0");
        assert_eq!(vec.size, 6);
        assert!(vec.data.iter().all(|x| *x == 1.0));
    }

    #[test]
    fn test_copy_vector() {
        let vec = Vector::rand(100);
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
        let vec1 = Vector::rand(100);
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
        let vec1 = Vector::rand(100);
        let vec2 = add_scalar_vector(2.0, &vec1);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == 2.0 + *x));
    }

    #[test]
    fn test_subtract_scalar_vector() {
        let vec1 = Vector::rand(100);
        let vec2 = subtract_scalar_vector(2.0, &vec1);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == *x - 2.0));
    }

    #[test]
    fn test_multiply_scalar_vector() {
        let vec1 = Vector::rand(100);
        let vec2 = multiply_scalar_vector(2.0, &vec1);
        assert!(vec1
            .data
            .iter()
            .zip(vec2.data.iter())
            .all(|(x, y)| *y == *x * 2.0));
    }

    #[test]
    fn test_element_wise_operation_vectors() {
        let vec1 = Vector::rand(100);
        let vec2 = Vector::rand(100);
        let vec3 = element_wise_operation_vectors(&vec1, &vec2, |x, y| x * 2.0 + y);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *y + *x * 2.0));
    }

    #[test]
    fn test_add_vectors() {
        let vec1 = Vector::rand(100);
        let vec2 = Vector::rand(100);
        let vec3 = add_vectors(&vec1, &vec2);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *y + *x));
    }

    #[test]
    fn test_subtract_vectors() {
        let vec1 = Vector::rand(100);
        let vec2 = Vector::rand(100);
        let vec3 = subtract_vectors(&vec1, &vec2);
        assert!(vec3
            .data
            .iter()
            .zip(vec1.data.iter().zip(vec2.data.iter()))
            .all(|(z, (x, y))| *z == *x - *y));
    }

    #[test]
    fn test_multiply_vectors() {
        let vec1 = Vector::rand(100);
        let vec2 = Vector::rand(100);
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
        let mat = Matrix::rand(100, 200);
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
        let mat = Matrix::rand(100, 200);
        assert_eq!(mat.shape(), (100, 200));
    }

    #[test]
    fn test_size() {
        let mat = Matrix::rand(100, 200);
        assert_eq!(mat.size(), 20000);
    }

    #[test]
    fn test_copy_matrix() {
        let mat = Matrix::rand(100, 200);
        let mat_copy = mat.copy();
        assert_eq!(mat.shape(), mat_copy.shape());
        assert!(mat.is_equal(&mat_copy));
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix::rand(100, 200);
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
        let mat = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
        let mat2 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
        let mat2 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
        let mat2 = Matrix::rand(100, 200);
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
        let mat1 = Matrix::rand(100, 200);
        let mat2 = Matrix::rand(100, 200);
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
        assert!(add_vector_matrix(&mat, &vec).is_equal(&Matrix::from_str("3 0 2, 2 -2 1")))
    }

    #[test]
    fn test_subtract_vector_matrix() {
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1 0");
        assert!(subtract_vector_matrix(&mat, &vec).is_equal(&Matrix::from_str("-1 -2 2, -2 -4 1")))
    }

    #[test]
    fn test_multiply_vector_matrix() {
        let mat = Matrix::from_str("1 -1 2, 0 -3 1");
        let vec = Vector::from_str("2 1 0");
        assert!(multiply_vector_matrix(&mat, &vec).is_equal(&Matrix::from_str("2 -1 0, 0 -3 0")))
    }
}
