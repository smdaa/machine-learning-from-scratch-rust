use rand::Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};

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
}

pub fn transpose(mat: &Matrix) -> Matrix {
    let mut new_mat: Matrix = Matrix::new(mat.n_rows, mat.n_columns, 0.0);
    for i in 0..mat.n_rows {
        for j in 0..mat.n_columns {
            new_mat.data[j][i] = mat.data[i][j]
        }
    }
    new_mat
}

pub fn minor(mat: &Matrix, i: usize, j: usize) -> Matrix {
    let mut new_mat = mat.copy();
    // remove ith row
    new_mat.data.remove(i);
    new_mat.n_rows -= 1;

    // remove jth column
    new_mat.data.iter_mut().for_each(|row| {
        row.remove(j);
    });
    new_mat.n_columns -= 1;

    new_mat
}

// O(n!)
pub fn naive_det(mat: &Matrix) -> f64 {
    assert_eq!(mat.n_rows, mat.n_columns, "Matrix must be square");

    match mat.n_rows {
        0 => 1.0,
        1 => mat.data[0][0],
        2 => mat.data[0][0] * mat.data[1][1] - mat.data[0][1] * mat.data[1][0],
        _ => {
            let mut acc = 0.0;
            for j in 0..mat.n_columns {
                let minor = minor(mat, 0, j);
                let sign = if j & 1 == 1 { -1.0 } else { 1.0 };
                acc += (sign) * mat.data[0][j] * naive_det(&minor);
            }

            acc
        }
    }
}

// using Gauss elimination
pub fn det(mat: &Matrix) -> f64 {
    assert_eq!(mat.n_rows, mat.n_columns, "Matrix must be square");
    let mut mat_copy = mat.copy();
    let mut acc = 1.0;

    for j in 0..mat_copy.n_columns {
        // find and swap pivot
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
            //swap rows j and pivot_row
            mat_copy.data.swap(j, pivot_row);
            acc *= -1.0;
        }

        acc *= pivot;

        // update elements
        for i in j + 1..mat_copy.n_rows {
            for c in j + 1..mat_copy.n_columns {
                mat_copy.data[i][c] -= mat_copy.data[i][j] * mat_copy.data[j][c] / pivot;
            }
        }
    }

    acc
}

pub fn dot(mat1:&Matrix, mat2:&Matrix) -> Matrix {
    assert_eq!(mat1.n_columns, mat2.n_rows);
    let mut mat = Matrix::new(mat1.n_rows, mat2.n_columns, 0.0);

    for i in 0..mat.n_rows{
        for j in 0..mat.n_columns {
            let mut acc = 0.0;
            for k in 0..mat1.n_columns{
                acc += mat1.data[i][k] * mat2.data[k][j];
            }
            mat.data[i][j] = acc;
        }
    }

    mat
}

pub fn element_wise_operation_matrix(mat:&Matrix, op: impl Fn(f64) -> f64) -> Matrix{
    let mut new_mat = mat.copy();
    new_mat.data.iter_mut().for_each(|row| row.iter_mut().for_each(|x| *x = op(*x)));
    new_mat
}

pub fn add_scalar_matrix(scalar:f64, mat:&Matrix) -> Matrix{
    element_wise_operation_matrix(mat, |x| scalar+x)
}

pub fn subtract_scalar_matrix(scalar:f64, mat:&Matrix) -> Matrix{
    element_wise_operation_matrix(mat, |x| x-scalar)
}

pub fn multiply_scalar_matrix(scalar:f64, mat:&Matrix) -> Matrix{
    element_wise_operation_matrix(mat, |x| scalar*x)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let mat = Matrix::new(100, 100, 1.0);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        assert!(mat.data.iter().all(|row| row.iter().all(|x| *x==1.0)));
    }

    #[test]
    fn test_eye(){
        let mat = Matrix::eye(100);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        for i in 0..mat.n_rows{
            for j in 0..mat.n_columns{
                if j==i{
                    assert_eq!(mat.data[i][j], 1.0);
                } else {
                    assert_eq!(mat.data[i][j], 0.0);
                }

            }
        }
    }

    #[test]
    fn test_rand(){
        let mat = Matrix::rand(100, 100);
        assert_eq!(mat.n_rows, 100);
        assert_eq!(mat.n_columns, 100);
        assert!(mat.data.iter().all(|row| row.iter().all(|x| *x > 0.0 && *x < 1.0)));
    }

    #[test]
    fn test_from_str(){
        let mat = Matrix::from_str("1.0 1.0 1.0, 1.0 1.0 1.0, 1.0 1.0 1.0");
        assert_eq!(mat.n_rows, 3);
        assert_eq!(mat.n_columns, 3);
        assert!(mat.data.iter().all(|row| row.iter().all(|x| *x==1.0)));
    }

    #[test]
    fn test_shape(){
        let mat = Matrix::rand(100, 100);
        assert_eq!(mat.shape(), (100, 100));
    }

    #[test]
    fn test_size(){
        let mat = Matrix::rand(100, 100);
        assert_eq!(mat.size(), 10000);
    }

}
