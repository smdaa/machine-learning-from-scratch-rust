use crate::common::matrix::*;
use crate::common::vector::*;
use num_traits::float::Float;
use rand_distr::uniform::SampleUniform;
use std::fmt::Display;
use std::str::FromStr;

pub fn qr_decomposition<T: Float + SampleUniform + FromStr + Display + Send + Sync>(
    matrix: &Matrix<T>,
) -> (bool, Matrix<T>, Matrix<T>) {
    /*
       For an m-by-n matrix A with m >= n, the QR decomposition is an m-by-n
       orthogonal matrix Q and an n-by-n upper triangular matrix R so that
       A = Q*R.
    */
    let m = matrix.n_rows;
    let n = matrix.n_columns;
    let mut qr = matrix.clone();
    let mut r_diag: Vector<T> = Vector::zeros(n);

    /*
       Main loop
    */
    for k in 0..n {
        /*
           Compute 2-norm of k-th column without under/overflow.
        */
        let mut nrm = T::zero();
        for i in k..m {
            nrm = (nrm.powi(2) + qr.data[i * n + k].powi(2)).sqrt();
        }

        if nrm != T::zero() {
            /*
               Form k-th Householder vector.
            */
            if qr.data[k * n + k] < T::zero() {
                nrm = -nrm;
            }
            for i in k..m {
                qr.data[i * n + k] = qr.data[i * n + k] / nrm;
            }
            qr.data[k * n + k] = qr.data[k * n + k] + T::one();

            /*
               Apply transformation to remaining columns.
            */
            for j in k + 1..n {
                let mut s = T::zero();
                for i in k..m {
                    s = s + qr.data[i * n + k] * qr.data[i * n + j];
                }
                s = -s / qr.data[k * n + k];
                for i in k..m {
                    qr.data[i * n + j] = qr.data[i * n + j] + s * qr.data[i * n + k];
                }
            }
        }
        r_diag.data[k] = -nrm;
    }

    /*
       Is A full rank?
    */
    let full_rank = r_diag.data.iter().all(|x| *x != T::zero());

    /*
       Construct r.
    */
    let mut r: Matrix<T> = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i < j {
                r.data[i * n + j] = qr.data[i * n + j];
            } else if i == j {
                r.data[i * n + j] = r_diag.data[i];
            } else {
                r.data[i * n + j] = T::zero();
            }
        }
    }

    /*
       Construct q (economy-sized).
    */
    let mut q: Matrix<T> = Matrix::zeros(m, n);
    for k in (0..n).rev() {
        for i in 0..m {
            q.data[i * n + k] = T::zero();
        }
        q.data[k * n + k] = T::one();
        for j in k..n {
            if qr.data[k * n + k] != T::zero() {
                let mut s = T::zero();
                for i in k..m {
                    s = s + qr.data[i * n + k] * q.data[i * n + j];
                }
                s = -s / qr.data[k * n + k];
                for i in k..m {
                    q.data[i * n + j] = q.data[i * n + j] + s * qr.data[i * n + k];
                }
            }
        }
    }

    (full_rank, q, r)
}

pub fn back_substitution<T: Float + SampleUniform + FromStr + Display + Send + Sync>(
    a: &Matrix<T>,
    b: &Vector<T>,
) -> Vector<T> {
    let n = b.n;
    let m = a.n_columns;
    let mut x: Vector<T> = Vector::zeros(n);

    for i in (0..n).rev() {
        let mut s = T::zero();
        for j in i + 1..n {
            s = s + a.data[i * m + j] * x.data[j];
        }
        x.data[i] = (b.data[i] - s) / a.data[i * m + i]
    }
    x
}

pub fn least_squares<T: Float + SampleUniform + FromStr + Display + Send + Sync>(
    a: &Matrix<T>,
    b: &Vector<T>,
) -> Vector<T> {
    // compute QR factorization 𝐴 = 𝑄𝑅 (2𝑚𝑛2 flops if 𝐴 is 𝑚 × 𝑛)
    let (_, q, r) = qr_decomposition(&a);

    // matrix-vector product 𝑑 = 𝑄𝑇 𝑏 (2𝑚𝑛 flops)
    let d = q.transpose().dot_vector(&b);

    // solve 𝑅𝑥 = 𝑑 by back substitution (𝑛2 flops)
    let x = back_substitution(&r, &d);

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_qr_decomposition() {
        let a = Matrix {
            n_rows: 4,
            n_columns: 3,
            data: vec![
                1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0,
            ],
        };
        let (full_rank, q, r) = qr_decomposition(&a);
        let a_ = q.dot_matrix(&r);
        assert!(full_rank);
        assert_eq!((a.n_rows, a.n_columns), (a_.n_rows, a_.n_columns));
        assert!(a
            .data
            .iter()
            .zip(a_.data.iter())
            .all(|(x, y)| (x - y).abs() < f64::EPSILON.sqrt()));

        let a = Matrix {
            n_rows: 3,
            n_columns: 3,
            data: vec![3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 4.0, 2.0, 3.0],
        };
        let (full_rank, q, r) = qr_decomposition(&a);
        let a_ = q.dot_matrix(&r);
        assert!(full_rank);
        assert_eq!((a.n_rows, a.n_columns), (a_.n_rows, a_.n_columns));
        assert!(a
            .data
            .iter()
            .zip(a_.data.iter())
            .all(|(x, y)| (x - y).abs() < f64::EPSILON.sqrt()));
    }

    #[test]
    fn test_back_substitution() {
        let a = Matrix {
            n_rows: 3,
            n_columns: 3,
            data: vec![1.0, -2.0, 1.0, 0.0, 1.0, 6.0, 0.0, 0.0, 1.0],
        };
        let b = Vector {
            n: 3,
            data: vec![4.0, -1.0, 2.0],
        };
        let x = back_substitution(&a, &b);
        assert_eq!(x.n, 3);
        assert_eq!(x.data, vec![-24.0, -13.0, 2.0]);
    }

    #[test]
    fn test_least_squares() {
        let a = Matrix {
            n_rows: 3,
            n_columns: 2,
            data: vec![3.0, -6.0, 4.0, -8.0, 0.0, 1.0],
        };
        let b = Vector {
            n: 3,
            data: vec![-1.0, 7.0, 2.0],
        };
        let x = least_squares(&a, &b);
        assert_eq!(x.n, 2);
        assert_eq!(x.data, vec![5.0, 2.0]);
    }
}
