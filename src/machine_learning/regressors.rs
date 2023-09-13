use crate::common::matrix::*;
use crate::common::vector::*;
use num_traits::float::Float;
use rand_distr::uniform::SampleUniform;
use std::fmt::Display;
use std::str::FromStr;

pub fn linear_regression<T: Float + SampleUniform + FromStr + Display + Send + Sync>(
    x: Matrix<T>,
    y: Vector<T>,
) {
}
