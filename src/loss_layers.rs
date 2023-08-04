use crate::matrix::*;
use num_traits::float::Float;
use rand_distr::uniform::SampleUniform;
use std::fmt::Display;
use std::str::FromStr;

pub trait LossLayer<T> {
    fn forward(&mut self, z: &Matrix<T>, y: &Matrix<T>);
    fn backward(&mut self, y: &Matrix<T>);
    fn loss(&self) -> T;
}

pub struct BCELossLayer<T> {
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix<T>,
    pub grad: Matrix<T>,
}

impl<T: Float + SampleUniform + FromStr + Display + Send + Sync> BCELossLayer<T> {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size,
            loss: 0.0,
            a: Matrix::zeros(batch_size, 1),
            grad: Matrix::zeros(batch_size, 1),
        }
    }
    pub fn forward(&mut self, z: &Matrix<T>, y: &Matrix<T>) {
        self.a.copy_content_from(z);
        self.a.element_wise_operation(|x| 1.0 / (1.0 + (-x).exp()));
        self.loss = z
            .data
            .iter()
            .zip(y.data.iter())
            .map(|(z_n, y_n)| z_n.max(0.0) - z_n * y_n + (1.0 + (-z_n.abs()).exp()).ln())
            .sum::<f32>()
            / (self.batch_size as f32);
    }
}
