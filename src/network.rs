use crate::activation_functions::*;
use crate::cost_functions::*;
use crate::layer::*;
use crate::matrix::*;

pub struct Network {
    pub n_layers: usize,
    pub layers: Vec<Linear>,
    pub cost_function: fn(&Matrix, &Matrix) -> Matrix,
}

impl Network {
    pub fn new(config: &((usize, &str), Vec<(&str, usize, usize, &str)>)) -> Self {
        let batch_size = (config.0).0;
        let cost_function: fn(&Matrix, &Matrix) -> Matrix;
        match (config.0).1 {
            "cross_entropy" => cost_function = cross_entropy as fn(&Matrix, &Matrix) -> Matrix,
            _ => panic!("Invalid cost function: {}", (config.0).1),
        }

        let mut layers = Vec::new();
        for (layer_type, in_size, out_size, activation_fn) in &config.1 {
            match *layer_type {
                "dense" => {
                    let activation_function: fn(&Matrix) -> Matrix;
                    let d_activation_function: fn(&Matrix) -> Matrix;
                    match *activation_fn {
                        "tanh" => {
                            activation_function = tanh as fn(&Matrix) -> Matrix;
                            d_activation_function = tanh as fn(&Matrix) -> Matrix
                        }
                        "sigmoid" => {
                            activation_function = sigmoid as fn(&Matrix) -> Matrix;
                            d_activation_function = d_sigmoid as fn(&Matrix) -> Matrix
                        }
                        "softmax" => {
                            activation_function = softmax as fn(&Matrix) -> Matrix;
                            d_activation_function = softmax as fn(&Matrix) -> Matrix
                        }
                        _ => panic!("Invalid activation function: {}", activation_fn),
                    }
                    let dense = Linear::new(
                        *in_size,
                        *out_size,
                        batch_size,
                        activation_function,
                        d_activation_function,
                    );
                    layers.push(dense);
                }
                _ => panic!("Invalid layer type: {}", layer_type),
            }
        }
        let n_layers = layers.len();
        Network {
            n_layers,
            layers,
            cost_function,
        }
    }

    pub fn forward_pass(&mut self, x_train: &Vec<Matrix>) -> Vec<Matrix> {
        x_train
            .iter()
            .map(|x_batch| {
                self.layers
                    .iter_mut()
                    .fold(x_batch.copy(), |input, layer| layer.forward_pass(&input))
            })
            .collect()
    }

    pub fn backward_pass(&mut self, x_train: &Vec<Matrix>, y_train: &Vec<Matrix>) {
        for i in 0..y_train.len() {
            let mut right_grad_output =
                (self.cost_function)(&self.layers[self.n_layers - 1].a, &y_train[i]);
            for j in (1..self.n_layers).rev() {
                let left_activation_output = &self.layers[j - 1].a.copy();
                self.layers[j].backward_pass(&right_grad_output, left_activation_output);

                right_grad_output =
                    dot_matrix_matrix(&self.layers[j].db, &transpose(&self.layers[j].w));
            }
            self.layers[0].backward_pass(&right_grad_output, &x_train[i]);
        }
    }
}
