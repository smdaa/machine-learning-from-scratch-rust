use crate::activation_function::*;
use crate::layer::*;
use crate::matrix::*;

pub struct Network {
    pub layers: Vec<Dense>,
    pub activations: Vec<fn(&Matrix) -> Matrix>,
}

impl Network {
    pub fn new(config: &Vec<((&str, usize, usize, usize), &str)>) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        for ((layer_type, in_size, out_size, batch_size), activation_fn) in config {
            match *layer_type {
                "dense" => {
                    let dense = Dense::new(*in_size, *out_size, *batch_size);
                    layers.push(dense);
                }
                _ => panic!("Invalid layer type: {}", layer_type),
            }
            match *activation_fn {
                "tanh" => activations.push(tanh as fn(&Matrix) -> Matrix),
                "sigmoid" => activations.push(sigmoid as fn(&Matrix) -> Matrix),
                _ => panic!("Invalid activation function: {}", activation_fn),
            }
        }

        Network {
            layers,
            activations,
        }
    }
}

pub fn forward_pass_network(network: &Network, x: &Matrix) -> Matrix {
    network
        .layers
        .iter()
        .zip(network.activations.iter())
        .fold(x.copy(), |input, (dense, activation_fn)| {
            activation_pass(&forward_pass_dense(dense, &input), activation_fn)
        })
}
