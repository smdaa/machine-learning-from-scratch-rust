#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]

use std::process::exit;

mod cost_functions;
mod layer;
mod matrix;

use cost_functions::*;
use layer::*;
use matrix::*;

pub fn batch(x: &Matrix, batch_size: usize) -> Vec<Matrix> {
    let n_rows = x.n_rows;
    let n_columns = x.n_columns;
    let n_batches = n_rows / batch_size;

    let batches = (0..n_batches)
        .map(|i| {
            let start_idx = i * batch_size;
            let end_idx = start_idx + batch_size;
            x.slice((start_idx, end_idx - 1), (0, n_columns - 1))
        })
        .collect();

    batches
}

pub fn unbatch(batches: &Vec<Matrix>) -> Matrix {
    let n_columns = batches[0].n_columns;
    let n_rows: usize = batches.iter().map(|batch| batch.n_rows).sum();
    let data: Vec<f32> = batches
        .iter()
        .flat_map(|batch| batch.data.clone())
        .collect();

    Matrix {
        n_rows,
        n_columns,
        data,
    }
}

fn main() {

    let in_size = 3;
    let batch_size = 5;
    let mut layer = SoftmaxLayer::new(in_size, batch_size);
    let x = Matrix::from_str("10 11 12, 13 14 15, 16 17 18, 19 20 21, 22 23 24");
    layer.forward(&x);
    layer.a.print();

    exit(0);
    // load training data
    let x_train = Matrix::from_txt("./test_data/test_2d_clustering/2_circles/x_train.txt");
    let y_train = Matrix::from_txt("./test_data/test_2d_clustering/2_circles/y_train.txt");
    assert_eq!(x_train.n_rows, y_train.n_rows);

    let n_train: usize = x_train.n_rows;
    let in_size: usize = x_train.n_columns;
    let hidden_size: usize = 8;
    let out_size: usize = 1;
    let learning_rate = 0.1;
    let batch_size: usize = 32;
    let n_epochs = 10;

    // create batches
    let x_batches = batch(&x_train, batch_size);
    let y_batches = batch(&y_train, batch_size);
    let n_batches = x_batches.len();

    // create network
    let mut linear_layer_0 = LinearLayer::new(in_size, hidden_size, batch_size);
    let mut relu_layer = ReluLayer::new(hidden_size, batch_size);
    let mut linear_layer_1 = LinearLayer::new(hidden_size, out_size, batch_size);
    let mut sigmoid_layer = SigmoidLayer::new(out_size, batch_size);

    // train
    println!(
        "------------------- Training start n_train: {} -------------------",
        n_train
    );
    use std::time::Instant;
    let now = Instant::now();
    for i in 0..n_epochs {
        let mut loss_avg = 0.0;
        let mut total_correct = 0;

        for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
            // forward pass
            linear_layer_0.forward(&x_batch);
            relu_layer.forward(&(linear_layer_0.z));
            linear_layer_1.forward(&(relu_layer.a));
            sigmoid_layer.forward(&(linear_layer_1.z));

            // compute loss
            let (loss, mut grad) = binary_cross_entropy(y_batch, &(sigmoid_layer.a));

            // backward pass
            sigmoid_layer.backward();
            grad.multiply_matrix(&(sigmoid_layer.grad));

            linear_layer_1.backward(&grad);

            relu_layer.backward();
            let mut grad = relu_layer.grad.copy();
            grad.multiply_matrix(&(linear_layer_1.grad));

            linear_layer_0.backward(&grad);

            // update weights
            linear_layer_0.update_weights(learning_rate);
            linear_layer_1.update_weights(learning_rate);

            let correct = sigmoid_layer
                .a
                .data
                .iter()
                .zip(y_batch.data.iter())
                .map(|(&y_hat_n, &y_n)| (y_n == y_hat_n.round()) as i32)
                .sum::<i32>();

            loss_avg = loss_avg + loss;
            total_correct = total_correct + correct
        }

        println!(
            "\tepoch : {}, loss : {}, precision : {}",
            i,
            loss_avg,
            (total_correct as f32) / (n_train as f32)
        );
    }
    let elapsed = now.elapsed();
    println!(
        "------------------- Training end elapsed time: {:?} -------------------",
        elapsed
    );

    // load testing data
    let x_test = Matrix::from_txt("./test_data/test_2d_clustering/2_circles/x_test.txt");
    let y_test = Matrix::from_txt("./test_data/test_2d_clustering/2_circles/y_test.txt");
    assert_eq!(x_test.n_rows, y_test.n_rows);

    let n_test: usize = x_test.n_rows;

    let x_batches = batch(&x_test, batch_size);
    let y_batches = batch(&y_test, batch_size);

    let mut total_correct = 0;
    for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
        // forward pass
        linear_layer_0.forward(&x_batch);
        relu_layer.forward(&(linear_layer_0.z));
        linear_layer_1.forward(&(relu_layer.a));
        sigmoid_layer.forward(&(linear_layer_1.z));

        let correct = sigmoid_layer
            .a
            .data
            .iter()
            .zip(y_batch.data.iter())
            .map(|(&y_hat_n, &y_n)| (y_n == y_hat_n.round()) as i32)
            .sum::<i32>();

        total_correct = total_correct + correct
    }

    println!(
        "n_test : {}, precision : {}",
        n_test,
        (total_correct as f32) / (n_test as f32)
    );
}
