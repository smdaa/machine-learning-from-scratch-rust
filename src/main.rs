#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unreachable_code)]

use std::process::exit;

use image::{ImageBuffer, RgbImage};

//mod layers;
mod matrix;

//use layers::*;
use matrix::*;

/*
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

pub fn test_mnist() {
    // load training data
    let x_train = Matrix::from_txt("./test_data/test_multiclass_clustering/mnist/x_train.txt");
    let y_train = Matrix::from_txt("./test_data/test_multiclass_clustering/mnist/y_train.txt");
    assert_eq!(x_train.n_rows, y_train.n_rows);

    let n_train: usize = x_train.n_rows;
    let in_size: usize = x_train.n_columns;
    let hidden_size = 32;
    let out_size: usize = 10;
    let learning_rate = 0.01;
    let batch_size: usize = 64;
    let n_epochs = 20;

    // create batches
    let x_batches = batch(&x_train, batch_size);
    let y_batches = batch(&y_train, batch_size);
    let n_batches = x_batches.len();

    // create network
    let mut linear_layer_0 = LinearLayer::new(in_size, hidden_size, batch_size);
    let mut relu_layer = ReluLayer::new(hidden_size, batch_size);
    let mut linear_layer_1 = LinearLayer::new(hidden_size, out_size, batch_size);
    let mut ce_layer = CELossLayer::new(out_size, batch_size);

    // train
    println!(
        "------------------- Training start n_train: {} -------------------",
        n_train
    );
    use std::time::Instant;
    let now = Instant::now();
    for i in 0..n_epochs {
        let mut loss_avg = 0.0;
        let mut total_correct = 0.0;

        for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
            // forward pass
            linear_layer_0.forward(&x_batch);
            relu_layer.forward(&linear_layer_0.z);
            linear_layer_1.forward(&relu_layer.a);
            ce_layer.forward(&(linear_layer_1.z), y_batch);

            // backward pass
            ce_layer.backward(y_batch);
            linear_layer_1.backward(&(ce_layer.grad));
            relu_layer.backward(&linear_layer_1.grad);
            linear_layer_0.backward(&relu_layer.grad);

            // update weights
            linear_layer_0.update_weights(learning_rate);
            linear_layer_1.update_weights(learning_rate);

            let prediction = ce_layer.a.max_idx(1);
            let truth = y_batch.max_idx(1);
            let correct = prediction
                .data
                .iter()
                .zip(truth.data.iter())
                .map(|(&y_hat_n, &y_n)| {
                    ((y_n - y_hat_n.round()).abs() < f32::EPSILON.sqrt()) as i32
                })
                .sum::<i32>();
            total_correct += correct as f32;
            loss_avg += ce_layer.loss;
        }
        loss_avg /= n_batches as f32;
        total_correct /= n_train as f32;

        println!(
            "\tepoch : {}, loss : {}, precision : {}",
            i, loss_avg, total_correct
        );
    }
    let elapsed = now.elapsed();
    println!(
        "------------------- Training end elapsed time: {:?} -------------------",
        elapsed
    );

    // load testing data
    let x_test = Matrix::from_txt("./test_data/test_multiclass_clustering/mnist/x_test.txt");
    let y_test = Matrix::from_txt("./test_data/test_multiclass_clustering/mnist/y_test.txt");
    assert_eq!(x_test.n_rows, y_test.n_rows);

    let n_test: usize = x_test.n_rows;

    let x_batches = batch(&x_test, batch_size);
    let y_batches = batch(&y_test, batch_size);

    let mut total_correct = 0.0;
    for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
        // forward pass
        linear_layer_0.forward(&x_batch);
        relu_layer.forward(&linear_layer_0.z);
        linear_layer_1.forward(&relu_layer.a);
        ce_layer.forward(&(linear_layer_1.z), y_batch);

        let prediction = ce_layer.a.max_idx(1);
        let truth = y_batch.max_idx(1);
        let correct = prediction
            .data
            .iter()
            .zip(truth.data.iter())
            .map(|(&y_hat_n, &y_n)| ((y_n - y_hat_n.round()).abs() < f32::EPSILON.sqrt()) as i32)
            .sum::<i32>();
        total_correct += correct as f32;
    }
    println!(
        "n_test : {}, total_correct : {}, precision : {}",
        n_test,
        total_correct,
        (total_correct / (n_test as f32))
    );
}

pub fn test_random() {
    // load training data
    let x_train = Matrix::from_txt("./test_data/test_multiclass_clustering/random/x_train.txt");
    let y_train = Matrix::from_txt("./test_data/test_multiclass_clustering/random/y_train.txt");
    assert_eq!(x_train.n_rows, y_train.n_rows);

    let n_train: usize = x_train.n_rows;
    let in_size: usize = x_train.n_columns;
    let hidden_size = 64;
    let out_size: usize = 3;
    let learning_rate = 0.01;
    let batch_size: usize = 32;
    let n_epochs = 1000;

    // create batches
    let x_batches = batch(&x_train, batch_size);
    let y_batches = batch(&y_train, batch_size);
    let n_batches = x_batches.len();

    // create network
    let mut linear_layer_0 = LinearLayer::new(in_size, hidden_size, batch_size);
    let mut relu_layer = ReluLayer::new(hidden_size, batch_size);
    let mut linear_layer_1 = LinearLayer::new(hidden_size, out_size, batch_size);
    let mut ce_layer = CELossLayer::new(out_size, batch_size);

    // train
    println!(
        "------------------- Training start n_train: {} -------------------",
        n_train
    );
    use std::time::Instant;
    let now = Instant::now();
    for i in 0..n_epochs {
        let mut loss_avg = 0.0;
        let mut total_correct = 0.0;

        for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
            // forward pass
            linear_layer_0.forward(&x_batch);
            relu_layer.forward(&linear_layer_0.z);
            linear_layer_1.forward(&relu_layer.a);
            ce_layer.forward(&(linear_layer_1.z), y_batch);

            // backward pass
            ce_layer.backward(y_batch);
            linear_layer_1.backward(&(ce_layer.grad));
            relu_layer.backward(&linear_layer_1.grad);
            linear_layer_0.backward(&relu_layer.grad);

            // update weights
            linear_layer_0.update_weights(learning_rate);
            linear_layer_1.update_weights(learning_rate);

            let prediction = ce_layer.a.max_idx(1);
            let truth = y_batch.max_idx(1);
            let correct = prediction
                .data
                .iter()
                .zip(truth.data.iter())
                .map(|(&y_hat_n, &y_n)| {
                    ((y_n - y_hat_n.round()).abs() < f32::EPSILON.sqrt()) as i32
                })
                .sum::<i32>();
            total_correct += correct as f32;
            loss_avg += ce_layer.loss;
        }
        loss_avg /= n_batches as f32;
        total_correct /= n_train as f32;

        println!(
            "\tepoch : {}, loss : {}, precision : {}",
            i, loss_avg, total_correct
        );
    }
    let elapsed = now.elapsed();
    println!(
        "------------------- Training end elapsed time: {:?} -------------------",
        elapsed
    );

    // load testing data
    let x_test = Matrix::from_txt("./test_data/test_multiclass_clustering/random/x_test.txt");
    let y_test = Matrix::from_txt("./test_data/test_multiclass_clustering/random/y_test.txt");
    assert_eq!(x_test.n_rows, y_test.n_rows);

    let n_test: usize = x_test.n_rows;

    let x_batches = batch(&x_test, batch_size);
    let y_batches = batch(&y_test, batch_size);

    let mut total_correct = 0.0;
    for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
        // forward pass
        linear_layer_0.forward(&x_batch);
        relu_layer.forward(&linear_layer_0.z);
        linear_layer_1.forward(&relu_layer.a);
        ce_layer.forward(&(linear_layer_1.z), y_batch);

        let prediction = ce_layer.a.max_idx(1);
        let truth = y_batch.max_idx(1);
        let correct = prediction
            .data
            .iter()
            .zip(truth.data.iter())
            .map(|(&y_hat_n, &y_n)| ((y_n - y_hat_n.round()).abs() < f32::EPSILON.sqrt()) as i32)
            .sum::<i32>();
        total_correct += correct as f32;
    }
    println!(
        "n_test : {}, total_correct : {}, precision : {}",
        n_test,
        total_correct,
        (total_correct / (n_test as f32))
    );
}

pub fn test_2_circles() {
    // load training data
    let x_train = Matrix::from_txt("./test_data/test_binary_clustering/2_circles/x_train.txt");
    let y_train = Matrix::from_txt("./test_data/test_binary_clustering/2_circles/y_train.txt");
    assert_eq!(x_train.n_rows, y_train.n_rows);

    let n_train: usize = x_train.n_rows;
    let in_size: usize = x_train.n_columns;
    let hidden_size: usize = 8;
    let out_size: usize = 1;
    let learning_rate = 0.1;
    let batch_size: usize = 32;
    let n_epochs = 1000;

    // create batches
    let x_batches = batch(&x_train, batch_size);
    let y_batches = batch(&y_train, batch_size);
    let n_batches = x_batches.len();

    // create network
    let mut linear_layer_0 = LinearLayer::new(in_size, hidden_size, batch_size);
    let mut relu_layer = ReluLayer::new(hidden_size, batch_size);
    let mut linear_layer_1 = LinearLayer::new(hidden_size, out_size, batch_size);
    let mut bceloss_layer = BCELossLayer::new(batch_size);

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
            bceloss_layer.forward(&(linear_layer_1.z), y_batch);

            // backward pass
            bceloss_layer.backward(y_batch);
            linear_layer_1.backward(&(bceloss_layer.grad));
            relu_layer.backward(&(linear_layer_1.grad));
            linear_layer_0.backward(&(relu_layer.grad));

            // update weights
            linear_layer_0.update_weights(learning_rate);
            linear_layer_1.update_weights(learning_rate);

            let correct = bceloss_layer
                .a
                .data
                .iter()
                .zip(y_batch.data.iter())
                .map(|(&y_hat_n, &y_n)| {
                    ((y_n - y_hat_n.round()).abs() < f32::EPSILON.sqrt()) as i32
                })
                .sum::<i32>();

            loss_avg += bceloss_layer.loss;
            total_correct = total_correct + correct
        }
        loss_avg /= n_batches as f32;

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
    let x_test = Matrix::from_txt("./test_data/test_binary_clustering/2_circles/x_test.txt");
    let y_test = Matrix::from_txt("./test_data/test_binary_clustering/2_circles/y_test.txt");
    assert_eq!(x_test.n_rows, y_test.n_rows);

    let n_test: usize = x_test.n_rows;

    let x_batches = batch(&x_test, batch_size);
    let y_batches = batch(&y_test, batch_size);

    let mut total_correct = 0.0;
    for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
        // forward pass
        linear_layer_0.forward(&x_batch);
        relu_layer.forward(&(linear_layer_0.z));
        linear_layer_1.forward(&(relu_layer.a));
        bceloss_layer.forward(&(linear_layer_1.z), y_batch);

        let correct = bceloss_layer
            .a
            .data
            .iter()
            .zip(y_batch.data.iter())
            .map(|(&y_hat_n, &y_n)| ((y_n - y_hat_n.round()).abs() < f32::EPSILON.sqrt()) as i32)
            .sum::<i32>();

        total_correct += correct as f32;
    }

    println!(
        "n_test : {}, total_correct : {}, precision : {}",
        n_test,
        total_correct,
        (total_correct / (n_test as f32))
    );
}

*/


fn main() {
    /*
    
    let input_image_height = 500;
    let input_image_width = 500;

    let input_image_red = Matrix::rand(input_image_height, input_image_width, 0.0, 1.0);
    let input_image_green = Matrix::rand(input_image_height, input_image_width, 0.0, 1.0);
    let input_image_blue = Matrix::rand(input_image_height, input_image_width, 0.0, 1.0);

    //let input_image_red = Matrix::new(input_image_height, input_image_width, 0.5);
    //let input_image_green = Matrix::new(input_image_height, input_image_width, 0.5);
    //let input_image_blue = Matrix::new(input_image_height, input_image_width, 0.5);

    let input_image = vec![
        input_image_red.copy(),
        input_image_green.copy(),
        input_image_blue.copy(),
    ];
    let input_batch = vec![input_image];

    let mut layer = Conv2dLayer::new(
        1,
        3,
        3,
        (input_image_height, input_image_width),
        (3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
    );
    layer.forward(&input_batch);
    let output_image_red = layer.output[0][0].copy();
    let output_image_green = layer.output[0][1].copy();
    let output_image_blue = layer.output[0][2].copy();
    let output_image_height = output_image_red.n_rows;
    let output_image_width = output_image_red.n_columns;

    let mut input_image_file: RgbImage = ImageBuffer::new(
        input_image_width.try_into().unwrap(),
        input_image_height.try_into().unwrap(),
    );
    for i in 0..input_image_height {
        for j in 0..input_image_width {
            let red = (255.0 * input_image_red.data[i * input_image_width + j]) as u8;
            let green = (255.0 * input_image_green.data[i * input_image_width + j]) as u8;
            let blue = (255.0 * input_image_blue.data[i * input_image_width + j]) as u8;
            *input_image_file.get_pixel_mut(j.try_into().unwrap(), i.try_into().unwrap()) =
                image::Rgb([red, green, blue]);
        }
    }
    input_image_file.save("input.png").unwrap();

    let mut output_image_file: RgbImage = ImageBuffer::new(
        output_image_width.try_into().unwrap(),
        output_image_height.try_into().unwrap(),
    );
    for i in 0..output_image_height {
        for j in 0..output_image_width {
            let red = (255.0 * output_image_red.data[i * output_image_width + j]) as u8;
            let green = (255.0 * output_image_green.data[i * output_image_width + j]) as u8;
            let blue = (255.0 * output_image_blue.data[i * output_image_width + j]) as u8;
            *output_image_file.get_pixel_mut(j.try_into().unwrap(), i.try_into().unwrap()) =
                image::Rgb([red, green, blue]);
        }
    }
    output_image_file.save("output.png").unwrap();

    // test_mnist();
    // test_2_circles();
    //test_random();
    */
}
