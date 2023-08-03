use crate::matrix::*;

pub struct LinearLayer {
    pub in_size: usize,
    pub out_size: usize,
    pub batch_size: usize,
    pub w: Matrix,
    pub b: Matrix,
    pub z: Matrix,
    pub x: Matrix,
    pub dw: Matrix,
    pub db: Matrix,
    pub grad: Matrix,
}

impl LinearLayer {
    pub fn new(in_size: usize, out_size: usize, batch_size: usize) -> Self {
        let std_dev = (6.0_f32).sqrt() / ((in_size + out_size) as f32).sqrt();
        Self {
            in_size: in_size,
            out_size: out_size,
            batch_size: batch_size,
            w: Matrix::randn(in_size, out_size, -std_dev, std_dev),
            b: Matrix::randn(batch_size, out_size, -std_dev, std_dev),
            z: Matrix::new(batch_size, out_size, 0.0),
            x: Matrix::new(batch_size, in_size, 0.0),
            dw: Matrix::new(in_size, out_size, 0.0),
            db: Matrix::new(batch_size, out_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, x: &Matrix) {
        self.x.copy_from(x);
        self.z.copy_from(&(x.dot_matrix(&self.w)));
        self.z.add_matrix(&self.b);
    }

    pub fn backward(&mut self, upstream_grad: &Matrix) {
        self.dw
            .copy_from(&((self.x.transpose()).dot_matrix(upstream_grad)));

        let temp = upstream_grad.sum(0);
        self.db.copy_from(&(temp.repeat(self.batch_size, 0)));

        self.grad
            .copy_from(&(&upstream_grad.dot_matrix(&(self.w.transpose()))));
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.dw.multiply_scalar(-learning_rate);
        self.w.add_matrix(&(self.dw));

        self.db.multiply_scalar(-learning_rate);
        self.b.add_matrix(&(self.db));
    }
}

pub struct BCELossLayer {
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix,
    pub grad: Matrix,
}

impl BCELossLayer {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size,
            loss: 0.0,
            a: Matrix::new(batch_size, 1, 0.0),
            grad: Matrix::new(batch_size, 1, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix, y: &Matrix) {
        self.a.copy_from(z);
        self.a.element_wise_operation(|x| 1.0 / (1.0 + (-x).exp()));
        self.loss = z
            .data
            .iter()
            .zip(y.data.iter())
            .map(|(z_n, y_n)| z_n.max(0.0) - z_n * y_n + (1.0 + (-z_n.abs()).exp()).ln())
            .sum::<f32>()
            / (self.batch_size as f32);
    }

    pub fn backward(&mut self, y: &Matrix) {
        self.grad.copy_from(&(self.a));
        self.grad.subtract_matrix(y);
    }
}

pub struct ReluLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub a: Matrix,
    pub grad: Matrix,
}

impl ReluLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix) {
        self.a.copy_from(z);
        self.a
            .element_wise_operation(|x| if x > 0.0 { x } else { 0.0 });
    }

    pub fn backward(&mut self, upstream_grad: &Matrix) {
        self.grad.copy_from(&(self.a));
        self.grad
            .element_wise_operation(|x| if x > 0.0 { 1.0 } else { 0.0 });
        self.grad.multiply_matrix(upstream_grad);
    }
}

pub struct CELossLayer {
    pub in_size: usize,
    pub batch_size: usize,
    pub loss: f32,
    pub a: Matrix,
    pub grad: Matrix,
}

impl CELossLayer {
    pub fn new(in_size: usize, batch_size: usize) -> Self {
        Self {
            in_size: in_size,
            batch_size: batch_size,
            loss: 0.0,
            a: Matrix::new(batch_size, in_size, 0.0),
            grad: Matrix::new(batch_size, in_size, 0.0),
        }
    }

    pub fn forward(&mut self, z: &Matrix, y: &Matrix) {
        self.a.copy_from(z);
        self.a.subtract_column(&(self.a.max(1)));
        self.a.element_wise_operation(|x| x.exp());
        self.a.divide_column(&(self.a.sum(1)));

        self.loss = y
            .data
            .iter()
            .zip(self.a.data.iter())
            .map(|(&y_n, a_n)| if y_n > 0.0 { -a_n.ln() } else { 0.0 })
            .sum::<f32>()
            / (self.batch_size as f32);
    }

    pub fn backward(&mut self, y: &Matrix) {
        self.grad.copy_from(&(self.a));
        self.grad.subtract_matrix(y);
    }
}

pub struct Conv2dLayer {
    pub batch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub in_shape: (usize, usize),
    pub out_shape: (usize, usize),
    pub kernel_shape: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub input: Vec<Vec<Matrix>>,
    pub output: Vec<Vec<Matrix>>,
    pub weight: Vec<Vec<Matrix>>,
}

impl Conv2dLayer {
    pub fn new(
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        in_shape: (usize, usize),
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Self {
        assert!(stride.0 >= 1 && stride.1 >= 1);
        assert!(dilation.0 >= 1 && dilation.1 >= 1);
        assert!(kernel_shape.0 % 2 == 1 && kernel_shape.1 % 2 == 1);
        assert!(in_shape.0 + 2 * padding.0 >= kernel_shape.0);
        assert!(in_shape.1 + 2 * padding.1 >= kernel_shape.1);

        let out_height =
            ((in_shape.0 + 2 * padding.0 - dilation.0 * (kernel_shape.0 - 1) - 1) / stride.0) + 1;
        let out_width =
            ((in_shape.1 + 2 * padding.1 - dilation.1 * (kernel_shape.1 - 1) - 1) / stride.1) + 1;
        assert!(out_height > 0 && out_width > 0);

        let n = out_channels * kernel_shape.0 * kernel_shape.1;
        let std_dev = (1.0) / (n as f32).sqrt();
        Self {
            batch_size: batch_size,
            in_channels: in_channels,
            out_channels: out_channels,
            in_shape: in_shape,
            out_shape: (out_height, out_width),
            kernel_shape: kernel_shape,
            stride: stride,
            padding: padding,
            dilation: dilation,
            input: (0..batch_size)
                .map(|_| {
                    (0..in_channels)
                        .map(|_| Matrix::new(in_shape.0, in_shape.1, 0.0))
                        .collect()
                })
                .collect(),
            weight: (0..out_channels)
                .map(|_| {
                    (0..in_channels)
                        .map(|_| Matrix::randn(kernel_shape.0, kernel_shape.1, -std_dev, std_dev))
                        .collect()
                })
                .collect(),
            output: (0..batch_size)
                .map(|_| {
                    (0..out_channels)
                        .map(|_| Matrix::new(out_height, out_width, 0.0))
                        .collect()
                })
                .collect(),
        }
    }

    pub fn forward(&mut self, input: &Vec<Vec<Matrix>>) {
        self.input = input
            .iter()
            .map(|batch| batch.iter().map(|img| img.copy()).collect())
            .collect();
        self.output
            .iter_mut()
            .zip(self.input.iter())
            .for_each(|(output_batch, input_batch)| {
                output_batch
                    .iter_mut()
                    .zip(&self.weight)
                    .for_each(|(output_img, kernels)| {
                        let convolutions = input_batch
                            .iter()
                            .zip(kernels)
                            .map(|(input_img, kernel)| {
                                input_img.convolution_2d(
                                    kernel,
                                    self.stride,
                                    self.dilation,
                                    self.padding,
                                )
                            })
                            .collect::<Vec<Matrix>>();
                        let mut convolution = Matrix::new(self.out_shape.0, self.out_shape.1, 0.0);
                        for mat in &convolutions {
                            convolution.add_matrix(mat);
                        }
                        output_img.copy_from(&convolution);
                    });
            });
    }
}
