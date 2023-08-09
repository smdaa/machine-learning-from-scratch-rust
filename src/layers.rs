use crate::matrix::*;

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
                        .map(|_| Matrix::rand(kernel_shape.0, kernel_shape.1, -std_dev, std_dev))
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
