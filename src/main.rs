use crate::common::matrix::*;
use crate::common::vector::*;

mod common {
    pub mod linear_algebra;
    pub mod matrix;
    pub mod vector;
}

mod deep_learning {
    pub mod activation_layers;
    pub mod linear_layer;
    pub mod loss_layers;
}

mod machine_learning {
    pub mod regressors;

    pub mod classifiers;
}

fn main() {
    let a: Matrix<f32> = Matrix::eye(4);
    let c: Vector<f32> = Vector::new(4, 2.0);
    let b = a.insert_column(&c, 3);

    a.print();
    b.print();
}
