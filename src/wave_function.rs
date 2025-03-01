use crate::parameters;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::Xspace;
use std::fs::File;
use std::io::BufWriter;

pub struct WaveFunction<'a> {
    pub psi: Array<Complex<f64>, Ix1>,
    x: &'a Xspace,
}

impl<'a> WaveFunction<'a> {
    pub fn new(psi: Array<Complex<f64>, Ix1>, x: &'a Xspace) -> Self {
        Self { psi, x }
    }

    //Возвращает норму волновой функции
    pub fn norm(&self) -> f64 {
        f64::sqrt(self.psi.mapv(|a| (a.re.powi(2) + a.im.powi(2))).sum() * self.x.dx)
    }

    // Нормирует волновую функцию на 1
    pub fn normalization_by_1(&mut self) {
        let norm: f64 = self.norm();
        let j = Complex::I;
        self.psi *= (1. + 0. * j) / norm;
    }

    // Сохраняет волновую функцию в файл
    pub fn save_psi(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.psi.write_npy(writer)?;
        Ok(())
    }

    // Загружает волновую функцию из файла.
    pub fn init_from_file(psi_path: &str, x: &'a Xspace) -> Self {
        let reader = File::open(psi_path).unwrap();
        Self {
            psi: Array::<Complex<f64>, Ix1>::read_npy(reader).unwrap(),
            x,
        }
    }
}
