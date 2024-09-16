extern crate fstrings;

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;

pub struct Point2e1D {
    pub x1: f64,
    pub x2: f64,
}

#[derive(Debug, Clone)]
pub struct Tspace {
    // параметры временной сетки
    pub t0: f64,
    pub dt: f64,
    pub n_steps: usize,
    pub nt: usize,
    pub current: f64,
    pub grid: Array<f64, Ix1>,
}

impl Tspace {
    pub fn new(t0: f64, dt: f64, n_steps: usize, nt: usize) -> Self {
        Self {
            t0,
            dt,
            n_steps,
            nt,
            current: t0,
            // костыль
            grid: Array::linspace(t0, t0 + dt * n_steps as f64 * (nt - 1) as f64, nt),
        }
    }

    pub fn t_step(&self) -> f64 {
        // возвращает временной шаг срезов волновой функции
        self.dt * self.n_steps as f64
    }

    pub fn last(&self) -> f64 {
        // возвращает последний элемент сетки временных срезов
        self.t0 + self.t_step() * (self.nt - 1) as f64
    }

    pub fn get_grid(&self) -> Array<f64, Ix1> {
        // возвращает временную сетку срезов
        Array::linspace(self.t0, self.last(), self.nt)
    }

    // Сохраняет временную сетку в файл
    pub fn save_grid(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.grid.write_npy(writer)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Xspace {
    // размерность пространства
    pub dim: usize,
    // параметры координатной сетки
    pub x0: f64,
    pub dx: f64,
    pub n: usize,
    // сетка
    pub grid: Array<f64, Ix1>,
}

impl Xspace {
    pub fn new(x0: f64, dx: f64, n: usize) -> Self {
        Self {
            dim: 1,
            x0: x0.clone(), // костыль!
            dx: dx.clone(),
            n: n.clone(),
            grid: Array::linspace(x0, x0 + dx * (n - 1) as f64, n),
        }
    }

    pub fn load(dir_path: &str) -> Self {
        // Загружает массивы координат из файлов.
        //
        // dir_path - путь к директории с координатными массивами. Массивы
        // в этой директории должны называться: x0.png, x1.png, x2.png и так
        // далее в зависимости от размерности пространства.
        // Всего таких массивов в этой директории dir_path должо быть dim штук.
        //
        // dim - размерность пространства.

        let x_path = String::from(dir_path) + f!("/x.npy").as_str();
        let reader = File::open(x_path).unwrap();
        let x = Array1::<f64>::read_npy(reader).unwrap();
        let x0 = x[[0]];
        let dx = (x[[1]] - x[[0]]);
        let n = x.len();

        Self {
            dim: 1,
            x0: x0,
            dx: dx,
            n: n,
            grid: x,
        }
    }

    pub fn save(&self, dir_path: &str) -> Result<(), WriteNpyError> {
        // Сохраняет массивы координат в файлы
        //
        // dir_path - путь к папке, в которую будут сохранены массивы.
        //
        // Массивы сохраняются с названиями x0.png, x1.png, x2.png и так
        // далее в зависимости от размерности пространства.

        let x_path = String::from(dir_path) + f!("/x.npy").as_str();
        let writer = BufWriter::new(File::create(x_path)?);
        self.grid.write_npy(writer)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Pspace {
    // размерность пространства
    pub dim: usize,
    // параметры импульсной сетки
    pub p0: f64,
    pub dp: f64,
    pub n: usize,
    // сетка
    pub grid: Array<f64, Ix1>,
}

impl Pspace {
    pub fn init(x: &Xspace) -> Self {
        let p0 = -PI / x.dx;
        let dp = 2. * PI / (x.n as f64 * x.dx);
        Self {
            dim: x.dim.clone(),
            p0: p0.clone(),
            dp: dp.clone(),
            n: x.n.clone(),
            grid: Array::linspace(p0, p0 + dp * (x.n - 1) as f64, x.n),
        }
    }

    pub fn save(&self, dir_path: &str) -> Result<(), WriteNpyError> {
        // Сохраняет массивы импульсов в файлы
        //
        // dir_path - путь к папке, в которую будут сохранены массивы.
        //
        // Массивы сохраняются с названиями p0.png, p1.png, p2.png и так
        // далее в зависимости от размерности пространства.

        let p_path = String::from(dir_path) + f!("p.npy").as_str();
        let writer = BufWriter::new(File::create(p_path)?);
        self.grid.write_npy(writer)?;
        Ok(())
    }
}
