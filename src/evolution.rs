use crate::field;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field1D;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use potentials::AtomicPotential;
use rayon::prelude::*;
use std::f64::consts::PI;
use wave_function::WaveFunction;

pub fn time_step_evol(
    fft: &mut FftMaker1d,
    psi: &mut WaveFunction,
    field: &Field1D,
    u: &AtomicPotential,
    x: &Xspace,
    p: &Pspace,
    t: &mut Tspace,
) {
    modify_psi(psi, x, p);
    x_evol_half(psi, u, t, field, x);

    for _i in 0..t.n_steps - 1 {
        fft.do_fft(psi);
        // Можно оптимизировать p_evol
        p_evol(psi, p, t.dt);
        fft.do_ifft(psi);
        x_evol(psi, u, t, field, x);
        t.current += t.dt;
    }

    fft.do_fft(psi);
    p_evol(psi, p, t.dt);
    fft.do_ifft(psi);
    x_evol_half(psi, u, t, field, x);
    demodify_psi(psi, x, p);
    t.current += t.dt;
}

pub fn x_evol_half(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field: &Field1D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let electric_field_potential = field.potential_as_array(t.current, x);

    multizip((
        psi.psi.iter_mut(),
        atomic_potential.potential.iter(),
        electric_field_potential.iter(),
    ))
    .par_bridge()
    .for_each(|(psi_elem, atomic_potential_elem, field_potential)| {
        *psi_elem *= (-j * 0.5 * t.dt * (atomic_potential_elem - field_potential)).exp();
    });
}

pub fn x_evol(
    psi: &mut WaveFunction,
    atomic_potential: &AtomicPotential,
    t: &Tspace,
    field: &Field1D,
    x: &Xspace,
) {
    // эволюция в координатном пространстве на половину временного шага
    let j = Complex::I;

    let electric_field_potential = field.potential_as_array(t.current, x);

    multizip((
        psi.psi.iter_mut(),
        atomic_potential.potential.iter(),
        electric_field_potential.iter(),
    ))
    .par_bridge()
    .for_each(|(psi_elem, atomic_potential_elem, field_potential)| {
        *psi_elem *= (-j * t.dt * (atomic_potential_elem - field_potential)).exp();
    });
}

pub fn p_evol(psi: &mut WaveFunction, p: &Pspace, dt: f64) {
    // эволюция в импульсном пространстве
    let j = Complex::I;
    psi.psi
        .iter_mut()
        .zip(p.grid.iter())
        .par_bridge()
        .for_each(|(psi_elem, p_elem)| {
            *psi_elem *= (-j * 0.5 * dt * p_elem.powi(2)).exp();
        });
}

pub fn demodify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // демодифицирует "psi для DFT" обратно в psi
    let j = Complex::I;
    psi.psi
        .iter_mut()
        .zip(x.grid.iter())
        .par_bridge()
        .for_each(|(psi_elem, x_elem)| {
            *psi_elem *= (2. * PI).sqrt() / x.dx * (j * p.p0 * x_elem).exp();
        });
}

pub fn modify_psi(psi: &mut WaveFunction, x: &Xspace, p: &Pspace) {
    // модифицирует psi для DFT (в нашем сучае FFT)
    let j = Complex::I;
    psi.psi
        .iter_mut()
        .zip(x.grid.iter())
        .par_bridge()
        .for_each(|(psi_elem, x_elem)| {
            *psi_elem *= x.dx / (2. * PI).sqrt() * (-j * p.p0 * x_elem).exp();
        });
}

pub struct FftMaker1d {
    pub handler: FftHandler<f64>,
    pub psi_temp: Array<Complex<f64>, Ix1>,
}

impl FftMaker1d {
    pub fn new(n: &usize) -> Self {
        Self {
            handler: FftHandler::new(*n),
            psi_temp: Array::zeros(*n),
        }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler, 0);
        psi.psi = self.psi_temp.clone();
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler, 0);
        psi.psi = self.psi_temp.clone();
    }
}
