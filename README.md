# Rust Split Step Fourier Method 1D

**Запускать `cargo run --release` надо из каталога /src !**

`/src/arrays_saved` - директория с сохраненными массивами. 
Перед запуском кода в `/src/arrays_saved` должны лежать файлы:

- `x0.npy` (массив координатной сетки)
- `psi_initial.npy` (комплексный массив волновой функции)
- `atomic_potential.npy` (комплексный массив атомного потенциала
                        ВМЕСТЕ с поглощающим слоем!)
**Если эти массивы лежат в других каталогах, то указать полные пути к ним в `src/main.rs`**

В директорию `/src/arrays_saved/time_evol/psi_x` будут сохраняться 
временные срезы пространственной волновой функции.

Временная сетка задается в файле `/src/main.rs`

Внешнее электрическое поле задается в файле `/src/field.rs`

## Как посчитать эволюцию
 
#### Шаг 1
Кладем в каталог  `/src/arrays_saved` начальные заранее сгенерированные массивы:
- `x0.npy` (массив координатной сетки)
- `psi_initial.npy` (комплексный массив волновой функции)
- `atomic_potential.npy` (комплексный массив атомного потенциала
                        ВМЕСТЕ с поглощающим слоем!)
**Если эти массивы лежат в других каталогах, то указать полные пути к ним в `src/main.rs`**

Можно построить графики этих начальных массивов в `/src/python/plot_initial_arrays.ipynb`

#### Шаг 2
Задаем в `/src/main.rs` временную сетку. Например:
```rust
let mut t = Tspace::new(0., 0.1, 100, 50);
```
где `Tspace::new(t0: f64, dt: f64, n_steps: usize, nt: usize)`

#### Шаг 3
Задаем внешнее электрическое поле в файле `/src/field.rs` и инициализируем его в `/src/main.rs`.
Например:
`/src/main.rs`
```rust
let field1d = Field1D {
    amplitude: 0.035,
    omega: 0.04,
    x_envelop: 50.0001,
};
```

#### Шаг 4
Запускаем расчет эволюции волновой функции. Из директории `/src`:
```bash
cargo run --release
```
После этого появятся:
- `/src/arrays_saved/time_evol/t.npy` - массив временной сетки
- `/src/arrays_saved/time_evol/psi_x` - каталог с массивами временных срезов волновой функции
Эти срезы согласованы с массивом временной сетки

Можно построить графики временных срезов волновой функции, запустив `/src/python/plot_psi_x_time_slises.py`
После этого появится каталог `/src/imgs/time_evol/psi_x` в графиками временных врезов волновой фукции

