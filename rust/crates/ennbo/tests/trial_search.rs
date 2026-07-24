use ennbo::{AcquisitionKind, ComputeBackend, WeightAsk, WeightLeaf, WeightSearch};

fn leaves() -> Vec<WeightLeaf> {
    vec![
        WeightLeaf::new(0, 257, 4, 0.25, 1.0, 0.75).unwrap(),
        WeightLeaf::new(257, 263, 8, 0.5, 0.5, 1.0).unwrap(),
    ]
}

fn base() -> Vec<u8> {
    let row_bytes = 257usize.div_ceil(2) + 263;
    (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

fn ask(
    backend: ComputeBackend,
    acquisition: AcquisitionKind,
) -> Result<(usize, f32, Vec<u8>), String> {
    let mut search = WeightSearch::new(&base(), 0.25, leaves(), 4, backend)?;
    let warm = search.ask(
        &[17],
        WeightAsk {
            neighbors: 1,
            length: 1.0,
            ..WeightAsk::default()
        },
    )?;
    search.tell(warm, 0.75, true)?;
    let trial = search.ask(
        &[19, 23, 29, 31],
        WeightAsk {
            neighbors: 2,
            length: 0.65,
            beta: 1.3,
            acquisition,
            seed: 41,
            ..WeightAsk::default()
        },
    )?;
    Ok((trial.index, trial.score, search.row(trial)?))
}

#[test]
fn cpu_trial_search_is_repeatable() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let left = ask(ComputeBackend::Cpu, acquisition).unwrap();
        let right = ask(ComputeBackend::Cpu, acquisition).unwrap();
        assert_eq!(left, right);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_matches_cpu_trial_bytes() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let cpu = ask(ComputeBackend::Cpu, acquisition).unwrap();
        let metal = ask(ComputeBackend::Metal, acquisition).unwrap();
        assert_eq!(metal.0, cpu.0);
        assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
        assert_eq!(metal.2, cpu.2);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_matches_cpu_when_a_device_exists() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let cpu = ask(ComputeBackend::Cpu, acquisition).unwrap();
        let opencl = match ask(ComputeBackend::OpenCl, acquisition) {
            Ok(value) => value,
            Err(error) if error.contains("no OpenCL GPU or CPU device") => return,
            Err(error) => panic!("{error}"),
        };
        assert_eq!(opencl.0, cpu.0);
        assert!((opencl.1 - cpu.1).abs() <= 1.0e-5);
        assert_eq!(opencl.2, cpu.2);
    }
}
