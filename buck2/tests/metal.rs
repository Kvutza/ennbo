use ennx::{AcquisitionKind, ComputeBackend, WeightAsk, WeightLeaf, WeightSearch};

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

fn ask(backend: ComputeBackend) -> (usize, f32, Vec<u8>) {
    let base = base();
    let mut search = WeightSearch::new(&base, 0.25, leaves(), 4, backend).unwrap();
    let warm = search
        .ask(
            &[17],
            WeightAsk {
                neighbors: 1,
                length: 1.0,
                ..WeightAsk::default()
            },
        )
        .unwrap();
    search.tell(warm, 0.75, true).unwrap();
    let trial = search
        .ask(
            &[19, 23, 29, 31],
            WeightAsk {
                neighbors: 2,
                length: 0.65,
                beta: 1.3,
                acquisition: AcquisitionKind::Ucb,
                seed: 41,
                ..WeightAsk::default()
            },
        )
        .unwrap();
    (trial.index, trial.score, search.row(trial).unwrap())
}

#[test]
fn metal_matches_cpu() {
    let cpu = ask(ComputeBackend::Cpu);
    let metal = ask(ComputeBackend::Metal);
    assert_eq!(metal.0, cpu.0);
    assert!((metal.1 - cpu.1).abs() <= 1.0e-5);
    assert_eq!(metal.2, cpu.2);
}

#[test]
fn agx_matches_cpu() {
    let cpu = ask(ComputeBackend::Cpu);
    let agx = ask(ComputeBackend::Agx);
    assert_eq!(agx.0, cpu.0);
    assert!((agx.1 - cpu.1).abs() <= 1.0e-5);
    assert_eq!(agx.2, cpu.2);
}
