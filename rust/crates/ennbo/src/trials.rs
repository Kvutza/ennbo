use std::collections::VecDeque;

use crate::weights::{AcquisitionKind, ComputeBackend};

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

const MAX_HISTORY: usize = 16;
const TILE_ELEMENTS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leaf {
    pub offset: usize,
    pub length: usize,
    pub bits: u8,
    pub scale: f32,
    pub weight: f32,
    pub radius: f32,
}

impl Leaf {
    pub fn new(
        offset: usize,
        length: usize,
        bits: u8,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        if length == 0 {
            return Err("leaf length must be positive".to_string());
        }
        if bits != 4 && bits != 8 {
            return Err(format!("leaf bits must be 4 or 8, got {bits}"));
        }
        for (name, value) in [("scale", scale), ("weight", weight), ("radius", radius)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        Ok(Self {
            offset,
            length,
            bits,
            scale,
            weight,
            radius,
        })
    }

    fn bytes(self) -> usize {
        match self.bits {
            4 => self.length.div_ceil(2),
            8 => self.length,
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ask {
    pub length: f32,
    pub neighbors: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub acquisition: AcquisitionKind,
    pub seed: u64,
}

impl Default for Ask {
    fn default() -> Self {
        Self {
            length: 0.8,
            neighbors: 10,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            beta: 1.0,
            acquisition: AcquisitionKind::Ucb,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trial {
    id: u64,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
struct Record {
    slot: usize,
    value: f32,
}

#[derive(Debug, Clone, Copy)]
struct Pending {
    id: u64,
    slot: usize,
}

enum Engine {
    Cpu(Cpu),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Engine),
    #[cfg(feature = "opencl")]
    OpenCl(opencl::Engine),
}

pub struct Search {
    leaves: Vec<Leaf>,
    row_bytes: usize,
    capacity: usize,
    slots: usize,
    base: usize,
    history: VecDeque<Record>,
    pending: Option<Pending>,
    next_id: u64,
    engine: Engine,
}

impl Search {
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Leaf>,
        capacity: usize,
        backend: ComputeBackend,
    ) -> Result<Self, String> {
        if !base_value.is_finite() {
            return Err("base value must be finite".to_string());
        }
        if capacity == 0 || capacity > MAX_HISTORY {
            return Err(format!("history capacity must be in 1..={MAX_HISTORY}"));
        }
        let row_bytes = check_layout(&leaves)?;
        if base.len() != row_bytes {
            return Err(format!(
                "base row has {} bytes, expected {row_bytes}",
                base.len()
            ));
        }
        let slots = capacity + 2;
        let engine = Engine::new(base, &leaves, slots, backend)?;
        Ok(Self {
            leaves,
            row_bytes,
            capacity,
            slots,
            base: 0,
            history: VecDeque::from([Record {
                slot: 0,
                value: base_value,
            }]),
            pending: None,
            next_id: 0,
            engine,
        })
    }

    pub fn ask(&mut self, seeds: &[u64], config: Ask) -> Result<Trial, String> {
        if self.pending.is_some() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        check_ask(seeds, self.history.len(), config)?;
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history: Vec<(usize, f32)> = self
            .history
            .iter()
            .map(|record| (record.slot, record.value))
            .collect();
        let (index, score) =
            self.engine
                .ask(self.base, &history, slot, seeds, &self.leaves, config)?;
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.pending = Some(Pending { id, slot });
        Ok(Trial {
            id,
            index,
            seed: seeds[index],
            score,
        })
    }

    pub fn row(&self, trial: Trial) -> Result<Vec<u8>, String> {
        let pending = self.pending_for(trial)?;
        self.engine.read(pending.slot, self.row_bytes)
    }

    pub fn tell(&mut self, trial: Trial, value: f32, accept: bool) -> Result<(), String> {
        if !value.is_finite() {
            return Err("trial value must be finite".to_string());
        }
        let pending = self.pending_for(trial)?;
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(Record {
            slot: pending.slot,
            value,
        });
        if accept {
            self.base = pending.slot;
        }
        self.pending = None;
        Ok(())
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    fn pending_for(&self, trial: Trial) -> Result<Pending, String> {
        match self.pending {
            Some(pending) if pending.id == trial.id => Ok(pending),
            Some(_) => Err("trial does not match the pending ask".to_string()),
            None => Err("there is no pending trial".to_string()),
        }
    }

    fn free_slot(&self) -> Option<usize> {
        (0..self.slots).find(|slot| {
            *slot != self.base
                && self.history.iter().all(|record| record.slot != *slot)
                && self.pending.map(|pending| pending.slot) != Some(*slot)
        })
    }
}

impl Engine {
    #[allow(unused_variables)]
    fn new(
        base: &[u8],
        leaves: &[Leaf],
        slots: usize,
        backend: ComputeBackend,
    ) -> Result<Self, String> {
        match backend {
            ComputeBackend::Cpu => Ok(Self::Cpu(Cpu::new(base, slots))),
            ComputeBackend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Metal(metal::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err("Metal trial search is not available in this build".to_string())
                }
            }
            ComputeBackend::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Self::OpenCl(opencl::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err("OpenCL trial search is not available in this build".to_string())
                }
            }
            ComputeBackend::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    return Ok(Self::Metal(metal::Engine::new(base, leaves, slots)?));
                }
                #[cfg(all(feature = "opencl", not(all(target_os = "macos", feature = "metal"))))]
                {
                    return Ok(Self::OpenCl(opencl::Engine::new(base, leaves, slots)?));
                }
                #[allow(unreachable_code)]
                Ok(Self::Cpu(Cpu::new(base, slots)))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        match self {
            Self::Cpu(engine) => engine.ask(base, history, trial, seeds, leaves, config),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => engine.ask(base, history, trial, seeds, leaves, config),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.ask(base, history, trial, seeds, leaves, config),
        }
    }

    #[allow(unused_variables)]
    fn read(&self, slot: usize, row_bytes: usize) -> Result<Vec<u8>, String> {
        match self {
            Self::Cpu(engine) => Ok(engine.read(slot).to_vec()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => Ok(engine.read(slot, row_bytes)),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.read(slot, row_bytes),
        }
    }
}

struct Cpu {
    rows: Vec<u8>,
    row_bytes: usize,
}

impl Cpu {
    fn new(base: &[u8], slots: usize) -> Self {
        let mut rows = vec![0; slots * base.len()];
        rows[..base.len()].copy_from_slice(base);
        Self {
            rows,
            row_bytes: base.len(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        let steps = make_steps(leaves, config.length);
        let base = self.read(base_slot).to_vec();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let mut best_index = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (index, &seed) in seeds.iter().enumerate() {
            let mut nearest = vec![(f32::INFINITY, 0usize); config.neighbors];
            for (observation_index, &(slot, _)) in history.iter().enumerate() {
                let distance = trial_distance(&base, self.read(slot), leaves, &steps, seed);
                insert_neighbor(&mut nearest, distance, observation_index);
            }
            let score = score(&nearest, history, draws[index], config);
            if score > best_score || (score == best_score && index < best_index) {
                best_index = index;
                best_score = score;
            }
        }
        let row = materialize(&base, leaves, &steps, seeds[best_index]);
        self.read_mut(trial_slot).copy_from_slice(&row);
        Ok((best_index, best_score))
    }

    fn read(&self, slot: usize) -> &[u8] {
        &self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }

    fn read_mut(&mut self, slot: usize) -> &mut [u8] {
        &mut self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Step {
    pub byte_offset: u32,
    pub element_offset: u32,
    pub length: u32,
    pub bits: u32,
    pub scale: f32,
    pub weight: f32,
    pub whole: u32,
    pub threshold: u32,
}

pub(crate) fn make_steps(leaves: &[Leaf], length: f32) -> Vec<Step> {
    let mut byte_offset = 0usize;
    leaves
        .iter()
        .map(|leaf| {
            let max_code = (1u32 << leaf.bits) - 1;
            let amplitude = (length * leaf.radius / leaf.scale).clamp(0.0, max_code as f32);
            let whole = amplitude.floor() as u32;
            let threshold = if whole == max_code {
                0
            } else {
                ((amplitude - whole as f32) * (u32::MAX as f32)) as u32
            };
            let step = Step {
                byte_offset: byte_offset as u32,
                element_offset: leaf.offset as u32,
                length: leaf.length as u32,
                bits: u32::from(leaf.bits),
                scale: leaf.scale,
                weight: leaf.weight,
                whole,
                threshold,
            };
            byte_offset += leaf.bytes();
            step
        })
        .collect()
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

pub(crate) fn make_tiles(leaves: &[Leaf]) -> Vec<Tile> {
    let mut tiles = Vec::new();
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let mut start = 0usize;
        while start < leaf.length {
            let length = (leaf.length - start).min(TILE_ELEMENTS);
            tiles.push(Tile {
                leaf: leaf_index as u32,
                start: start as u32,
                length: length as u32,
                pad: 0,
            });
            start += length;
        }
    }
    tiles
}

fn check_layout(leaves: &[Leaf]) -> Result<usize, String> {
    if leaves.is_empty() {
        return Err("at least one leaf is required".to_string());
    }
    let mut offset = 0usize;
    let mut row_bytes = 0usize;
    for leaf in leaves {
        if leaf.offset != offset {
            return Err(format!(
                "leaf offset {} does not continue parameter offset {offset}",
                leaf.offset
            ));
        }
        offset = offset
            .checked_add(leaf.length)
            .ok_or("parameter count overflow")?;
        row_bytes = row_bytes
            .checked_add(leaf.bytes())
            .ok_or("row byte count overflow")?;
    }
    if offset > u32::MAX as usize || row_bytes > u32::MAX as usize {
        return Err("trial search currently supports at most u32::MAX parameters and bytes".into());
    }
    Ok(row_bytes)
}

fn check_ask(seeds: &[u64], observations: usize, config: Ask) -> Result<(), String> {
    if seeds.is_empty() {
        return Err("ask requires at least one seed".to_string());
    }
    if config.neighbors == 0 || config.neighbors > observations {
        return Err(format!(
            "neighbor count must be between one and {observations}"
        ));
    }
    for (name, value) in [
        ("length", config.length),
        ("epistemic_scale", config.epistemic_scale),
        ("aleatoric_scale", config.aleatoric_scale),
        ("y_scale", config.y_scale),
        ("beta", config.beta),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and nonnegative"));
        }
    }
    Ok(())
}

pub(crate) fn perturb(code: u32, seed: u64, element: u32, step: Step) -> u32 {
    let random = hash(seed, element);
    let sign = random & 1;
    let extra = u32::from((random >> 1) < (step.threshold >> 1));
    let amount = step.whole + extra;
    if amount == 0 {
        return code;
    }
    let max_code = (1u32 << step.bits) - 1;
    if sign == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

fn hash(seed: u64, element: u32) -> u32 {
    let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= (seed >> 32) as u32;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

fn materialize(base: &[u8], leaves: &[Leaf], steps: &[Step], seed: u64) -> Vec<u8> {
    let mut row = vec![0u8; base.len()];
    for (&leaf, &step) in leaves.iter().zip(steps) {
        match leaf.bits {
            4 => {
                for element in 0..leaf.length {
                    let byte = step.byte_offset as usize + element / 2;
                    let shift = (element & 1) * 4;
                    let code = u32::from((base[byte] >> shift) & 0x0f);
                    let value = perturb(code, seed, leaf.offset as u32 + element as u32, step);
                    row[byte] |= (value as u8) << shift;
                }
            }
            8 => {
                for element in 0..leaf.length {
                    let byte = step.byte_offset as usize + element;
                    let code = u32::from(base[byte]);
                    row[byte] =
                        perturb(code, seed, leaf.offset as u32 + element as u32, step) as u8;
                }
            }
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
    row
}

fn trial_distance(
    base: &[u8],
    observation: &[u8],
    leaves: &[Leaf],
    steps: &[Step],
    seed: u64,
) -> f32 {
    let mut distance = 0.0;
    for (&leaf, &step) in leaves.iter().zip(steps) {
        for element in 0..leaf.length {
            let byte =
                step.byte_offset as usize + if leaf.bits == 4 { element / 2 } else { element };
            let shift = if leaf.bits == 4 { (element & 1) * 4 } else { 0 };
            let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
            let code = u32::from((base[byte] >> shift) & mask);
            let candidate = perturb(code, seed, leaf.offset as u32 + element as u32, step) as f32;
            let observed = f32::from((observation[byte] >> shift) & mask);
            let delta = (candidate - observed) * leaf.scale;
            distance = delta.mul_add(delta * leaf.weight, distance);
        }
    }
    distance
}

fn insert_neighbor(nearest: &mut [(f32, usize)], distance: f32, index: usize) {
    let Some(position) = nearest.iter().position(|&(other_distance, other_index)| {
        distance < other_distance || (distance == other_distance && index < other_index)
    }) else {
        return;
    };
    for i in (position + 1..nearest.len()).rev() {
        nearest[i] = nearest[i - 1];
    }
    nearest[position] = (distance, index);
}

fn score(nearest: &[(f32, usize)], history: &[(usize, f32)], draw: f32, config: Ask) -> f32 {
    let mut weight_sum = 0.0;
    let mut weighted_value = 0.0;
    for &(distance, index) in nearest {
        let variance = 1.0e-9 + config.epistemic_scale * distance + config.aleatoric_scale;
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * history[index].1;
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * config.y_scale;
    match config.acquisition {
        AcquisitionKind::Ucb => mean + config.beta * se,
        AcquisitionKind::Thompson => mean + se * draw,
        AcquisitionKind::Pareto => mean + se,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves() -> Vec<Leaf> {
        vec![
            Leaf::new(0, 5, 4, 0.25, 1.0, 0.75).unwrap(),
            Leaf::new(5, 4, 8, 0.5, 0.5, 1.0).unwrap(),
        ]
    }

    #[test]
    fn cpu_search_is_deterministic_and_updates_every_leaf() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut left = Search::new(&base, 1.0, leaves(), 4, ComputeBackend::Cpu).unwrap();
        let mut right = Search::new(&base, 1.0, leaves(), 4, ComputeBackend::Cpu).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 1.0,
            ..Ask::default()
        };
        let a = left.ask(&[7, 11, 13], config).unwrap();
        let b = right.ask(&[7, 11, 13], config).unwrap();
        assert_eq!(a, b);
        let row = left.row(a).unwrap();
        assert_eq!(row, right.row(b).unwrap());
        assert_ne!(&row[..3], &base[..3]);
        assert_ne!(&row[3..], &base[3..]);
    }

    #[test]
    fn accepted_trial_becomes_the_next_center() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 1.0,
            ..Ask::default()
        };
        let first = search.ask(&[5], config).unwrap();
        let first_row = search.row(first).unwrap();
        search.tell(first, 1.0, true).unwrap();
        let second = search.ask(&[9], config).unwrap();
        let second_row = search.row(second).unwrap();
        assert_ne!(first_row, second_row);
        assert_eq!(search.history_len(), 2);
    }

    #[test]
    fn rejected_trial_does_not_replace_the_center() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        let mut control = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 1.0,
            ..Ask::default()
        };
        let rejected = search.ask(&[5], config).unwrap();
        search.tell(rejected, -1.0, false).unwrap();
        let next = search.ask(&[5], config).unwrap();
        let expected = control.ask(&[5], config).unwrap();
        assert_eq!(search.row(next).unwrap(), control.row(expected).unwrap());
    }
}
