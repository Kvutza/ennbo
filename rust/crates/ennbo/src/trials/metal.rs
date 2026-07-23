use std::ffi::c_void;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use super::{make_steps, make_tiles, Ask, Leaf, Step, Tile};

const THREADS: u64 = 256;
const SOURCE: &str = include_str!("trials.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Seed {
    low: u32,
    high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    row_bytes: u32,
    history: u32,
    candidates: u32,
    leaves: u32,
    tiles: u32,
    neighbors: u32,
    base_slot: u32,
    trial_slot: u32,
    acquisition: u32,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
}

struct Scratch {
    history_slots: Buffer,
    outcomes: Buffer,
    seeds: Buffer,
    draws: Buffer,
    scores: Buffer,
    partials: Buffer,
    choice: Buffer,
    leaves: Buffer,
    tiles: Buffer,
    candidate_capacity: usize,
}

pub(super) struct Engine {
    device: Device,
    queue: CommandQueue,
    rows: Buffer,
    row_bytes: usize,
    tile_count: usize,
    distance: ComputePipelineState,
    score: ComputePipelineState,
    pick: ComputePipelineState,
    write: ComputePipelineState,
    scratch: Scratch,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Leaf], slots: usize) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no default Metal device found")?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(|error| format!("failed to compile Metal trial kernels: {error}"))?;
        let distance = pipeline(&device, &library, "distance_trials")?;
        let score = pipeline(&device, &library, "score_trials")?;
        let pick = pipeline(&device, &library, "pick_trial")?;
        let write = pipeline(&device, &library, "write_trial")?;
        let row_bytes = base.len();
        let tiles = make_tiles(leaves);
        let rows = shared(&device, slots.saturating_mul(row_bytes), "model rows")?;
        copy_to(&rows, base);
        let scratch = Scratch {
            history_slots: shared(
                &device,
                super::MAX_HISTORY * size_of::<u32>(),
                "history slots",
            )?,
            outcomes: shared(&device, super::MAX_HISTORY * size_of::<f32>(), "outcomes")?,
            seeds: shared(&device, size_of::<Seed>(), "seeds")?,
            draws: shared(&device, size_of::<f32>(), "draws")?,
            scores: shared(&device, size_of::<f32>(), "scores")?,
            partials: shared(
                &device,
                super::MAX_HISTORY
                    .saturating_mul(tiles.len())
                    .saturating_mul(size_of::<f32>()),
                "partial distances",
            )?,
            choice: shared(&device, size_of::<u32>(), "choice")?,
            leaves: shared(
                &device,
                leaves.len().saturating_mul(size_of::<Step>()),
                "leaves",
            )?,
            tiles: shared(
                &device,
                tiles.len().saturating_mul(size_of::<Tile>()),
                "tiles",
            )?,
            candidate_capacity: 1,
        };
        copy_to(&scratch.tiles, &tiles);
        Ok(Self {
            queue: device.new_command_queue(),
            device,
            rows,
            row_bytes,
            tile_count: tiles.len(),
            distance,
            score,
            pick,
            write,
            scratch,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        self.ensure_candidates(seeds.len())?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let steps = make_steps(leaves, config.length);
        copy_to(&self.scratch.history_slots, &history_slots);
        copy_to(&self.scratch.outcomes, &outcomes);
        copy_to(&self.scratch.seeds, &seeds);
        copy_to(&self.scratch.draws, &draws);
        copy_to(&self.scratch.leaves, &steps);

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            history: to_u32(history.len(), "history length")?,
            candidates: to_u32(seeds.len(), "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            acquisition: crate::weights::acquisition_code(config.acquisition),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
        };

        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        encoder.set_buffer(5, Some(&self.scratch.partials), 0);
        set_params(encoder, 6, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (seeds.len().div_ceil(2) * params.tiles as usize) as u64,
                height: 1,
                depth: 1,
            },
            group(THREADS),
        );

        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(encoder, 4, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: seeds.len() as u64,
                height: 1,
                depth: 1,
            },
            group(THREADS),
        );

        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        set_params(encoder, 2, &params);
        encoder.dispatch_thread_groups(group(1), group(1));

        encoder.set_compute_pipeline_state(&self.write);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.choice), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        set_params(encoder, 5, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: params.tiles as u64,
                height: 1,
                depth: 1,
            },
            group(THREADS),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        let index = read_one::<u32>(&self.scratch.choice) as usize;
        let scores = read_slice::<f32>(&self.scratch.scores, seeds.len());
        Ok((index, scores[index]))
    }

    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Vec<u8> {
        let start = slot * row_bytes;
        unsafe {
            std::slice::from_raw_parts(self.rows.contents().cast::<u8>().add(start), row_bytes)
                .to_vec()
        }
    }

    fn ensure_candidates(&mut self, count: usize) -> Result<(), String> {
        if count <= self.scratch.candidate_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.seeds = shared(
            &self.device,
            capacity.saturating_mul(size_of::<Seed>()),
            "seeds",
        )?;
        self.scratch.draws = shared(
            &self.device,
            capacity.saturating_mul(size_of::<f32>()),
            "draws",
        )?;
        self.scratch.scores = shared(
            &self.device,
            capacity.saturating_mul(size_of::<f32>()),
            "scores",
        )?;
        let partial_count = capacity
            .checked_mul(super::MAX_HISTORY)
            .and_then(|value| value.checked_mul(self.tile_count))
            .ok_or("partial distance buffer size overflow")?;
        self.scratch.partials = shared(
            &self.device,
            partial_count.saturating_mul(size_of::<f32>()),
            "partial distances",
        )?;
        self.scratch.candidate_capacity = capacity;
        Ok(())
    }
}

fn pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|error| format!("missing Metal trial kernel {name}: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("failed to create Metal trial pipeline {name}: {error}"))
}

fn shared(device: &Device, bytes: usize, name: &str) -> Result<Buffer, String> {
    if bytes == 0 {
        return Err(format!("{name} buffer cannot be empty"));
    }
    Ok(device.new_buffer(
        bytes as u64,
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
    ))
}

fn copy_to<T>(buffer: &Buffer, values: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
            std::mem::size_of_val(values),
        );
    }
}

fn read_one<T: Copy>(buffer: &Buffer) -> T {
    unsafe { *buffer.contents().cast::<T>() }
}

fn read_slice<T: Copy>(buffer: &Buffer, len: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), len) }
}

fn set_params(encoder: &metal::ComputeCommandEncoderRef, index: u64, params: &Params) {
    encoder.set_bytes(
        index,
        size_of::<Params>() as u64,
        (params as *const Params).cast::<c_void>(),
    );
}

fn group(width: u64) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} exceeds u32 range"))
}

const fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
}
