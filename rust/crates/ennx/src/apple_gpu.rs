//! Shared Apple GPU runtime used by ENNX Metal backends.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use metal::{
    BinaryArchiveDescriptor, Buffer, CommandQueue, CompileOptions, ComputePipelineDescriptor,
    ComputePipelineState, Device, MTLResourceOptions, URL,
};

static RUNTIME: OnceLock<Result<Arc<Runtime>, String>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    G13,
    G14,
    G15,
    G16,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub target: Target,
}

pub fn device_info() -> Result<DeviceInfo, String> {
    Runtime::shared().map(|runtime| runtime.info().clone())
}

pub(crate) struct Runtime {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    info: DeviceInfo,
    pipelines: Mutex<HashMap<(u64, String), ComputePipelineState>>,
}

impl Runtime {
    pub(crate) fn shared() -> Result<Arc<Self>, String> {
        RUNTIME
            .get_or_init(|| Self::new().map(Arc::new))
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no default Metal device found")?;
        let name = device.name().to_string();
        let info = DeviceInfo {
            target: target_from_name(&name),
            name,
        };
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            info,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub(crate) fn pipeline(
        &self,
        source: &str,
        label: &str,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let key = (source_hash(source), name.to_string());
        if let Some(pipeline) = self
            .pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .get(&key)
        {
            return Ok(pipeline.to_owned());
        }
        let library = self
            .device
            .new_library_with_source(source, &CompileOptions::new())
            .map_err(|error| format!("{label} Metal compile: {error}"))?;
        let function = library
            .get_function(name, None)
            .map_err(|error| format!("missing Metal kernel {name}: {error}"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("Metal pipeline {name}: {error}"))?;
        self.pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .insert(key, pipeline.to_owned());
        Ok(pipeline)
    }

    pub(crate) fn agx_pipeline(
        &self,
        source: &str,
        label: &str,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let cache_name = format!("agx:{name}");
        let key = (source_hash(source), cache_name.clone());
        if let Some(pipeline) = self
            .pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .get(&key)
        {
            return Ok(pipeline.to_owned());
        }

        let library = self
            .device
            .new_library_with_source(source, &CompileOptions::new())
            .map_err(|error| format!("{label} Metal compile: {error}"))?;
        let function = library
            .get_function(name, None)
            .map_err(|error| format!("missing Metal kernel {name}: {error}"))?;
        let pipeline_desc = ComputePipelineDescriptor::new();
        pipeline_desc.set_label(&cache_name);
        pipeline_desc.set_compute_function(Some(&function));

        let path = archive_path(&self.info, source, name)?;
        // metal-rs wraps the autoreleased `+[NSURL URLWithString:]` result as
        // owned. Let the Objective-C autorelease pool release it exactly once.
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            path.display()
        )));
        let archive_desc = BinaryArchiveDescriptor::new();
        let exists = path.exists();
        if exists {
            require_agx_slice(&path)?;
            archive_desc.set_url(&url);
        }
        let archive = self
            .device
            .new_binary_archive_with_descriptor(&archive_desc)
            .map_err(|error| format!("open AGX archive {}: {error}", path.display()))?;
        pipeline_desc.set_binary_archives(&[&archive]);
        if !exists {
            archive
                .add_compute_pipeline_functions_with_descriptor(&pipeline_desc)
                .map_err(|error| format!("compile AGX archive: {error}"))?;
            archive
                .serialize_to_url(&url)
                .map_err(|error| format!("write AGX archive {}: {error}", path.display()))?;
            require_agx_slice(&path)?;
        }

        let pipeline = self
            .device
            .new_compute_pipeline_state(&pipeline_desc)
            .map_err(|error| format!("AGX archive miss for {name}: {error}"))?;
        self.pipelines
            .lock()
            .map_err(|_| "Apple GPU pipeline cache poisoned")?
            .insert(key, pipeline.to_owned());
        Ok(pipeline)
    }

    pub(crate) fn buffer<T>(&self, elements: usize) -> Buffer {
        self.device.new_buffer(
            (elements.max(1) * size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        )
    }

    pub(crate) fn buffer_with<T>(&self, values: &[T]) -> Buffer {
        if values.is_empty() {
            return self.buffer::<T>(1);
        }
        self.device.new_buffer_with_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        )
    }
}

fn source_hash(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

fn archive_path(info: &DeviceInfo, source: &str, name: &str) -> Result<std::path::PathBuf, String> {
    let device = source_hash(&info.name);
    let kernel = source_hash(&format!("{}:{name}", source_hash(source)));
    let directory = std::env::temp_dir()
        .join("ennx-agx")
        .join(format!("{device:016x}"));
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("create AGX cache {}: {error}", directory.display()))?;
    Ok(directory.join(format!("{kernel:016x}.metalarc")))
}

fn require_agx_slice(path: &std::path::Path) -> Result<(), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read AGX archive {}: {error}", path.display()))?;
    if has_agx_slice(&bytes) {
        Ok(())
    } else {
        Err(format!(
            "binary archive {} contains no applegpu slice",
            path.display()
        ))
    }
}

fn has_agx_slice(bytes: &[u8]) -> bool {
    const FAT_MAGIC: u32 = 0xcafe_babe;
    const FAT_MAGIC_64: u32 = 0xcafe_babf;
    const METAL_FAT_MAGIC: u32 = 0xcbfe_babe;
    const CPU_TYPE_APPLEGPU: u32 = 0x0100_0013;
    let word = |offset: usize| {
        bytes
            .get(offset..offset + 4)
            .map(|value| u32::from_be_bytes(value.try_into().expect("four bytes")))
    };
    let Some(magic) = word(0) else {
        return false;
    };
    let stride = match magic {
        FAT_MAGIC | METAL_FAT_MAGIC => 20,
        FAT_MAGIC_64 => 32,
        _ => return false,
    };
    let Some(count) = word(4) else {
        return false;
    };
    (0..count as usize).any(|index| word(8 + index * stride) == Some(CPU_TYPE_APPLEGPU))
}

fn target_from_name(name: &str) -> Target {
    let compact: String = name
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact.starts_with("AppleM1") {
        Target::G13
    } else if compact.starts_with("AppleM2") {
        Target::G14
    } else if compact.starts_with("AppleM3") {
        Target::G15
    } else if compact.starts_with("AppleM4") {
        Target::G16
    } else {
        Target::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::{device_info, has_agx_slice, source_hash, target_from_name, Runtime, Target};

    #[test]
    fn maps_m_series_names_to_native_generations() {
        assert_eq!(target_from_name("Apple M1 Max"), Target::G13);
        assert_eq!(target_from_name("Apple M2"), Target::G14);
        assert_eq!(target_from_name("Apple M3 Pro"), Target::G15);
        assert_eq!(target_from_name("Apple M4"), Target::G16);
        assert_eq!(target_from_name("Future GPU"), Target::Unknown);
    }

    #[test]
    fn recognizes_metal_native_archive_slice() {
        let mut archive = vec![0xcb, 0xfe, 0xba, 0xbe, 0, 0, 0, 1];
        archive.extend_from_slice(&0x0100_0013u32.to_be_bytes());
        archive.extend_from_slice(&[0; 16]);
        assert!(has_agx_slice(&archive));
        archive[11] = 0x17;
        assert!(!has_agx_slice(&archive));
    }

    #[test]
    fn shared_runtime_caches_pipelines_and_builds_buffers() {
        let runtime = Runtime::shared().unwrap();
        assert_eq!(device_info().unwrap(), runtime.info().clone());
        let source = "kernel void copy_one(device uint *x [[buffer(0)]]) { x[0] = x[0]; }";
        let first = runtime.pipeline(source, "test", "copy_one").unwrap();
        let second = runtime.pipeline(source, "test", "copy_one").unwrap();
        assert_eq!(
            first.thread_execution_width(),
            second.thread_execution_width()
        );
        let native = runtime.agx_pipeline(source, "test", "copy_one").unwrap();
        assert_eq!(
            native.thread_execution_width(),
            first.thread_execution_width()
        );
        assert_eq!(source_hash(source), source_hash(source));
        assert!(runtime.buffer::<u32>(4).length() >= 16);
        assert!(runtime.buffer_with(&[1u32, 2, 3]).length() >= 12);
    }
}
