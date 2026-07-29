use super::Leaf;

const TILE_ELEMENTS: usize = 65_536;

pub(crate) fn check_layout(leaves: &[Leaf]) -> Result<usize, String> {
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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Step {
    pub byte_offset: u32,
    pub element_offset: u32,
    pub length: u32,
    pub bits: u32,
    pub encoding: u32,
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
                encoding: leaf.encoding as u32,
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
