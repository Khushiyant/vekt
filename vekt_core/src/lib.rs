pub mod blobs;
pub mod compression;
pub mod diff;
pub mod errors;
pub mod remote;
pub mod storage;
pub mod utils;
pub mod validation;

use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::BTreeMap;

use storage::{ManifestTensor, RawHeader, VektManifest};

pub trait ModelArchiver {
    fn process(&self, save_blobs: bool) -> Result<VektManifest, Box<dyn std::error::Error>>;
    fn restore(
        manifest: &VektManifest,
        output_path: &std::path::Path,
        filter: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct SafetensorFile {
    pub header: RawHeader,
    pub mmap: Mmap,
    pub header_len: usize,
}

impl SafetensorFile {
    pub fn new(mmap: Mmap, header: RawHeader, header_len: usize) -> Self {
        SafetensorFile {
            header,
            mmap,
            header_len,
        }
    }
    pub fn open(path: &str) -> std::io::Result<Self> {
        // Open the file and create a memory-mapped buffer
        use std::fs::File;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read the header length (first 8 bytes)
        let header_len_bytes = &mmap[0..8];
        let header_len = usize::from_le_bytes(header_len_bytes.try_into().unwrap());

        // Read the header JSON
        let header_json_bytes = &mmap[8..8 + header_len];
        let header_json_str = std::str::from_utf8(header_json_bytes).unwrap();
        let header: RawHeader = serde_json::from_str(header_json_str).unwrap();

        Ok(SafetensorFile::new(mmap, header, header_len))
    }
}

impl ModelArchiver for SafetensorFile {
    fn process(&self, save_blobs: bool) -> Result<VektManifest, Box<dyn std::error::Error>> {
        let header_entries: Vec<(usize, &String, &storage::RawTensorMetaData)> = self
            .header
            .iter()
            .enumerate()
            .map(|(i, (k, v))| (i, k, v))
            .collect();

        let results: BTreeMap<String, ManifestTensor> = header_entries
            .par_iter()
            .filter_map(
                |(index, tensor_name, tensor_meta)| {
                    let (start, end) = tensor_meta.data_offsets;
                    let absolute_start = self.header_len + 8 + start;
                    let absolute_end = self.header_len + 8 + end;
                    if absolute_end > self.mmap.len() {
                        eprintln!(
                            "Corrupt Tensor '{}': Ends at byte {}, but file is only {} bytes. Skipping.",
                            tensor_name, absolute_end, self.mmap.len()
                        );
                        return None;
                    }
                    let data_slice = &self.mmap[absolute_start..absolute_end];

                    // Use centralized blob hash computation
                    let hash_hex = blobs::compute_blob_hash(data_slice);

                    if save_blobs {
                        // Use centralized blob saving with deduplication
                        if let Err(e) = blobs::write_blob_atomic(data_slice) {
                            eprintln!("Failed to save blob for tensor '{}': {}", tensor_name, e);
                        }
                    }

                    Some((
                        (*tensor_name).clone(),
                        ManifestTensor {
                            shape: tensor_meta.shape.clone(),
                            dtype: tensor_meta.dtype.clone(),
                            hash: hash_hex,
                            extra: tensor_meta.extra.clone(),
                            index: *index,
                        },
                    ))
                },
            )
            .collect();
        Ok(VektManifest {
            tensors: results,
            version: "1.0".to_string(),
            total_size: self.mmap.len(),
        })
    }
    fn restore(
        manifest: &VektManifest,
        output_path: &std::path::Path,
        filter: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Delegate to the existing restore logic in VektManifest
        // In a real multi-format system, this logic would likely live here or in a Safetensors-specific module
        manifest.restore(output_path, filter)
    }
}
