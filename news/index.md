# Changelog

## cudasparsr 0.2.0

- Adopted the shared `cudaverse-stage/1` provenance contract for sparse
  construction, materialization, multiplication, matrix-vector products,
  and reductions. CUDA multiplication followed by CPU materialization is
  now represented as hybrid execution.
- Fixed `cuda_sparse(existing, drop_zeros = TRUE)` so an otherwise
  reusable `cudasparse` object no longer bypasses removal of explicitly
  stored zeros.
- [`sparse_info()`](https://cudaverse.github.io/cudasparsr/reference/sparse_info.md)
  now reports the provenance schema and aggregate compute device.

## cudasparsr 0.1.2

- Sparse row and column names are now retained across construction,
  format conversion, CPU/CUDA storage, `dgCMatrix` round-trips,
  products, and reductions. Labeled sparse-dense products reject
  incompatible contracted dimensions.

## cudasparsr 0.1.1

- Large sparse objects now print a compact metadata summary without
  materializing the complete `dgCMatrix`; the threshold is controlled by
  `options(cudasparsr.max_print = 100)`.
- `drop_zeros` now requires one non-missing logical value instead of
  silently treating invalid values as `FALSE`.
- Declared `cudatensr (>= 0.1.1)` as the tested dense-core compatibility
  floor.
