# Changelog

## cudasparsr 0.1.1

- Large sparse objects now print a compact metadata summary without
  materializing the complete `dgCMatrix`; the threshold is controlled by
  `options(cudasparsr.max_print = 100)`.
- `drop_zeros` now requires one non-missing logical value instead of
  silently treating invalid values as `FALSE`.
- Declared `cudatensr (>= 0.1.1)` as the tested dense-core compatibility
  floor.
