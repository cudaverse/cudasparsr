# Sparse backends and compute provenance

`cudasparsr` records requested devices separately from actual
computation and the device holding each output. The records follow the
shared `cudaverse-stage/1` schema. This tutorial is fully runnable on
CPU; its CUDA section is optional.

## Inspect the runtime

``` r

library(cudatensr)
library(cudasparsr)
#> 
#> Attaching package: 'cudasparsr'
#> The following object is masked from 'package:cudatensr':
#> 
#>     cuda_provenance
library(Matrix)

diagnostics <- cuda_diagnostics()
diagnostics
#> <cuda_diagnostics available=FALSE devices=0 torch=0.17.0 reason=backend_error>
```

The diagnostics identify whether torch is installed, whether CUDA is
usable, the visible device count, and a stable reason such as
`torch_not_installed`, `cuda_unavailable`, or `cuda_available`.

Construction follows this policy:

| Requested device | Actual sparse backend | Sparse output device | Fallback |
|----|----|----|----|
| `"cpu"` | Matrix | CPU | `FALSE` |
| `"auto"` with usable CUDA | torch COO | CUDA | `FALSE` |
| `"auto"` without usable CUDA | Matrix | CPU | `TRUE`, with a reason |
| `"cuda"` | torch COO, or an error | CUDA when successful | Never silent |

An explicit CUDA request is strict. It raises a
`cudaverse_cuda_unavailable` condition if no usable CUDA backend exists.

## A portable CPU sparse workflow

``` r

counts <- sparseMatrix(
  i = c(1, 2, 3, 4, 1, 3),
  j = c(1, 1, 2, 2, 3, 3),
  x = c(2, 1, 4, 3, 5, 2),
  dims = c(4, 3),
  dimnames = list(
    feature = paste0("gene_", 1:4),
    sample = paste0("sample_", 1:3)
  )
)

sparse_cpu <- cuda_sparse(counts, device = "cpu")
sparse_info(sparse_cpu)
#> $shape
#> [1] 4 3
#> 
#> $nnz
#> [1] 6
#> 
#> $density
#> [1] 0.5
#> 
#> $format
#> [1] "csr"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $provenance_schema
#> [1] "cudaverse-stage/1"
#> 
#> $compute_device
#> [1] "cpu"
cuda_provenance(sparse_cpu)
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>                   stage requested_device device backend selection_reason
#>  sparse_materialization              cpu    cpu  Matrix     explicit_cpu
#>  fallback output_device
#>     FALSE           cpu
```

The provenance table exposes:

- `requested_device`: the caller’s request or inherited/fixed policy;
- `device`: the actual computation device;
- `backend`: `Matrix`, `torch-coo`, or another concrete implementation;
- `selection_reason`: why that device was used;
- `fallback`: whether an `"auto"` request selected CPU;
- `output_device`: where the stage placed its output.

Sparse-by-dense multiplication is also runnable without CUDA:

``` r

dense <- matrix(
  seq_len(6) / 10,
  nrow = 3,
  dimnames = list(
    sample = paste0("sample_", 1:3),
    component = c("component_1", "component_2")
  )
)

product <- sparse_matmul_dense(sparse_cpu, dense)
to_cpu(product)
#>         component
#> feature  component_1 component_2
#>   gene_1         1.7         3.8
#>   gene_2         0.1         0.4
#>   gene_3         1.4         3.2
#>   gene_4         0.6         1.5

product_provenance <- cuda_provenance(product)
product_provenance
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>            stage requested_device device backend selection_reason fallback
#>  sparse_multiply        inherited    cpu  Matrix inherited_device    FALSE
#>  output_device
#>            cpu
attr(product_provenance, "schema")
#> [1] "cudaverse-stage/1"
attr(product_provenance, "compute_device")
#> [1] "cpu"
```

Reductions are currently fixed-CPU Matrix stages, even when their input
sparse object was created for CUDA:

``` r

row_totals <- sparse_row_sums(sparse_cpu)
column_totals <- sparse_col_sums(sparse_cpu)

row_totals
#> gene_1 gene_2 gene_3 gene_4 
#>      7      1      6      3 
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$row_reduction
#> $requested_device
#> [1] "fixed-cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $selection_reason
#> [1] "algorithm_cpu_only"
#> 
#> $fallback
#> [1] FALSE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
column_totals
#> sample_1 sample_2 sample_3 
#>        3        7        7 
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$column_reduction
#> $requested_device
#> [1] "fixed-cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $selection_reason
#> [1] "algorithm_cpu_only"
#> 
#> $fallback
#> [1] FALSE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
cuda_provenance(row_totals)
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>          stage requested_device device backend   selection_reason fallback
#>  row_reduction        fixed-cpu    cpu  Matrix algorithm_cpu_only    FALSE
#>  output_device
#>            cpu
```

## Automatic selection is observable

``` r

sparse_auto <- cuda_sparse(counts, device = "auto")
cuda_provenance(sparse_auto)
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>                   stage requested_device device backend selection_reason
#>  sparse_materialization             auto    cpu  Matrix    backend_error
#>  fallback output_device
#>      TRUE           cpu
```

On a CPU-only machine, the row records the automatic request, actual CPU
execution, Matrix backend, fallback flag, and diagnostic reason. This is
different from an explicit `device = "cpu"` request.

## Optional CUDA path

``` r

if (cuda_available()) {
  sparse_gpu <- cuda_sparse(counts, device = "cuda")
  cuda_provenance(sparse_gpu)

  gpu_product <- sparse_matmul_dense(sparse_gpu, dense)
  to_cpu(gpu_product)
  cuda_provenance(gpu_product)
}
```

For the product, the sparse multiplication stage computes on CUDA, but
the dense result is materialized on CPU. The provenance table therefore
shows the CUDA compute stage and CPU output-transfer stage instead of
describing the whole operation as GPU-native.

The guard makes the installed vignette portable; it does not count as
hardware coverage.

## Sparse memory and transfer limits

The logical `format` may be `"csr"` or `"coo"`, but the current CUDA
backend uses a coalesced torch COO tensor. CSR row-pointer metadata are
retained for inspection and future native cuSPARSE integration; they do
not imply native CUDA CSR storage.

Important current limits are:

- construction first extracts triplets through the R Matrix ecosystem;
- a CUDA object retains R-side indices/values and also uploads COO
  storage, so both host and device memory are involved;
- a CUDA sparse-by-dense product uploads the dense operand, computes on
  CUDA, then copies the complete dense result to CPU;
- passing a CUDA `cudatensor` as the dense operand currently
  materializes it on CPU before the sparse multiplication path uploads
  the values again;
- [`to_dgCMatrix()`](https://cudaverse.github.io/cudasparsr/reference/to_dgCMatrix.md),
  row sums, and column sums are full CPU Matrix operations;
- dense product memory scales with `nrow(x) * ncol(y)`, even when `x` is
  very sparse;
- there is no out-of-core sparse execution, native cuSPARSE CSR kernel,
  or sparse result type for sparse-by-dense multiplication yet.

Printing large sparse objects avoids complete Matrix materialization and
shows metadata only. Call
[`to_dgCMatrix()`](https://cudaverse.github.io/cudasparsr/reference/to_dgCMatrix.md)
or
[`to_cpu()`](https://cudaverse.github.io/cudatensr/reference/to_cpu.html)
only when the corresponding full host allocation is intended.

## Hardware-gated validation

Normal package checks are portable CPU checks. The repository’s
`cuda-parity` workflow runs on the self-hosted NVIDIA runner only after
manual dispatch or when the organization variable
`CUDAVERSE_NVIDIA_CI=enabled`.

The hardware job sets `CUDAVERSE_REQUIRE_CUDA=true`, requires
`nvidia-smi`, requires torch to expose at least one CUDA device, and
runs mandatory CPU/CUDA parity and provenance checks. Missing hardware
is a failure, not a successful skip.
