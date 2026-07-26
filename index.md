# cudasparsr

`cudasparsr` is the sparse matrix foundation of the **cudaverse**,
designed around R’s `Matrix` ecosystem and single-cell-scale sparse
data.

## Current capabilities

- Conversion from base matrices and `Matrix` sparse matrices, including
  `dgCMatrix`.
- COO and CSR metadata with explicit 0-based CSR row pointers.
- Sparse matrix-vector multiplication.
- Sparse-by-dense matrix multiplication.
- Row and column sums.
- Preservation of row and column identifiers through COO/CSR conversion,
  `Matrix` round-trips, sparse products, and reductions.
- Transparent CPU backend for development and CI.
- Optional CUDA COO multiplication through the public R torch sparse
  API.

## Installation

``` r

# install.packages("pak")
pak::pak("cudaverse/cudasparsr")
```

## Example

``` r

library(cudasparsr)
library(Matrix)

counts <- rsparsematrix(1000, 500, density = 0.01)
x <- cuda_sparse(counts)

sparse_info(x)
sparse_row_sums(x)
sparse_col_sums(x)

dense <- matrix(rnorm(500 * 10), 500, 10)
result <- sparse_matmul_dense(x, dense)
```

Row and column names remain attached to the sparse object and are
restored by
[`to_dgCMatrix()`](https://cudaverse.github.io/cudasparsr/reference/to_dgCMatrix.md).
Sparse-dense multiplication verifies labels on the contracted dimension
when both inputs provide them, then carries sparse row names and dense
output column names into the result.

## Backend note

The public R torch API provides a documented COO constructor. CUDA
objects therefore use a coalesced torch COO tensor internally even when
the logical format is recorded as CSR. CSR metadata are still exposed
for future native cuSPARSE integration. In this first release, dense
products are returned to the CPU after CUDA computation.

Use
[`cudatensr::cuda_diagnostics()`](https://cudaverse.github.io/cudatensr/reference/cuda_diagnostics.html)
to inspect the runtime and
[`cuda_provenance()`](https://cudaverse.github.io/cudasparsr/reference/cuda_provenance.md)
to inspect the actual stages. The [backend and provenance
tutorial](https://cudaverse.github.io/cudasparsr/articles/backend-provenance.html)
contains runnable CPU examples, an optional CUDA path, sparse memory and
transfer limitations, and the NVIDIA hardware-test contract.

| Request | Sparse storage backend | Sparse output device | Fallback |
|----|----|----|----|
| `"cpu"` | Matrix metadata/storage | CPU | No |
| `"auto"` with usable CUDA | torch COO | CUDA | No |
| `"auto"` without usable CUDA | Matrix | CPU | Yes, recorded with the reason |
| `"cuda"` | torch COO, or a strict error | CUDA when successful | Never silently |

This table describes
[`cuda_sparse()`](https://cudaverse.github.io/cudasparsr/reference/cuda_sparse.md)
construction. CUDA sparse-dense multiplication currently returns its
dense result to CPU, and row/column reductions are fixed-CPU Matrix
stages. Consult
[`cuda_provenance()`](https://cudaverse.github.io/cudasparsr/reference/cuda_provenance.md)
instead of inferring an end-to-end backend from the input object.

Printing objects with more than 100 stored values shows only metadata,
avoiding an unexpected full sparse-matrix materialization. Use
`to_dgCMatrix(x)` when you need the complete R `Matrix`, or change the
display threshold with `options(cudasparsr.max_print = 500)`.

For installation, device verification, memory advice, and common
failures, see the cudaverse [GPU setup and troubleshooting
guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li
