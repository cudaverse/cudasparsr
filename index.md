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

## Backend note

The public R torch API provides a documented COO constructor. CUDA
objects therefore use a coalesced torch COO tensor internally even when
the logical format is recorded as CSR. CSR metadata are still exposed
for future native cuSPARSE integration. In this first release, dense
products are returned to the CPU after CUDA computation.

Printing objects with more than 100 stored values shows only metadata,
avoiding an unexpected full sparse-matrix materialization. Use
`to_dgCMatrix(x)` when you need the complete R `Matrix`, or change the
display threshold with `options(cudasparsr.max_print = 500)`.

For installation, device verification, memory advice, and common
failures, see the cudaverse [GPU setup and troubleshooting
guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li
