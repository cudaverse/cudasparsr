# Create a GPU-aware sparse matrix

Create a GPU-aware sparse matrix

## Usage

``` r
cuda_sparse(
  x,
  format = c("csr", "coo"),
  device = c("auto", "cuda", "cpu"),
  drop_zeros = TRUE
)
```

## Arguments

- x:

  A numeric matrix, a sparse matrix from the `Matrix` package, or a
  `cudasparse` object.

- format:

  Logical storage format, `"csr"` or `"coo"`.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

- drop_zeros:

  Whether to remove explicitly stored zeros.

## Value

A `cudasparse` list. Stable public metadata include one-based COO `i`
and `j`, numeric `values`, zero-based CSR `row_ptr` and `col_index`,
integer `shape`, matrix `dimnames`, logical `format`, actual `device`,
and `backend`. `storage` is backend-internal and should not be accessed
directly.

## Examples

``` r
library(Matrix)
x <- rsparsematrix(5, 4, density = 0.25)
cuda_sparse(x, device = "cpu")
#> <cudasparse[5x4] nnz=5 format=csr device=cpu backend=Matrix>
#> 5 x 4 sparse Matrix of class "dgCMatrix"
#>                         
#> [1,]  .    -0.47  .    .
#> [2,]  .     .     0.75 .
#> [3,]  .     .    -0.55 .
#> [4,]  .     .     .    .
#> [5,] -0.93  .    -0.86 .
```
