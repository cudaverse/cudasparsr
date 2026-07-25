# Sparse matrix-vector multiplication

Sparse matrix-vector multiplication

## Usage

``` r
sparse_matvec(x, y)
```

## Arguments

- x:

  A `cudasparse` matrix.

- y:

  A numeric vector.

## Value

A numeric vector.

## Examples

``` r
sparse_matvec(cuda_sparse(diag(3), device = "cpu"), 1:3)
#> [1] 1 2 3
```
