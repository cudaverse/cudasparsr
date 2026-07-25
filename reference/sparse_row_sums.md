# Sparse row and column reductions

Sparse row and column reductions

## Usage

``` r
sparse_row_sums(x)

sparse_col_sums(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

A numeric vector.

## Examples

``` r
x <- cuda_sparse(matrix(1:6, 2), device = "cpu")
sparse_row_sums(x)
#> [1]  9 12
sparse_col_sums(x)
#> [1]  3  7 11
```
