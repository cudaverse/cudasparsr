test_that("Matrix sparse inputs preserve values and dimensions", {
  source <- Matrix::sparseMatrix(
    i = c(1, 3, 4),
    j = c(2, 1, 3),
    x = c(2, 5, -1),
    dims = c(4, 3)
  )
  x <- cuda_sparse(source, device = "cpu")

  expect_s3_class(x, "cudasparse")
  expect_identical(dim(x), c(4L, 3L))
  expect_equal(as.matrix(to_dgCMatrix(x)), as.matrix(source))
  expect_equal(sparse_info(x)$nnz, 3)
})

test_that("COO and CSR metadata are consistent", {
  x <- cuda_sparse(diag(4), format = "csr", device = "cpu")
  expect_identical(sparse_info(as_coo(x))$format, "coo")
  expect_identical(sparse_info(as_csr(as_coo(x)))$format, "csr")
  expect_identical(x$row_ptr, 0:4)
})

test_that("sparse dense multiplication matches Matrix", {
  source <- Matrix::rsparsematrix(6, 4, density = 0.3)
  dense <- matrix(seq_len(12), 4, 3)
  x <- cuda_sparse(source, device = "cpu")

  product <- cudatensr::to_cpu(sparse_matmul_dense(x, dense))
  expect_equal(product, as.matrix(source %*% dense))
  expect_equal(sparse_matvec(x, 1:4), as.vector(source %*% (1:4)))
})

test_that("sparse reductions match Matrix", {
  source <- Matrix::rsparsematrix(5, 4, density = 0.4)
  x <- cuda_sparse(source, device = "cpu")

  expect_equal(sparse_row_sums(x), as.numeric(Matrix::rowSums(source)))
  expect_equal(sparse_col_sums(x), as.numeric(Matrix::colSums(source)))
})

test_that("invalid inputs fail clearly", {
  expect_error(cuda_sparse(matrix(c(1, NA), 1), device = "cpu"), "finite")
  expect_error(cuda_sparse(letters[1:3], device = "cpu"), "matrix")
  expect_error(
    cuda_sparse(diag(2), device = "cpu", drop_zeros = NA),
    "TRUE or FALSE"
  )

  x <- cuda_sparse(diag(3), device = "cpu")
  expect_error(sparse_matvec(x, 1:2), "one value per column")
  expect_error(sparse_matmul_dense(x, matrix(1:8, 4)), "not conformable")
})

test_that("large sparse printing avoids materializing all entries", {
  x <- cuda_sparse(diag(5), device = "cpu")
  old_options <- options(cudasparsr.max_print = 3)
  on.exit(options(old_options), add = TRUE)

  output <- capture.output(print(x))

  expect_true(any(grepl("stored values omitted", output)))
  expect_false(any(grepl("5 x 5 sparse Matrix", output, fixed = TRUE)))
})
