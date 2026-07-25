.check_sparse <- function(x, argument = "x") {
  if (!inherits(x, "cudasparse")) {
    stop(sprintf("`%s` must be a `cudasparse` matrix.", argument),
         call. = FALSE)
  }
  invisible(x)
}

.validate_sparse_dimnames <- function(value, shape, argument = "dimnames") {
  if (is.null(value)) {
    return(NULL)
  }
  if (!is.list(value) || length(value) != length(shape)) {
    stop(
      sprintf(
        "`%s` must be NULL or a list with one entry per sparse dimension.",
        argument
      ),
      call. = FALSE
    )
  }

  result <- vector("list", length(shape))
  for (index in seq_along(shape)) {
    labels <- value[[index]]
    if (is.null(labels)) {
      next
    }
    if (!is.character(labels) || !is.null(dim(labels)) ||
        length(labels) != shape[[index]]) {
      stop(
        sprintf(
          "Each non-NULL `%s` entry must be a character vector matching its sparse dimension.",
          argument
        ),
        call. = FALSE
      )
    }
    result[[index]] <- labels
  }
  if (!is.null(names(value))) {
    names(result) <- names(value)
  }
  result
}

.sparse_dimnames <- function(x) {
  .validate_sparse_dimnames(x$dimnames, x$shape)
}

.sparse_axis_name <- function(value, index) {
  if (is.null(value) || is.null(names(value))) {
    return(NULL)
  }
  names(value)[[index]]
}

.meaningful_sparse_axis_name <- function(value) {
  !is.null(value) && (is.na(value) || nzchar(value))
}

.sparse_product_dimnames <- function(x, y_dimnames) {
  x_dimnames <- .sparse_dimnames(x)
  x_inner <- if (is.null(x_dimnames)) NULL else x_dimnames[[2L]]
  y_inner <- if (is.null(y_dimnames)) NULL else y_dimnames[[1L]]
  if (!is.null(x_inner) && !is.null(y_inner) &&
      !identical(x_inner, y_inner)) {
    stop(
      "Sparse and dense inner dimension names are incompatible.",
      call. = FALSE
    )
  }

  labels <- list(
    if (is.null(x_dimnames)) NULL else x_dimnames[[1L]],
    if (is.null(y_dimnames)) NULL else y_dimnames[[2L]]
  )
  x_axis <- .sparse_axis_name(x_dimnames, 1L)
  y_axis <- .sparse_axis_name(y_dimnames, 2L)
  has_axis_names <- .meaningful_sparse_axis_name(x_axis) ||
    .meaningful_sparse_axis_name(y_axis)
  if (has_axis_names) {
    names(labels) <- c(
      if (.meaningful_sparse_axis_name(x_axis)) x_axis else "",
      if (.meaningful_sparse_axis_name(y_axis)) y_axis else ""
    )
  }
  if (all(vapply(labels, is.null, logical(1))) && !has_axis_names) {
    return(NULL)
  }
  labels
}

.triplet_matrix <- function(x) {
  Matrix::sparseMatrix(
    i = x$i,
    j = x$j,
    x = x$values,
    dims = x$shape,
    dimnames = .sparse_dimnames(x),
    giveCsparse = TRUE
  )
}

.sparse_torch <- function(i, j, values, shape) {
  indices <- torch::torch_tensor(
    rbind(i - 1L, j - 1L),
    dtype = torch::torch_int64(),
    device = "cuda"
  )
  torch_values <- torch::torch_tensor(
    values,
    dtype = torch::torch_float64(),
    device = "cuda"
  )
  torch::torch_sparse_coo_tensor(
    indices = indices,
    values = torch_values,
    size = shape,
    device = "cuda"
  )$coalesce()
}

#' Create a GPU-aware sparse matrix
#'
#' @param x A numeric matrix, a sparse matrix from the `Matrix` package, or a
#'   `cudasparse` object.
#' @param format Logical storage format, `"csr"` or `"coo"`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @param drop_zeros Whether to remove explicitly stored zeros.
#'
#' @return A `cudasparse` list. Stable public metadata include one-based COO
#'   `i` and `j`, numeric `values`, zero-based CSR `row_ptr` and `col_index`,
#'   integer `shape`, matrix `dimnames`, logical `format`, actual `device`, and
#'   `backend`. `storage` is backend-internal and should not be accessed
#'   directly.
#' @export
#' @examples
#' library(Matrix)
#' x <- rsparsematrix(5, 4, density = 0.25)
#' cuda_sparse(x, device = "cpu")
cuda_sparse <- function(x, format = c("csr", "coo"),
                        device = c("auto", "cuda", "cpu"),
                        drop_zeros = TRUE) {
  format <- match.arg(format)
  device <- match.arg(device)
  if (!is.logical(drop_zeros) || length(drop_zeros) != 1L ||
      is.na(drop_zeros)) {
    stop("`drop_zeros` must be TRUE or FALSE.", call. = FALSE)
  }
  if (inherits(x, "cudasparse")) {
    input_dimnames <- .sparse_dimnames(x)
    if (device == "auto") {
      device <- if (cudatensr::cuda_available()) "cuda" else "cpu"
    }
    if (identical(x$format, format) && identical(x$device, device)) {
      return(x)
    }
    x <- .triplet_matrix(x)
  } else {
    input_dimnames <- NULL
  }
  if (!(is.matrix(x) || methods::is(x, "Matrix"))) {
    stop("`x` must be a numeric matrix or a `Matrix` sparse matrix.",
         call. = FALSE)
  }
  if (is.matrix(x) && !is.numeric(x)) {
    stop("`x` must contain numeric values.", call. = FALSE)
  }
  if (is.null(input_dimnames)) {
    input_dimnames <- .validate_sparse_dimnames(
      dimnames(x),
      dim(x),
      "dimnames(x)"
    )
  }
  if (device == "auto") {
    device <- if (cudatensr::cuda_available()) "cuda" else "cpu"
  }
  if (device == "cuda" && !cudatensr::cuda_available()) {
    stop("CUDA is unavailable; use `device = \"cpu\"`.", call. = FALSE)
  }

  sparse <- if (methods::is(x, "Matrix")) {
    converted <- tryCatch(
      methods::as(x, "dMatrix"),
      error = function(...) NULL
    )
    if (is.null(converted)) {
      stop("`x` must contain numeric values.", call. = FALSE)
    }
    methods::as(converted, "generalMatrix")
  } else {
    Matrix::Matrix(x, sparse = TRUE)
  }
  sparse <- methods::as(methods::as(sparse, "dMatrix"), "generalMatrix")
  if (isTRUE(drop_zeros)) {
    sparse <- Matrix::drop0(sparse)
  }
  entries <- Matrix::summary(sparse)
  if (NROW(entries) == 0L) {
    i <- j <- integer()
    values <- numeric()
  } else {
    order_index <- order(entries$i, entries$j)
    i <- as.integer(entries$i[order_index])
    j <- as.integer(entries$j[order_index])
    values <- as.numeric(entries$x[order_index])
  }
  if (anyNA(values) || any(!is.finite(values))) {
    stop("`x` must contain finite, non-missing values.", call. = FALSE)
  }
  shape <- as.integer(dim(sparse))
  input_dimnames <- .validate_sparse_dimnames(
    input_dimnames,
    shape,
    "dimnames(x)"
  )
  row_ptr <- c(0L, cumsum(tabulate(i, nbins = shape[[1]])))
  col_index <- j - 1L

  storage <- if (device == "cuda") {
    .sparse_torch(i, j, values, shape)
  } else {
    NULL
  }
  backend <- if (device == "cuda") "torch-coo" else "Matrix"

  structure(
    list(
      i = i,
      j = j,
      values = values,
      row_ptr = as.integer(row_ptr),
      col_index = as.integer(col_index),
      shape = shape,
      dimnames = input_dimnames,
      format = format,
      device = device,
      backend = backend,
      storage = storage
    ),
    class = "cudasparse"
  )
}

#' Inspect sparse matrix metadata
#'
#' @param x A `cudasparse` matrix.
#' @return A named list containing `shape`, `nnz`, `density`, `format`,
#'   actual `device`, and `backend`.
#' @export
#' @examples
#' sparse_info(cuda_sparse(diag(3), device = "cpu"))
sparse_info <- function(x) {
  .check_sparse(x)
  list(
    shape = x$shape,
    nnz = length(x$values),
    density = length(x$values) / prod(x$shape),
    format = x$format,
    device = x$device,
    backend = x$backend
  )
}

#' Inspect sparse matrix dimension labels
#'
#' @param x A `cudasparse` matrix.
#' @return `NULL` for an unnamed matrix, otherwise its row and column names,
#'   following base R `dimnames()` semantics.
#' @export
dimnames.cudasparse <- function(x) {
  .check_sparse(x)
  .sparse_dimnames(x)
}

#' Convert sparse storage format
#'
#' @param x A `cudasparse` matrix.
#' @return A `cudasparse` matrix.
#' @export
#' @examples
#' x <- cuda_sparse(diag(3), device = "cpu")
#' as_coo(x)
#' as_csr(x)
as_coo <- function(x) {
  .check_sparse(x)
  x$format <- "coo"
  x
}

#' @rdname as_coo
#' @export
as_csr <- function(x) {
  .check_sparse(x)
  x$format <- "csr"
  x
}

#' Convert to an R sparse matrix
#'
#' @param x A `cudasparse` matrix.
#' @return A `Matrix::dgCMatrix`.
#' @export
#' @examples
#' to_dgCMatrix(cuda_sparse(diag(3), device = "cpu"))
to_dgCMatrix <- function(x) {
  .check_sparse(x)
  methods::as(.triplet_matrix(x), "dgCMatrix")
}

#' Sparse matrix by dense matrix multiplication
#'
#' @param x A `cudasparse` matrix.
#' @param y A numeric matrix or `cudatensor`.
#' @return A dense `cudatensor`. CUDA multiplication is transferred back to
#'   the CPU in this first release so the result has portable semantics.
#' @export
#' @examples
#' x <- cuda_sparse(diag(3), device = "cpu")
#' sparse_matmul_dense(x, matrix(1:6, 3, 2))
sparse_matmul_dense <- function(x, y) {
  .check_sparse(x)
  if (inherits(y, "cudatensor")) {
    y_shape <- cudatensr::tensor_shape(y)
    y_cpu <- cudatensr::to_cpu(y)
  } else {
    y_cpu <- y
    y_shape <- dim(y)
  }
  if (!is.numeric(y_cpu) || length(y_shape) != 2L ||
      anyNA(y_cpu) || any(!is.finite(y_cpu))) {
    stop("`y` must be a finite numeric matrix or two-dimensional tensor.",
         call. = FALSE)
  }
  if (x$shape[[2]] != y_shape[[1]]) {
    stop("Sparse and dense dimensions are not conformable.",
         call. = FALSE)
  }
  y_dimnames <- .validate_sparse_dimnames(
    dimnames(y_cpu),
    y_shape,
    "dimnames(y)"
  )
  result_dimnames <- .sparse_product_dimnames(x, y_dimnames)

  if (x$device == "cuda") {
    dense_gpu <- torch::torch_tensor(
      y_cpu,
      dtype = torch::torch_float64(),
      device = "cuda"
    )
    product <- x$storage$matmul(dense_gpu)
    result <- as.array(product$to(device = "cpu"))
  } else {
    result <- as.matrix(.triplet_matrix(x) %*% y_cpu)
  }
  dim(result) <- c(x$shape[[1]], y_shape[[2]])
  dimnames(result) <- result_dimnames
  cudatensr::cuda_tensor(result, device = "cpu", dtype = "float64")
}

#' Sparse matrix-vector multiplication
#'
#' @param x A `cudasparse` matrix.
#' @param y A numeric vector.
#' @return A numeric vector.
#' @export
#' @examples
#' sparse_matvec(cuda_sparse(diag(3), device = "cpu"), 1:3)
sparse_matvec <- function(x, y) {
  .check_sparse(x)
  if (!is.numeric(y) || is.matrix(y) || length(y) != x$shape[[2]] ||
      anyNA(y) || any(!is.finite(y))) {
    stop("`y` must be a finite numeric vector with one value per column.",
         call. = FALSE)
  }
  dense <- matrix(
    as.numeric(y),
    ncol = 1L,
    dimnames = list(names(y), NULL)
  )
  result <- as.vector(cudatensr::to_cpu(sparse_matmul_dense(x, dense)))
  sparse_dimnames <- .sparse_dimnames(x)
  if (!is.null(sparse_dimnames) && !is.null(sparse_dimnames[[1L]])) {
    names(result) <- sparse_dimnames[[1L]]
  }
  result
}

#' Sparse row and column reductions
#'
#' @param x A `cudasparse` matrix.
#' @return A numeric vector.
#' @export
#' @examples
#' x <- cuda_sparse(matrix(1:6, 2), device = "cpu")
#' sparse_row_sums(x)
#' sparse_col_sums(x)
sparse_row_sums <- function(x) {
  .check_sparse(x)
  result <- as.numeric(Matrix::rowSums(.triplet_matrix(x)))
  sparse_dimnames <- .sparse_dimnames(x)
  if (!is.null(sparse_dimnames) && !is.null(sparse_dimnames[[1L]])) {
    names(result) <- sparse_dimnames[[1L]]
  }
  result
}

#' @rdname sparse_row_sums
#' @export
sparse_col_sums <- function(x) {
  .check_sparse(x)
  result <- as.numeric(Matrix::colSums(.triplet_matrix(x)))
  sparse_dimnames <- .sparse_dimnames(x)
  if (!is.null(sparse_dimnames) && !is.null(sparse_dimnames[[2L]])) {
    names(result) <- sparse_dimnames[[2L]]
  }
  result
}

#' @export
dim.cudasparse <- function(x) {
  x$shape
}

#' @export
print.cudasparse <- function(x, ...) {
  info <- sparse_info(x)
  cat(sprintf(
    "<cudasparse[%sx%s] nnz=%s format=%s device=%s backend=%s>\n",
    info$shape[[1]], info$shape[[2]], info$nnz,
    info$format, info$device, info$backend
  ))
  max_values <- getOption("cudasparsr.max_print", 100L)
  if (!is.numeric(max_values) || length(max_values) != 1L ||
      is.na(max_values) || !is.finite(max_values) || max_values < 0) {
    max_values <- 100L
  }
  if (info$nnz <= max_values) {
    print(to_dgCMatrix(x), ...)
  } else {
    cat(
      sprintf(
        "<%s stored values omitted; use `to_dgCMatrix()` to materialize>\n",
        format(info$nnz, big.mark = ",", scientific = FALSE)
      )
    )
  }
  invisible(x)
}
