spectralClustering <- function(affinity, K, type=3) {
    d <- rowSums(affinity)
    d[d == 0] <- .Machine$double.eps
    D <- diag(d)
    L <- D - affinity
    if (type == 1) {
        NL <- L
    }
    else if (type == 2) {
        Di <- diag(1/d)
        NL <- Di %*% L
    }
    else if (type == 3) {
        Di <- diag(1/sqrt(d))
        NL <- Di %*% L %*% Di
    }
    eig <- eigen(NL)
    res <- sort(abs(eig$values), index.return = TRUE)
    U <- eig$vectors[, res$ix[1:K]]
    normalize <- function(x) x/sqrt(sum(x^2))
    if (type == 3) {
        U <- t(apply(U, 1, normalize))
    }
    eigDiscrete <- .discretisation(U)
    eigDiscrete <- eigDiscrete$discrete
    labels <- apply(eigDiscrete, 1, which.max)
    return(labels)
}
