# Hermitian updates.
# Deflation of the system has not been implemented yet. Therefore, we want a
# matrix that has a high spark, and is also dense. We do not want the submatrix
# that we are tracking to ever be nearly singular. Also, when we append a
# row/column to the system, after change-of-basis onto the original eigenvectors
# of the system, we never want any entry in the rank-one update matrix to be near
# zero. The dot product between variables being tracked should never be near zero.
# These issues can be resolved with more robust entry and exit checks (see LAPACK:
# DLAED7 which is the outer routine around DLAED9).
using Base.Iterators
using LinearAlgebra
using SparseArrays

struct HermitianUpdate
    H::Matrix{Float32}
    indices::Vector{Int}
    V::Matrix{Float32}
    Lambda::Vector{Float32}

end
HermitianUpdate(H) = HermitianUpdate(H, [], zeros(0,0), [])

function with_update(H::HermitianUpdate, index::Int)
    i = length(H.indices)
    # SVD of a 1 by 1 nonnegative matrix
    if i == 0
        return HermitianUpdate(H.H, [index], ones(1,1), H.H[index:index, index])
    end

    A = H.H[:,H.indices][H.indices,:]
    A = [A zeros(i,1)
            zeros(1,i+1)]
    # E is a symmetric matrix with a zero top-left block, and
    # with update as its bottom row and as its bottom column.
    update = H.H[[H.indices; index], index]
    # A is an i-by-i matrix. A = V Λ V'. The update (new column)
    # can be transformed by multiplying V' on the left, which
    # only affects the last column of the update matrix E.
    # Now, we want the eigenvalues in ascending order, and we have
    # added a zero row and column to the end of A (this comes first
    # in the list of eigenvectors).
    # V (augmented) has two permuted blocks (first representing the
    # final row/column, which will be zero, and second representing
    # the solved eigensystem A).
    V_augmented = [zeros(i,1) H.V
                   1 zeros(1,i)]
    # Change-of-basis to the diagonalization of the first i
    # variables.
    update = V_augmented' * update

    # We have A+E: after partial diagonalization, the support of
    # the matrix is the diagonal, and first row and column. We seek
    # a Cholesky factor, where the support will be the diagonal and
    # the last row.
    chol_diag = [NaN; sqrt.(H.Lambda)]
    chol_first_row = update ./ chol_diag
    chol_diag[1] = sqrt(update[1] - sum(chol_first_row[2:end] .^ 2))
    # Unscaled "zeta" update to be solved.
    chol_first_row[1] = chol_diag[1]

    # Current size of the system
    K = length(update)

    # Lambda before perturbation
    D_old = Vector{Float32}([0; H.Lambda])
    # Lambda after perturbation
    D = Array{Float32}(undef, i+1)
    # Change-of-basis to be applied inside LAPACK. The identity.
    Q = Matrix{Float32}(I, K, K)
    # Eigenvectors of L' L
    S = Matrix{Float32}(undef, i+1, i+1)

    Kstart = 1
    Kstop = K
    # Also the size of the system
    N = K
    ldq = K
    lds = K
    # Parameterize update into zeta (unit-normed) and rho (magnitude)
    rho = Float32(norm(chol_first_row))
    zeta = Array{Float32}(chol_first_row) / rho
    # Go from the norm of the vector to the norm of the rank-one matrix.
    rho = rho ^ 2
    # Status bits
    info = Vector{Int32}([-1000])
    status = ccall(
        (:slaed9_, "liblapack"),
        Int64,
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32},
            Ptr{Float32}, Ptr{Float32},
            Ref{Int32}, Ref{Float32},
            Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
            Ref{Int32}, Ptr{Int32}),
        K, Kstart, Kstop, N, D, Q, ldq, rho, D_old, zeta, S, lds, info)
    if 0 != info[1]
        error("LAPACK rank-one update status $info")
    end

    # Compute left singular vectors of L, where S are the right singular vectors
    # of L.
    chol_f = sparse(
        [1
         reduce(vcat, [ones(K-1) 2:K]')],
        [1
         reduce(vcat, [2:K 2:K]')],
        [chol_diag[1]
         reduce(vcat, [chol_first_row[2:K] chol_diag[2:K]]')]
    )
    U = chol_f * S * Diagonal(1 ./ sqrt.(D))
    # @assert U * U' ≈ I "Unitary matrix $U"

    # We have L' L = S D S' (then, V_augmented is applied on the outside). We
    # want to map L on the right so that its right singular vectors are mapped
    # to its left singular vectors. Cancel out the right singular vectors:
    # L = U D S', by right-multiplication by S. Then, apply the left singular
    # vectors on the right-hand side (U'). Transpose this unitary matrix because
    # L is transposed.
    V_augmented = V_augmented * (U' * S)'
    HermitianUpdate(H.H, [H.indices; index], V_augmented, D)
end