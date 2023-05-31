using Test
include(normpath(joinpath(@__FILE__,"..",".."))*"Algorithm/hermitianUpdate.jl")

A = [1 1e-10 1e-10
     1e-10 0.5^0.5 0.5^0.5
     1e-10 0.5^0.5 0.5^0.5];

function test_ev(H::HermitianUpdate)
    M = Hermitian{Float64}(H)
    D = H.V' * M * H.V
    # D is approximately a diagonal matrix.
    @test Diagonal(diag(D)) ≈ D
    # Lambda is in sorted order.
    @test sortperm(H.Lambda) == Vector(1:length(H.indices))
    # Eigenvectors correspond to eigenvalues.
    @test diag(D) ≈ H.Lambda
end

h = HermitianUpdate(A)
@test h.indices == []

h1 = with_update(h, 1)
@test h.indices == []
@test h1.indices == [1]
@test h1.Lambda == [1.]
@test h1.V == ones(1,1)
test_ev(h1)

h2 = with_update(h1, 2)
@test h2.indices == [1, 2]
@test h2.Lambda ≈ [0.5^0.5, 1]
@test h2.V ≈ [0 -1; -1 0]
test_ev(h2)

h3 = with_update(h2, 3)
@test h3.indices == 1:3
@test h3.Lambda ≈ [0, 1, 2^0.5]
@test h3.V ≈ [
    # Null space; variable 1; variables 2 and 3.
    [0; 1; -1] / sqrt(2) [1; 0; 0] [0; -1; -1] / sqrt(2)
]
test_ev(h3)

using Random
Random.seed!(0)
N = 20
A = rand(N, N)
A = A * A'
h = HermitianUpdate(A)
test_ev(h)

h1 = with_update(h, 1)
test_ev(h1)

h2 = with_update(h1, 2)
test_ev(h2)

h3 = with_update(h2, 3)
test_ev(h3)

h4 = with_update(h3, 4)
test_ev(h4)

h5 = with_update(h4, 5)
test_ev(h5)

h6 = with_update(h5, 6)
test_ev(h6)

h7 = with_update(h6, 7)
test_ev(h7)
@assert Hermitian{Float64}(h7) == A[1:7, 1:7]