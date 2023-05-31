using Test
include(normpath(joinpath(@__FILE__,"..",".."))*"Algorithm/hermitianUpdate.jl")

A = [1 1e-10 1e-10
     1e-10 0.5^0.5 0.5^0.5
     1e-10 0.5^0.5 0.5^0.5];

h = HermitianUpdate(A)
@test h.indices == []

h1 = with_update(h, 1)
@test h.indices == []
@test h1.indices == [1]
@test h1.Lambda == [1.]
@test h1.V == ones(1,1)

h2 = with_update(h1, 2)
@test h2.indices == [1, 2]
@test h2.Lambda ≈ [0.5^0.5, 1]
@test h2.V ≈ [0 1; 1 0]

h3 = with_update(h2, 3)
@test h3.indices == 1:3
@test h3.Lambda ≈ [0, 1, 2^0.5]
@test h3.V ≈ [
    # Null space; variable 1; variables 2 and 3.
    [-1; 0; 1] / sqrt(2) [0; 1; 0] [1; 0; 1] / sqrt(2)
]