using Test
using NLPModels

include("../src/ncl.jl")

function test_ncl(test::Bool) ::Test.DefaultTestSet
    @testset "NCL algorithm" begin
        @test true
    end
end