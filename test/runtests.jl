#using CommunityDetection
include("/opt/src/CommunityDetection.jl")
using .CommunityDetection
using LightGraphs
using SimpleWeightedGraphs
using LinearAlgebra: I, norm
using ArnoldiMethod: LR
using SparseArrays: sparse
using Test

""" Spectral embedding of the non-backtracking matrix of `g`
(see [Krzakala et al.](http://www.pnas.org/content/110/52/20935.short)).

`g`: input Graph
`k`: number of dimensions in which to embed

return : a matrix ϕ where ϕ[:,i] are the coordinates for vertex i.
"""
function nonbacktrack_embedding_dense(g::AbstractGraph, k::Int)
    B, edgeid = non_backtracking_matrix(g)
    λ, eigv = LightGraphs.LinAlg.eigs(B, nev=k+1, which=LR())
    ϕ = zeros(ComplexF32, k-1, nv(g))
    # TODO decide what to do with the stationary distribution ϕ[:,1]
    # this code just throws it away in favor of eigv[:,2:k+1].
    # we might also use the degree distribution to scale these vectors as is
    # common with the laplacian/adjacency methods.
    for n=1:k-1
        v= eigv[:,n+1]
        for i=1:nv(g)
            for j in neighbors(g, i)
                u = edgeid[Edge(j,i)]
                ϕ[n,i] += v[u]
            end
        end
    end
    return ϕ
end

@testset "CommunityDetection" begin

n = 10; k = 5
pg = PathGraph(n)
ϕ1 = CommunityDetection.nonbacktrack_embedding(pg, k)'

nbt = Nonbacktracking(pg)
B, emap = non_backtracking_matrix(pg)
Bs = sparse(nbt)
@test sparse(B) == Bs

# check that matvec works
x = ones(Float64, nbt.m)
y = nbt * x
z = B * x
@test norm(y-z) < 1e-8

#check that matmat works and full(nbt) == B
@test norm(nbt*Matrix{Float64}(I, nbt.m, nbt.m) - B) < 1e-8

#check that we can use the implicit matvec in nonbacktrack_embedding
@test size(y) == size(x)
ϕ2 = nonbacktrack_embedding_dense(pg, k)'
@test size(ϕ2) == size(ϕ1)

#check that this recovers communities in the path of cliques
@testset "community_detection_nback(z, k)" begin
    n=10
    g10 = CompleteGraph(n)
    z = copy(g10)
    for k=2:5
        z = blockdiag(z, g10)
        add_edge!(z, (k-1)*n, k*n)

        c = community_detection_nback(z, k)
        @test sort(union(c)) == [1:k;]
        a = collect(n:n:k*n)
        @test length(c[a]) == length(unique(c[a]))
        for i=1:k
            cluster_range = (1:n) .+ (i-1)*n
            @test length(unique(c[cluster_range])) == 1
        end
    end
end

@testset "community_detection_bethe(z, k)" begin
    n=10
    g10 = CompleteGraph(n)
    z = copy(g10)
    for k=2:5
        z = blockdiag(z, g10)
        add_edge!(z, (k-1)*n, k*n)

        c = community_detection_bethe(z, k)
        @test sort(union(c)) == [1:k;]
        a = collect(n:n:k*n)
        @test length(c[a]) == length(unique(c[a]))

        for i=1:k
            cluster_range = (1:n) .+ (i-1)*n
            @test length(unique(c[cluster_range])) == 1
        end

    end
end

@testset "community_detection_bethe(z)" begin
    n=10
    g10 = CompleteGraph(n)
    z = copy(g10)
    for k=2:5
        z = blockdiag(z, g10)
        add_edge!(z, (k-1)*n, k*n)

        c = community_detection_bethe(z)
        @test sort(union(c)) == [1:k;]
        a = collect(n:n:k*n)
        @test length(c[a]) == length(unique(c[a]))

        for i=1:k
            cluster_range = (1:n) .+ (i-1)*n
            @test length(unique(c[cluster_range])) == 1
        end
    end
end

@testset "Louvain" begin
    # test utility functions on unweighted graph
    g = Graph(5);
    add_edge!(g,1,2);
    add_edge!(g,1,3);
    add_edge!(g,2,5);
    add_edge!(g,3,5);
    add_edge!(g,4,5);
    @test CommunityDetection.sum_up_edges(g) == 5
    @test CommunityDetection.sum_incident_edges(g,5) == 3
    # test the meta-graph creation
    community_labels = [1,1,2,2,2];
    g_meta = CommunityDetection.create_meta_graph(g,community_labels);
    @test g_meta.weights[1,1] == 1
    @test g_meta.weights[2,2] == 2
    @test g_meta.weights[1,2] == 2
    @test g_meta.weights[2,1] == 2
    # test utility function on weighted graph
    g = SimpleWeightedGraph([1,1,5,5,5],[2,3,2,3,4],[1,2,3,4,5]);
    @test CommunityDetection.sum_up_edges(g) == 15
    @test CommunityDetection.sum_incident_edges(g,5) == 12
    # test the meta-graph creation
    g_meta = CommunityDetection.create_meta_graph(g,community_labels);
    @test g_meta.weights[1,1] == 1
    @test g_meta.weights[2,2] == 9
    @test g_meta.weights[1,2] == 5
    @test g_meta.weights[2,1] == 5
    # test the optimization of modularity. Two cliques are joined by one edge,
    # and two nodes from different cliques are put in the other's community.
    # The resulting optimized communities should be the two cliques.
    g = Graph(10);
    for i in 1:5
	    for j in 1:5
		    if i > j
			    add_edge!(g,i,j);
			    add_edge!(g,i+5,j+5);
		    end
	    end
    end
    add_edge!(g,3,6);
    community_labels = [2,1,1,1,1,2,2,2,1,2];
    sum_edges = CommunityDetection.sum_up_edges(g);
    CommunityDetection.optimize_modularity!(community_labels,g,sum_edges);
    @test community_labels == [1,1,1,1,1,2,2,2,2,2]
    # test on weighted graph
    g = SimpleWeightedGraph(10);
    for i in 1:5
	    for j in 1:5
		    if i > j
			    add_edge!(g,i,j,rand(2:5));
			    add_edge!(g,i+5,j+5,rand(2:5));
		    end
	    end
    end
    add_edge!(g,3,6,rand(2:5));
    community_labels = [2,1,1,1,1,2,2,2,1,2];
    sum_edges = CommunityDetection.sum_up_edges(g);
    CommunityDetection.optimize_modularity!(community_labels,g,sum_edges);
    @test community_labels == [1,1,1,1,1,2,2,2,2,2]
end
end
