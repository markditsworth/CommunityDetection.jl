module CommunityDetection
using LightGraphs, SimpleWeightedGraphs
using ArnoldiMethod: LR, SR
using LinearAlgebra: I, Diagonal
using Clustering: kmeans

export community_detection_nback, community_detection_bethe, community_detection_louvain

"""
    community_detection_louvain(g)
g: SimpleGraph or SimpleWeightedGraph
returns: community set - Dict(community id number => array of vertices)
Begins with each node in its own community, and iteratively joins communities such that the greatest increase in modularity
is achieved. Iterates until not increase in modularity is possible.
Code is largely drawn from the Zen python library (D. Ruths).

### References
- [V. Blondel](https://arxiv.org/pdf/0803.0476.pdf)
- [D. Ruths](https://github.com/networkdynamics/zenlib)
"""

function community_detection_louvain(g)
	comms = collect(1:nv(g));
	improved, cset = louvain_step(g,comms);
	println(cset);
	meta_g = create_meta_graph(g,comms);
	while improved
		comms = collect(1:nv(meta_g));
		improved, cset = louvain_step(meta_g,comms,cset);
		println(cset);
		if improved
			meta_g = create_meta_graph(meta_g,comms);
		end
	end
	return cset
end

""" louvain_step(g, comms::AbstractArray, cset::Dict)
Performs the step of optimizing community membership for modularity
"""

function louvain_step(g,comms::AbstractArray,cset::Dict)
	# get sum of edges
	sum_edges = sum_up_edges(g);
	# optimize modularity
	improved = optimize_modularity!(comms, g, sum_edges);
	renumber!(comms);
	# update community membership
	cset = make_community_set(comms,cset);
	return improved,cset
end

""" louvain_step(g, comms::AbstractArray)
Performs the step of optimizing community membership for modularity, with no initial community set
"""

function louvain_step(g,comms::AbstractArray)
        # get sum of edges
        sum_edges = sum_up_edges(g);
        # optimize modularity
        improved = optimize_modularity!(comms, g, sum_edges);
	renumber!(comms);
        # update community membership
        cset = make_community_set(comms);
        return improved,cset
end


""" renumber!(arr::AbstractArray)
Renumbers elements such that they are from 1:n
E.g. [2,4,3,3,6] -> [1,3,2,2,4]
"""

function renumber!(arr::AbstractArray)
	sorted_unique = sort(collect(Set(arr)));
	conversion_dict = Dict()
	for idx in 1:length(sorted_unique);
		conversion_dict[sorted_unique[idx]] = idx;
	end
	for i in 1:length(arr)
		arr[i] = conversion_dict[arr[i]];
	end
end

""" make_community_set(comms::AbstractArray)
returns dict of community members
"""

function make_community_set(comms::AbstractArray)
	community_set = Dict()
	for i in comms
		community_set[i] = i;
	end
	return community_set
end

""" make_community_set(comms::AbstractArray,cset::Dict)
returns consonlidated community set based on the joined communities
"""

function make_community_set(comms::AbstractArray,cset::Dict)
	community_set = Dict();
	# initialize community_set
	for i in 1:maximum(comms)
		community_set[i] = [];
	end
	for idx in 1:length(comms)
		append!(community_set[comms[idx]],cset[idx]);
	end
	return community_set
end

"""
    sum_up_edges(g::SimpleGraph)
returns the sum of edges in graph g
"""
function sum_up_edges(g::SimpleGraph)
	return ne(g);
end

"""
    sum_up_edges(g::SimpleWeightedGraph)
returns sum of weighted edges in graph g
"""

function sum_up_edges(g::SimpleWeightedGraph)
	sum_edges = 0.0;
	for edge in edges(g)
		sum_edges += edge.weight;
	end
	return sum_edges;
end

"""
    sum_incident_edges(g::SimpleGraph,v::Int)
returns the sum of the edges incident to vertex v
"""

function sum_incident_edges(g::SimpleGraph,v::Int)
	return length(neighbors(g,v))
end

"""
    sum_incident_edges(g::SimpleWeightedGraph,v::Int)
returns the sum of the edges incident to vertex v
"""

function sum_incident_edges(g::SimpleWeightedGraph,v::Int)
	k=0.0;
	for n in neighbors(g,v)
		k += g.weights[v,n]
	end
	return k
end

"""
    create_meta_graph(g::SimpleGraph,comms::AbstractArray)
consolidates the communities into nodes themselves, and lumps edges together.
"""

function create_meta_graph(g::SimpleGraph,comms::AbstractArray)
	G = SimpleWeightedGraph(maximum(comms));
	for e in edges(g)
		if has_edge(G,comms[e.src],comms[e.dst])
			G.weights[comms[e.src],comms[e.dst]] += 1;
			G.weights[comms[e.dst],comms[e.src]] = G.weights[comms[e.src],comms[e.dst]];
		else
			add_edge!(G,comms[e.src],comms[e.dst],1);
		end
	end
	return G
end

"""
    create_meta_graph(g::SimpleWeightedGraph,comms::AbstractArray)
consolidates the communities into nodes themselves, and lumps edges together.
"""

function create_meta_graph(g::SimpleWeightedGraph,comms::AbstractArray)
        G = SimpleWeightedGraph(maximum(comms));
        for e in edges(g)
                if has_edge(G,comms[e.src],comms[e.dst])
			G.weights[comms[e.src],comms[e.dst]] += g.weights[e.dst,e.src];
			G.weights[comms[e.dst],comms[e.src]] = G.weights[comms[e.src],comms[e.dst]];
                else
			add_edge!(G,comms[e.src],comms[e.dst],g.weights[e.dst,e.src]);
                end
        end
        return G
end


"""
    optimize_modularity!(comms::AbstractArray, g::SimpleGraph, sum_edges::Number)
modifies community membership such that modularity in g is maximized
"""

function optimize_modularity!(comms::AbstractArray, g::SimpleGraph, sum_edges::Number)
	moved = true;
	improved = false;
	while moved
		moved = false;
		for v in vertices(g)
			best_community = comms[v];
			old_community = comms[v];
			max_dq = 0;
				
			# calc ki
			ki = sum_incident_edges(g,v);
			for n in all_neighbors(g,v)
				# ignore self-loops
				if n == v
					continue
				end
				# calculate remaining required values
				sum_in = 0; #because g::SimpleGraph, it is the first step, and each node is its own community
				sum_to = length(all_neighbors(g,n)) - sum_in;
				kiin = convert(Int,has_edge(g,v,n));
				dq = modularity_change(sum_in, sum_to, ki, kiin, sum_edges);
				if dq > max_dq
					max_dq = dq;
					best_community = comms[n]
				end
			end
			if best_community != old_community
				moved = true;
				improved = true;
			end
			# update community assignments
			comms[v] = best_community;
		end
	end
	return improved
end

"""
    optimize_modularity!(comms::AbstractArray, g::SimpleWeightedGraph, sum_edges::Number)
modifies community membership such that modularity in g is maximized
"""

function optimize_modularity!(comms::AbstractArray, g::SimpleWeightedGraph, sum_edges::Number)
        moved = true;
        improved = false;
        while moved
                moved = false;
                for v in vertices(g)
                        best_community = comms[v];
                        old_community = comms[v];
                        max_dq = 0;

                        # calc ki
                        ki = 0;
			for i in all_neighbors(g,v)
				ki += g.weights[v,i]
			end
                        for n in all_neighbors(g,v)
                                # ignore self-loops
                                if n == v
                                        continue
                                end
                                # calculate remaining required values
				sum_in = g.weights[n,n];
				sum_to = 0;
				for neighbor in all_neighbors(g,n)
					sum_to += g.weights[n,neighbor];
				end
				kiin = g.weights[v,n];
                                dq = modularity_change(sum_in, sum_to, ki, kiin, sum_edges);
                                if dq > max_dq
                                        max_dq = dq;
                                        best_community = comms[n]
                                end
                        end
                        if best_community != old_community
                                moved = true;
                                improved = true;
                        end
                        # update community assignments
                        comms[v] = best_community;
                end
        end
        return improved
end



"""
    modularity_change(sum_in::Number, sum_to::Number, ki::Number, kiin::Number, m::NUmber)
Calculates the change in modularity
"""

function modularity_change(sum_in::Number, sum_to::Number, ki::Number, kiin::Number, m::Number)
	a = (sum_in + kiin)/(2*m);
	b = (sum_to + ki)/(2*m);
	c = sum_in/(2*m);
	d = sum_to/(2*m);
	f = ki/(2*m);
	dQ = a - (b^2) - c + (d^2) + (f^2);
	return dQ;
end

"""
    community_detection_nback(g::AbstractGraph, k::Int)

Return an array, indexed by vertex, containing commmunity assignments for
graph `g` detecting `k` communities.
Community detection is performed using the spectral properties of the 
non-backtracking matrix of `g`.

### References
- [Krzakala et al.](http://www.pnas.org/content/110/52/20935.short)
"""
function community_detection_nback(g::AbstractGraph, k::Int)
    #TODO insert check on connected_components
    ϕ = real(nonbacktrack_embedding(g, k))
    if k == 1
        c = fill(1, nv(g))
    elseif k==2
        c = community_detection_threshold(g, ϕ[1,:])
    else
        c = kmeans(ϕ, k).assignments
    end
    return c
end

function community_detection_threshold(g::AbstractGraph, coords::AbstractArray)
    # TODO use a more intelligent method to set the threshold
    # 0 based thresholds are highly sensitive to errors.
    c = ones(Int, nv(g))
    # idx = sortperm(λ, lt=(x,y)-> abs(x) > abs(y))[2:k] #the second eigenvector is the relevant one
    for i=1:nv(g)
        c[i] = coords[i] > 0 ?  1 : 2
    end
    return c
end


"""
	nonbacktrack_embedding(g::AbstractGraph, k::Int)

Perform spectral embedding of the non-backtracking matrix of `g`. Return
a matrix ϕ where ϕ[:,i] are the coordinates for vertex i.

### Implementation Notes
Does not explicitly construct the `non_backtracking_matrix`.
See `Nonbacktracking` for details.

### References
- [Krzakala et al.](http://www.pnas.org/content/110/52/20935.short).
"""
function nonbacktrack_embedding(g::AbstractGraph, k::Int)
    B = Nonbacktracking(g)
    λ, eigv = LightGraphs.LinAlg.eigs(B, nev=k+1, which=LR())
    ϕ = zeros(ComplexF32, nv(g), k-1)
    # TODO decide what to do with the stationary distribution ϕ[:,1]
    # this code just throws it away in favor of eigv[:,2:k+1].
    # we might also use the degree distribution to scale these vectors as is
    # common with the laplacian/adjacency methods.
    for n=1:k-1
        v= eigv[:,n+1]
        ϕ[:,n] = contract(B, v)
    end
    return ϕ'
end


"""
    community_detection_bethe(g::AbstractGraph, k=-1; kmax=15)

Perform detection for `k` communities using the spectral properties of the 
Bethe Hessian matrix associated to `g`.
If `k` is omitted or less than `1`, the optimal number of communities
will be automatically selected. In this case the maximum number of
detectable communities is given by `kmax`.
Return a vector containing the vertex assignments.

### References
- [Saade et al.](http://papers.nips.cc/paper/5520-spectral-clustering-of-graphs-with-the-bethe-hessian)
"""
function community_detection_bethe(g::AbstractGraph, k::Int=-1; kmax::Int=15)
    A = adjacency_matrix(g)
    D = Diagonal(degree(g))
    r = (sum(degree(g)) / nv(g))^0.5

    Hr = Matrix((r^2-1)*I, nv(g), nv(g)) - r*A + D;
    #Hmr = Matrix((r^2-1)*I, nv(g), nv(g)) + r*A + D;
    k >= 1 && (kmax = k)
    λ, eigv = LightGraphs.LinAlg.eigs(Hr, which=SR(), nev=min(kmax, nv(g)))

    # TODO eps() is chosen quite arbitrarily here, because some of eigenvalues
    # don't convert exactly to zero as they should. Some analysis could show
    # what threshold should be used instead
    q = something(findlast(x -> (x < -eps()), λ), 0)
    k > q && @warn("Using eigenvectors with positive eigenvalues,
                    some communities could be meaningless. Try to reduce `k`.")
    k < 1 && (k = q)
    k <= 1 && return fill(1, nv(g))
    labels = kmeans(collect(transpose(eigv[:,2:k])), k).assignments
    return labels
end
end #module
