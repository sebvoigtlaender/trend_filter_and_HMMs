using StatsBase, PyPlot
import Base: +, -, *

ind(x, k) = x == k ? Bool(1) : Bool(0)

struct PNormalCoefs{T}
    a_1::T
    a_2::T
    a_3::T

    PNormalCoefs{T}(a_3::Real) where {T} = new(0, 0, a_3)
    PNormalCoefs{T}(a_1::Real, a_2::Real, a_3::Real) where {T} = new(a_1, a_2, a_3)
end
+(a::PNormalCoefs{T}, b::PNormalCoefs{T}) where {T} = PNormalCoefs{T}(a.a_1 + b.a_1, a.a_2 + b.a_2, a.a_3 + b.a_3)
-(a::PNormalCoefs{T}, b::PNormalCoefs{T}) where {T} = PNormalCoefs{T}(a.a_1 - b.a_1, a.a_2 - b.a_2, a.a_3 - b.a_3)
+(a::PNormalCoefs{T}, b::Real) where {T} = PNormalCoefs{T}(a.a_1, a.a_2, a.a_3 + b)
-(a::PNormalCoefs{T}, b::Real) where {T} = PNormalCoefs{T}(a.a_1, a.a_2, a.a_3 - b)
*(a::Real, b::PNormalCoefs{T}) where {T} = PNormalCoefs{T}(a*b.a_1, a*b.a_2, a*b.a_3)

function solveforb(a::PNormalCoefs{T}, λ::Real, y::T) where T

    if a.a_3 - λ != 0 && ((a.a_1 + a.a_2)^2 + 4 * (a.a_3 - λ) * (a.a_2 - a.a_1 + (a.a_3 - λ))) >= 0
                        
        sol_1 = (-(a.a_1 + a.a_2) + sqrt((a.a_1 + a.a_2)^2 + 4 * (a.a_3 - λ) * (a.a_2 - a.a_1 + (a.a_3 - λ))))/(2 * (a.a_3 - λ)) 
        sol_2 = (-(a.a_1 + a.a_2) - sqrt((a.a_1 + a.a_2)^2 + 4 * (a.a_3 - λ) * (a.a_2 - a.a_1 + (a.a_3 - λ))))/(2 * (a.a_3 - λ))
        
        sol = (sol_2 >= 1 || sol_2 <= -1) ? sol_1 : sol_2  #gewagt

    elseif a.a_3 - λ != 0 && ((a.a_1 + a.a_2)^2 + 4 * (a.a_3 - λ) * (a.a_2 - a.a_1 + (a.a_3 - λ))) < 0

        sol = -ind(y, -1) + ind(y, 1)
    
    elseif a.a_3 - λ == 0 && a.a_1 + a.a_2 != 0

        sol = (a.a_2 - a.a_1)/(a.a_1 + a.a_2)

    elseif a.a_3 - λ == 0 && a.a_1 + a.a_2 == 0

        sol = -ind(y, -1) + ind(y, 1)
                
    else print("No solution found")
    
    end
    sol
end


b_lt(a::PNormalCoefs{T}, λ::Real, y::T, x::Real) where {T} = solveforb(a, λ, y) <= x
b_gt(a::PNormalCoefs{T}, λ::Real, y::T, x::Real) where {T} = solveforb(a, λ, y) >= x

mutable struct PKnot{T,S} # need mutable because of line 127 
    pos::T
    coefs::S
    sign::Int8
end

struct PFusedLasso{T,S} <: RegressionModel
    β::Vector{T}               # Coefficients
    knots::Vector{PKnot{T,S}}  # Active knots
    bp::Matrix{T}              # Backpointers
end

function StatsBase.fit(::Type{PFusedLasso}, y::AbstractVector{T}, γ::AbstractVector{T}, λ::Real; dofit::Bool=true) where T
    S = PNormalCoefs{T}
    pflsa = PFusedLasso{T,S}(Array{T}(undef, length(y)), Array{PKnot{T,S}}(undef, 2), Array{T}(undef, 2, length(y)-1))
    dofit && fit!(pflsa, y, γ, λ)
    pflsa
end

function StatsBase.fit!(pflsa::PFusedLasso{T,S}, y::AbstractVector{T}, γ::AbstractVector{T}, λ::Real) where {T,S}
    β = pflsa.β
    knots = pflsa.knots
    bp = pflsa.bp

    γ = vcat([0.0],γ)

    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))
    length(γ) == length(y) || throw(ArgumentError("perturbation size $(length(γ)) does not match input size $(length(y))"))

    resize!(knots, 2)
    knots[1] = PKnot{T,S}(-Inf, S(0), 1)
    knots[2] = PKnot{T,S}(Inf, S(0), -1)

    @inbounds for k = 1:length(y)-1
        t1 = 0
        t2 = 0
        aminus = PNormalCoefs{T}(ind(y[k], -1), ind(y[k], 1), 0)    # Algorithm 2 line 4
        for outer t1 = 1:length(knots)-1                            # Algorithm 2 line 5
            knot = knots[t1]
            aminus += knot.sign*knot.coefs
            b_lt(aminus, λ, y[k], knots[t1+1].pos) && break         # Algorithm 2 line 7-8
        end
        bminus = solveforb(aminus, λ, y[k])

        aplus = PNormalCoefs{T}(ind(y[k], -1), ind(y[k], 1), 0)     # Algorithm 2 line 15
        t2 = length(knots)
        while t2 >= 2
            knot = knots[t2]
            aplus -= knot.sign*knot.coefs                           # Algorithm 2 line 17
            b_gt(aplus, -λ, y[k], knots[t2-1].pos) && break
            t2 -= 1
        end
        bplus = solveforb(aplus, -λ, y[k])

        if t2 > t1
            deleteat!(knots, 1:t1)
            deleteat!(knots, t2-t1:length(knots))
        else
            resize!(knots, 0)
        end
        pushfirst!(knots, PKnot{T,S}(bminus, aminus - λ, 1))
        pushfirst!(knots, PKnot{T,S}(-Inf, S(λ), 1))
        push!(knots, PKnot{T,S}(bplus, aplus + λ, -1))
        push!(knots, PKnot{T,S}(Inf, S(-λ), -1))
#         for i = 3:length(knots)-2
#             knots[i].pos = knots[i].pos + γ[k]
#         end
        bp[1, k] = bminus
        bp[2, k] = bplus

    end

    aminus = PNormalCoefs{T}(ind(y[end], -1), ind(y[end], 1), 0)
    for t1 = 1:length(knots)
        knot = knots[t1]
        aminus += knot.sign*knot.coefs
        b_lt(aminus, 0, y[end], knots[t1+1].pos) && break
    end
    β[end] = solveforb(aminus, 0, y[end])
                
    # Backtrace
    for k = length(y)-1:-1:1                                        # Algorithm 1 line 6
        β[k] = min(bp[2, k], max(β[k+1] - γ[k+1], bp[1, k]))        # Algorithm 1 line 7
    end
    pflsa
end


scale(x) = (x .+ 1)/2

function signal!(T)
    bias_g = [sin(t) for t in collect(range(0, 2*π, length = T))]
    p, signal, p_movement = zeros(3, T), zeros(1, T), 1
    p[1, :] = scale(-bias_g) * p_movement
    p[2, :] = ones(T) .- p_movement
    p[3, :] = scale(bias_g) * p_movement
    signal = [findfirst(rand() .< cumsum(p[:, i])) for i = 1:T]
    bias_g, float(signal .- 2.0)
end

function process(str, i)
    turn_dir = h5read((string(str[i], ".h5")), "tail3")
    heat = h5read((string(str[i], ".h5")), "laser_state")
    heat = heat[1] == 1 ? vcat([0], heat) : heat
    heat = heat[end] == 1 ? vcat(heat, [0]) : heat;
    heat_on_idx = findall(diff(heat) .> 0)
    heat_off_idx = findall(diff(heat) .< 0)
    heat_idx = findall(heat .== 1)
    turn_idx = findall(turn_dir .!= 0)
    turn_all = turn_dir[turn_idx]
    heat_turn_idx = heat_idx[findall(turn_dir[heat_idx] .!= 0)] 
    heat_turn_dir = turn_dir[heat_turn_idx]
    heat_idx, turn_idx, turn_all, heat_turn_idx, heat_turn_dir
end

function model(str, i, λ_int::StepRangeLen, opacity)
    heat_idx, turn_idx, turn_all, heat_turn_idx, heat_turn_dir = process(str, i)
    y = float(heat_turn_dir)
    γ = zeros(length(y)-1)
    bias_opt = zeros(length(λ_int), length(y))
    for i = 1:length(λ_int)
        pflsa = fit(PFusedLasso, y, γ, λ_int[i])
        bias_opt[i, :] = pflsa.β
    end
    figure(figsize = (16, 2))
    plot(y.*1.1, "k.", markersize = 2)
    for i = 1:length(λ_int)
        plot(bias_opt[i, :], "m", alpha = opacity, linewidth = 0.5)
    end
    xlim(-1, length(y) + 1)
    ylim(-1.3, 1.3)
end

function toy_data(n_transitions, T)
    unique_p = rand(Float64, n_transitions)
    t_transition = sort([rand(1:T) for i in 1:n_transitions])
    t_transition[end] = T
    p = []
    x = []
    t = 1
    for i in 1:length(t_transition)
        while t <= t_transition[i]
            push!(p, unique_p[i])
            if rand(Float64) < p[i]
                push!(x, -1)
            else
                push!(x, 1)
            end
            t += 1
        end
    end
    p, x
end

n_transitions, T = 5, 100
bias, y = toy_data(n_transitions, T)
y = convert(Vector{Float64}, y)
γ = fill(0.0, T-1);
c = fill(4.0, T);

pflsa = fit(PFusedLasso, y, γ, 3);
println(bias)
println(y)
println(pflsa.β)