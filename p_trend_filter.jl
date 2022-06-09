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

StatsBase.coef(pflsa::PFusedLasso) = pflsa.β

struct NormalCoefs{T}
    lin::T
    quad::T

    NormalCoefs{T}(lin::Real) where {T} = new(lin, 0)
    NormalCoefs{T}(lin::Real, quad::Real) where {T} = new(lin, quad)
end
+(a::NormalCoefs{T}, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a.lin+b.lin, a.quad+b.quad)
-(a::NormalCoefs{T}, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a.lin-b.lin, a.quad-b.quad)
+(a::NormalCoefs{T}, b::Real) where {T} = NormalCoefs{T}(a.lin+b, a.quad)
-(a::NormalCoefs{T}, b::Real) where {T} = NormalCoefs{T}(a.lin-b, a.quad)
*(a::Real, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a*b.lin, a*b.quad)

# Implements Algorithm 2 lines 8 and 19
solveforb(a::NormalCoefs{T}, λ::Real) where {T} = (λ - a.lin)/a.quad

b_lt(a::NormalCoefs{T}, λ::Real, x::Real) where {T} = (λ - a.lin)/a.quad < x
b_gt(a::NormalCoefs{T}, λ::Real, x::Real) where {T} = (λ - a.lin)/a.quad > x

struct Knot{T,S}
    pos::T
    coefs::S
    sign::Int8
end

struct FusedLasso{T,S} <: RegressionModel
    β::Vector{T}              # Coefficients
    knots::Vector{Knot{T,S}}  # Active knots
    bp::Matrix{T}             # Backpointers
end

function StatsBase.fit(::Type{FusedLasso}, y::AbstractVector{T}, c::AbstractVector{T}, λ::Real; dofit::Bool=true) where T
    S = NormalCoefs{T}
    flsa = FusedLasso{T,S}(Array{T}(undef, length(y)), Array{Knot{T,S}}(undef, 2), Array{T}(undef, 2, length(y)-1))
    dofit && fit!(flsa, y, c, λ)
    flsa
end

function StatsBase.fit!(flsa::FusedLasso{T,S}, y::AbstractVector{T}, c::AbstractVector{T}, λ::Real) where {T,S}
    β = flsa.β
    knots = flsa.knots
    bp = flsa.bp

    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))
    length(c) == length(y) || throw(ArgumentError("coefficient vector size $(length(c)) does not match input size $(length(y))"))
    c[c .<= 1.0e-10] .= 1.0e-10 #easiest way to prevent division by zero while retaining soluion accuracy

    resize!(knots, 2)
    knots[1] = Knot{T,S}(-Inf, S(0), 1)
    knots[2] = Knot{T,S}(Inf, S(0), -1)

    # Algorithm 1 lines 2-5
    @inbounds for k = 1:length(y)-1
        t1 = 0
        t2 = 0
        
        aminus = NormalCoefs{T}(y[end]/(100*c[k]), -1/(100*c[k]))
        for outer t1 = 1:length(knots)-1                         
            knot = knots[t1]
            aminus += knot.sign*knot.coefs
            b_lt(aminus, λ, knots[t1+1].pos) && break
        end
        bminus = solveforb(aminus, λ)

        aplus = NormalCoefs{T}(y[end]/(100*c[k]), -1/(100*c[k]))
        t2 = length(knots)
        while t2 >= 2                
            knot = knots[t2]
            aplus -= knot.sign*knot.coefs                  
            b_gt(aplus, -λ, knots[t2-1].pos) && break
            t2 -= 1
        end
        bplus = solveforb(aplus, -λ)

        if t2 > t1
            deleteat!(knots, 1:t1)
            deleteat!(knots, t2-t1:length(knots))
        else
            resize!(knots, 0)
        end
        pushfirst!(knots, Knot{T,S}(bminus, aminus-λ, 1))
        pushfirst!(knots, Knot{T,S}(-Inf, S(λ), 1))
        push!(knots, Knot{T,S}(bplus, aplus+λ, -1))
        push!(knots, Knot{T,S}(Inf, S(-λ), -1))
        bp[1, k] = bminus
        bp[2, k] = bplus
    end

    # Algorithm 1 line 6
    aminus = NormalCoefs{T}(y[end]/(100*c[end]), -1/(100*c[end]))
    for t1 = 1:length(knots)
        knot = knots[t1]
        aminus += knot.sign*knot.coefs
        b_lt(aminus, 0, knots[t1+1].pos) && break
    end
    β[end] = solveforb(aminus, 0)

    # Backtrace
    for k = length(y)-1:-1:1                        # Algorithm 1 line 6
        β[k] = min(bp[2, k], max(β[k+1], bp[1, k])) # Algorithm 1 line 7
    end
    flsa
end

StatsBase.coef(flsa::FusedLasso) = flsa.β

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
# flsa = fit(FusedLasso, y, c, 0.3)

# figure(figsize = (16, 2))
# plot(y.*1.1, "k.", markersize = 2)
# plot(bias, "k", linewidth = 0.5)
# plot(pflsa.β, "m", linewidth = 0.5)
# plot(γ, "k", linewidth = 0.5)
# plot(flsa.β, "m", linewidth = 0.5)
# ylim(-1.3, 1.3)

# function update_α!(u::Vector{T}, λ::Real) where T
#     n_iter = 100
#     c = ones(length(u))
#     save_β = fill(u, 1)
#     oldobj = obj = Inf
#     tol = 1.0e-4 # level of accuracy
#     for i = 1:n_iter
#         flsa = fit(FusedLasso, u, c, λ)
#         c = abs.(u .- flsa.β)
        
#         # convergence check
#         oldobj = obj
#         obj = sum(abs, u .- flsa.β) - λ*sum(abs, diff(flsa.β))
#         print(abs(oldobj - obj) === NaN, "   ")
#         (abs(oldobj - obj) === NaN || (abs(oldobj - obj) < abs(obj * tol))) && break
        
#         append!(save_β, [flsa.β])
#     end
# #     print(length(save_β), "  ")
#     save_β[end]
# end;
# function update_α!(u::Vector{T}, λ::Real)
#     n_iter = 100
#     c = ones(length(u))
#     save_β = fill(randn(length(u)), 1)
#     a = zeros(0)
#     obj = Inf
#     tol = 0.1 # level of accuracy
#     for i = 1:n_iter
#         flsa = fit(FusedLasso, u, c, λ)
#         c = abs.(u .- flsa.β)
#         c[c .<= 1.0e-10] .= 1.0e-10
#         obj = sum(c) + λ * sum(abs, diff(flsa.β))
#         maj = 1/2 * sum(abs, (u .- save_β[i]).^2 ./ c) + 1/2 * sum(abs, c) + λ * sum(abs, diff(save_β[i]))
#         append!(a, abs(obj - maj))
#         (isnan(abs(maj - obj)) || (abs(maj - obj) < abs(obj * tol))) && break
        
#         append!(save_β, [flsa.β])
#     end
#     save_β, a
# end;

# using LinearAlgebra, SparseArrays, StatsBase
# using DSP: filt!

# macro extractfields(from, fields...)
#     esc(Expr(:block, [:($(fields[i]) = $(from).$(fields[i])) for i = 1:length(fields)]...))
# end

# struct DifferenceMatrix{T} <: AbstractMatrix{T}
#     k::Int
#     n::Int
#     b::Vector{T}                  # Coefficients for mul!
#     si::Vector{T}                 # State for mul!/At_mul_B!

#     function DifferenceMatrix{T}(k, n) where T
#         n >= 2*k+2 || throw(ArgumentError("signal must have length >= 2*order+2"))
#         b = T[ifelse(isodd(i), -1, 1)*binomial(k+1, i) for i = 0:k+1]
#         new(k, n, b, zeros(T, k+1))
#     end
# end

# Base.size(K::DifferenceMatrix) = (K.n-K.k-1, K.n)

# Multiply by difference matrix by filtering
# function LinearAlgebra.mul!(out::AbstractVector, K::DifferenceMatrix, x::AbstractVector, α::Real=1)
#     length(x) == size(K, 2) || throw(DimensionMismatch("length(x) == $(length(x)) != $(size(K, 2)) == size(K, 2)"))
#     length(out) == size(K, 1) || throw(DimensionMismatch("length(x) == $(length(x)) != $(size(K, 1)) == size(K, 1)"))
#     b = K.b
#     si = fill!(K.si, 0)
#     silen = length(b)-1
#     @inbounds for i = 1:length(x)
#         xi = x[i]
#         val = si[1] + b[1]*xi
#         for j=1:(silen-1)
#             si[j] = si[j+1] + b[j+1]*xi
#         end
#         si[silen] = b[silen+1]*xi
#         if i > silen
#             out[i-silen] = α*val
#         end
#     end
#     out
# end
# *(K::DifferenceMatrix, x::AbstractVector) = mul!(similar(x, size(K, 1)), K, x)

# function LinearAlgebra.mul!(out::AbstractVector, K::Adjoint{<:Any,<:DifferenceMatrix}, x::AbstractVector, α::Real=1)
#     length(x) == size(K.parent, 1) || throw(DimensionMismatch("length(x) == $(length(x)) != $(size(K.parent, 1)) == size(K.parent, 1)"))
#     length(out) == size(K.parent, 2) || throw(DimensionMismatch("length(out) == $(length(out)) != $(size(K.parent, 2)) == size(K.parent, 2)"))
#     b = K.parent.b
#     si = fill!(K.parent.si, 0)
#     silen = length(b)-1
#     isodd(silen) && (α = -α)
#     n = length(x)
#     @inbounds for i = 1:n
#         xi = x[i]
#         val = si[1] + b[1]*xi
#         for j=1:(silen-1)
#             si[j] = si[j+1] + b[j+1]*xi
#         end
#         si[silen] = b[silen+1]*xi
#         out[i] = α*val
#     end
#     @inbounds for i = 1:length(si)
#         out[n+i] = α*si[i]
#     end
#     out
# end
# LinearAlgebra.mul!(out::AbstractVector, K::Hermitian{<:Any,<:DifferenceMatrix}, x::AbstractVector, α::Real=1) = mul!(out, K.parent', x)
# *(K::Adjoint{<:Any,<:DifferenceMatrix}, x::AbstractVector) = mul!(similar(x, size(K.parent, 2)), K, x)
# *(K::Hermitian{<:Any,<:DifferenceMatrix}, x::AbstractVector) = mul!(similar(x, size(K.parent, 2)), K.parent', x)

# # Product with self, efficiently
# function *(K::Adjoint{<:Any,<:DifferenceMatrix}, K2::DifferenceMatrix)
#     K.parent === K2 || error("matrix multiplication only supported with same difference matrix")
#     computeDtD(K.parent.b, K.parent.n)
# end
# *(K::Hermitian{<:Any,<:DifferenceMatrix}, K2::DifferenceMatrix) = *(K.parent, K2::DifferenceMatrix)

# function computeDtD(c, n)
#     k = length(c) - 2
#     sgn = iseven(k)
#     cc = zeros(eltype(c), 2*length(c)-1)
#     for i = 1:length(c)
#         cc[i] = sgn ? -c[i] : c[i]
#     end
#     filt!(cc, c, [one(eltype(c))], cc)
#     sides = zeros(eltype(c), 2*length(c)-2, length(c)-1)
#     for j = 1:length(c)-1
#         for i = 1:j
#             sides[i, j] = sgn ? -c[i] : c[i]
#         end
#     end
#     filt!(sides, c, [one(eltype(c))], sides)
#     colptr = Vector{Int}(undef, n+1)
#     rowval = Vector{Int}(undef, (k+2)*(n-k-1)+(k+1)*n)
#     nzval = Vector{Float64}(undef, (k+2)*(n-k-1)+(k+1)*n)
#     idx = 1
#     for i = 1:k+1
#         colptr[i] = idx
#         for j = 1:k+i+1
#             rowval[idx+j-1] = j
#             nzval[idx+j-1] = sides[k+2+i-j, i]
#         end
#         idx += k+i+1
#     end
#     for i = k+2:n-(k+1)
#         colptr[i] = idx
#         for j = 1:length(cc)
#             rowval[idx+j-1] = i-k+j-2
#             nzval[idx+j-1] = cc[j]
#         end
#         idx += length(cc)
#     end
#     for i = k+1:-1:1
#         colptr[n-i+1] = idx
#         for j = 1:i+k+1
#             rowval[idx+j-1] = n-k-1-i+j
#             nzval[idx+j-1] = sides[j, i]
#         end
#         idx += i+k+1
#     end
#     colptr[end] = idx
#     return SparseMatrixCSC(n, n, colptr, rowval, nzval)
# end

# # Soft threshold
# S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)

# mutable struct TrendFilter{T,pS,S,VT}
#     Dkp1::DifferenceMatrix{T}                # D(k+1)
#     Dk::DifferenceMatrix{T}                  # D(k)
#     DktDk::SparseMatrixCSC{T,Int}            # Dk'Dk
#     β::Vector{T}                             
#     u::Vector{T}
#     γ::Vector{T}
#     c::Vector{T}
#     Dkβ::Vector{T}                           # Temporary storage for D(k)*β
#     Dkp1β::VT                                # Temporary storage for D(k+1)*β (aliases Dkβ)
#     pflsa::PFusedLasso{T,pS}
#     flsa::FusedLasso{T,S}                    # Fused lasso model
#     niter::Int                               # Number of ADMM iterations
# end

# function StatsBase.fit(::Type{TrendFilter}, y::AbstractVector{T}, λ; dofit::Bool=true, args...) where T
#     order = 1
#     Dkp1 = DifferenceMatrix{T}(order, length(y))
#     Dk = DifferenceMatrix{T}(order-1, length(y))
#     β = zeros(T, length(y))
#     u = zeros(T, size(Dk, 1))
#     γ = zeros(T, size(Dk, 1))
#     c = zeros(T, size(Dk, 1))
#     Dkβ = zeros(T, size(Dk, 1))
#     Dkp1β = view(Dkβ, 1:size(Dkp1, 1))
#     tf = TrendFilter(Dkp1, Dk, Dk'Dk, β, u, γ, c, Dkβ, Dkp1β, fit(PFusedLasso, β, γ, λ; dofit=false), fit(FusedLasso, Dkβ, c, λ; dofit=false), -1)
#     dofit && fit!(tf, y, λ; args...)
#     return tf
# end

# function StatsBase.fit!(tf::TrendFilter{T}, y::AbstractVector{T}, λ::Real; niter=30, tol=1e-6, ρ=λ) where T
    
#     @extractfields tf Dkp1 Dk DktDk β u γ c Dkβ Dkp1β pflsa flsa
#     length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

#     # Reuse this memory
#     ρDtαu = β
#     αpu = Dkβ

#     fact = cholesky(sparse(1.0I, size(Dk, 2), size(Dk, 2)) + ρ*DktDk)
#     α = coef(flsa)
#     fill!(α, 0.0)
#     β = coef(pflsa)
#     β = fill!(β, 0.0)
    
#     oldobj = obj = Inf
#     local iter
#     for outer iter = 1:niter
#         # update β
#         γ .= α .+ u
# #         print("γ: ", γ, "   ")
#         pflsa = fit(PFusedLasso, y, γ, ρ)
#         β = pflsa.β
# #         print("β: ", β, "   ")

#         # Build in convergence check here

#         # update α
#         mul!(Dkβ, Dk, β)
#         u .= Dkβ .- u
#         α = update_α!(u, λ/ρ)
#         print(sum(abs, diff(β) .- α), ", ")
# #         print(broadcast!(-, u, α, u))
#         # update u; u actually contains Dβ - u
#         u .= α .- u
# #         print("u: ", u, "   ")
#     end
    
#     # Save coefficients
#     tf.β = β
#     tf.niter = iter
#     tf
# end

# StatsBase.coef(tf::TrendFilter) = tf.β

# T = 200
# bias, y = signal!(T);

# tf = fit(TrendFilter, y, 5);
# pflsa = fit(PFusedLasso, y, zeros(length(y)-1), 5);

# figure(figsize = (16, 2))
# # plot(bias, "k", linewidth = 0.5)
# plot(y.*1.1, "k.", markersize = 2)
# plot(tf.β, "m", linewidth = 0.5)
# plot(pflsa.β, "k", linewidth = 0.5)
# # plot(coef(b), "m", linewidth = 0.5)
# xlim(-1, length(y) + 1)
# ylim(-1.3, 1.3)

# figure(figsize = (16, 2))
# plot(tf.γ, "m", linewidth = 0.5);

# λ = 1.0
# ρ = λ
# α = zeros(length(y)-1)
# u = zeros(length(y)-1)
# γ = zeros(length(y)-1)
# Dkβ = zeros(length(y)-1)
# Dk = DifferenceMatrix{Float64}(0, length(y));

# for iter = 1:3
#     γ .= α .+ u
# #     print("γ: ", γ, "   ")
#     pflsa = fit(PFusedLasso, y, γ, ρ)
#     β = pflsa.β
#     print(diff(β))
# #     print("β: ", β, "   ")


#     mul!(Dkβ, Dk, β)
#     u .= Dkβ .- u
#     print("u: ", u, "  ")
#     α = update_α!(u, λ/ρ)
# #     print("α: ", α, "  ")
# #     print("Db:", diff(β))
# #     print(sum(diff(β) .- α), ", ")
# #   print(broadcast!(-, u, α, u))
#     u .= α .- u
# #         print("u: ", u, "   ")
# end


