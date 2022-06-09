using PyPlot, Optim, MLBase

scale(x) = (x .+ 1) / 2
sign!(x) = x != 0.0 ? sign(x) : 1.0
ind(x, y) = x .== y ? 1 : 0

function signal!(T)
    
    bias_g = [sin(t) for t in linspace(0, 2*π, T)]
    p, signal, p_movement = zeros(3, T), zeros(1, T), 1
    p[1, :] = scale(-bias_g) * p_movement
    p[2, :] = ones(T) .- p_movement
    p[3, :] = scale(bias_g) * p_movement
    signal = [findfirst(rand() .< cumsum(p[:, i])) for i = 1:T]
    bias_g, signal
end

function p_emission(bias, signal)

    T = length(bias)
    p, ind, p_movement = zeros(3, T), zeros(3, T), .1
    bias = clamp.(bias, -0.9999999999,0.9999999999)
    
    p[1, :] = scale(-bias) * p_movement
    p[2, :] = ones(T) .- p_movement
    p[3, :] = scale(bias) * p_movement
    
    [ind[i, find(signal .== i)] .= 1 for i = 1:3]

    E = p .* ind
    E = sum(E, 1)
    E
end

function d_p!(x, signal)

    T = length(x)
    p, ind = zeros(3, T), zeros(3, T)
    x = clamp.(x, -0.9999999999, 0.9999999999)
    
    p[1, :] = 1 ./ (x .- 1)
    p[2, :] = zeros(T)
    p[3, :] = 1 ./ (x .+ 1)
    
    [ind[i, find(signal .== i)] .= 1 for i = 1:3]

    d = p .* ind
    sum(d, 1)
end

function fused_elastic_net(bias, signal, λ, µ)
    
    T = length(signal)
    p_total = sum(log.(p_emission(bias, signal))) - λ * sum(abs, (diff(bias))) - µ * sum(abs2, (diff(bias)))
    -p_total
end

function mixed_finite_g!(G, x, signal, λ, µ)

    ϵ, T = (1.0e-17)^(1/3), length(signal)

    d_p = d_p!(x, signal)
    p = p_emission(x, signal)
    d_log_likelihood = d_p ./ p * 1/4
    
    for i = 2:T-1
        d_sparsity = abs(x[i] + ϵ - x[i - 1]) - abs(x[i] - ϵ - x[i - 1]) + abs(x[i + 1] - x[i] - ϵ) - abs(x[i + 1] - x[i] + ϵ)
        d_int_sparsity = 4 * x[i] - 2 * (x[i + 1] + x[i - 1])
        G[i] = (-d_log_likelihood[i] + λ * d_sparsity / (2 * ϵ) + µ * d_int_sparsity)
    end
    
    G[1] = (-d_log_likelihood[1] + λ * (abs(x[2] - x[1] - ϵ) - abs(x[2] - x[1] + ϵ)) / (2 * ϵ) + µ * 2 * (x[1] - x[2])) 
    G[T] = (-d_log_likelihood[T] + λ * (abs(x[T] + ϵ - x[T - 1]) - abs(x[T] - ϵ - x[T - 1])) / (2 * ϵ) + µ * 2 * (x[T] - x[T - 1]))
    G
end


function finite_g!(G, x, signal, λ, µ)    
    
    ϵ, T = (1.0e-17)^(1/3), length(signal)
    d_log_likelihood = [log(p_emission(x + ϵ, signal)[i]) - log(p_emission(x - ϵ, signal)[i]) for i = 1:T]
    
    for i = 2:T-1
        d_sparsity = abs(x[i] + ϵ - x[i - 1]) - abs(x[i] - ϵ - x[i - 1]) + abs(x[i + 1] - x[i] - ϵ) - abs(x[i + 1] - x[i] + ϵ)
        d_int_sparsity = abs2(x[i] + ϵ - x[i - 1]) - abs2(x[i] - ϵ - x[i - 1]) + abs2(x[i + 1] - x[i] - ϵ) - abs2(x[i + 1] - x[i] + ϵ)
        G[i] = (-d_log_likelihood[i] + λ * d_sparsity + µ * d_int_sparsity) / (2 * ϵ)
    end
    
    G[1] = (-d_log_likelihood[1] + λ * (abs(x[2] - x[1] - ϵ) - abs(x[2] - x[1] + ϵ)) + µ * (abs2(x[2] - x[1] - ϵ) - abs2(x[2] - x[1] + ϵ))) / (2 * ϵ) 
    G[T] = (-d_log_likelihood[T] + λ * (abs(x[T] + ϵ - x[T - 1]) - abs(x[T] - ϵ - x[T - 1])) + µ * (abs2(x[T] + ϵ - x[T - 1]) - abs2(x[T] - ϵ - x[T - 1]))) / (2 * ϵ)
    G
end

function g!(G, x, signal, λ, µ)
    T = length(signal)
    d_p = d_p!(x, signal)
    for i = 2:T-1
        d_sparsity = sign!(x[i] - x[i - 1]) - sign!(x[i + 1] - x[i])
        d_int_sparsity = 4 * x[i] - 2 * (x[i + 1] + x[i - 1])
        G[i] = -d_p[i] + λ * d_sparsity + µ * d_int_sparsity
    end
    G[1] = -d_p[1] + λ * sign!(x[1] - x[2]) + µ * 2 * (x[1] - x[2])
    G[T] = -d_p[T] + λ * sign!(x[T] - x[T - 1]) + µ * 2 * (x[T] - x[T - 1])
    G
end

function h!(H, x, signal, µ)
    h = zeros((T, T))
    for i = 2:T-1
        h[i, i] = ind(1, signal[i]) / (1 - x[i])^2 - ind(3, signal[i]) / (1 + x[i])^2 + 4*µ
    end
    h[2:T + 1:T^2 - 1] = -2*µ
    h[T + 1:T + 1:T^2 - 1] = -2*µ
    h[1, 1], h[T, T] = 2*µ, 2*µ
    h
end

function opt!(signal, λ, µ)

    T = length(signal)
    h(bias) = fused_elastic_net(bias, signal, λ, µ)
    g!!(G, bias) = g!(G, bias, signal, λ, µ)
    h!!(H, bias) = h!(H, bias, signal, µ)

    Λ_0 = fill(-1.0, T)
    Λ_1 = fill(1.0, T)
    x_0 = rand(T)
    inner_optimizer = BFGS()
    res = optimize(h, g!!, h!!, Λ_0, Λ_1, x_0, Fminbox(inner_optimizer))
    bias_opt = Optim.minimizer(res)

    bias_opt, -fused_elastic_net(bias_opt, signal, λ, µ)
end

function fit!(signal)
    λ_grid = ("λ", [13.4])
    µ_grid = ("µ", [1, 3, 5])
    best_model, best_cfg, best_score = gridtune(
        (λ_grid, µ_grid) -> opt!(signal, λ_grid, µ_grid),
        m -> m[2],
        λ_grid, µ_grid, ord = Forward, verbose = true);
    λ_opt, µ_opt = best_cfg
    bias_opt, log_p = opt!(signal, λ_opt, µ_opt)
    bias_opt, log_p, λ_opt, µ_opt
end;

s = signal!(300)[1]
print(s)

bias_opt_0, log_p, λ_opt, µ_opt = fit!(signal)

figure(figsize = (16, 2))
plot(bias_g, "k", linewidth = 0.5)
plot(signal .- 2, "k.", markersize = 3)
plot(bias_opt_0, "m")
ylim(-1.3, 1.3)
title("log(p) = $log_p");

-fused_elastic_net(bias_opt[134], signal, 13.4, 3)

T = 300
bias_g, signal = signal!(T)
bias_opt, log_p = opt!(signal, 1, 1)

figure(figsize = (16, 2))
plot(bias_g, "k", linewidth = 0.5)
plot(signal .- 2, "k.", markersize = 3)
plot(bias_opt_0, "m", linewidth = 0.3)
ylim(-1.3, 1.3)
title("log(p) = $log_p")

λ, µ = 5, 6

T = length(signal)
h(x) = fused_elastic_net(x, signal, λ, µ)
g!(G, x) = mixed_finite_g!(G, x, signal, λ, µ)

Λ_0 = fill(-1.0, T)
Λ_1 = fill(1.0, T)
x_0 = clamp.(bias_opt .+ rand(T)*0.1, -0.99, 0.99)
inner_optimizer = BFGS()
res = optimize(h, g!, Λ_0, Λ_1, x_0, Fminbox(inner_optimizer))
bias_opt_0 = Optim.minimizer(res)
log_p_0 = -fused_elastic_net(bias_opt_0, signal, λ, µ)
bias_opt_0, log_p_0

figure(figsize = (16, 2))
plot(bias_g, "k", linewidth = 0.5)
plot(signal .- 2, "k.", markersize = 3)
plot(bias_opt_0, "m")
plot(bias_opt, "m")
ylim(-1.3, 1.3)
title("log(p) = $log_p_0");

function iter_fit!(signal)

    Λ = 0.1:0.1:20
    bias_opt = Array{Array{Float64,1}}(length(Λ))
    
    for i = 1:length(Λ)
        print(i)
        λ_grid = ("λ", [Λ[i]])
        µ_grid = ("µ", [1, 3, 5])
        best_model, best_cfg, best_score = gridtune(
            (λ_grid, µ_grid) -> opt!(signal, λ_grid, µ_grid),
            m -> m[2],
            λ_grid, µ_grid, ord = Forward, verbose = false);
        λ_opt, µ_opt = best_cfg
        bias_opt[i] = opt!(signal, λ_opt, µ_opt)[1]
    end

    bias_opt
end

bias_opt = iter_fit!(signal)

figure(figsize = (16, 2))
plot(bias_g, "k", linewidth = 0.5)
plot((signal .- 2)*1.1, "k.", markersize = 3)
for i = 1:length(bias_opt)
    plot(bias_opt[i], "m", linewidth = 0.1)
end
ylim(-1.3, 1.3)

figure(figsize = (16, 2))
plot(bias_g, "k", linewidth = 0.5)
plot((signal .- 2)*1.1, "k.", markersize = 3)
plot(bias_opt[171], "m", linewidth = 0.3)
ylim(-1.3, 1.3)

srand(1)
# f = rand(10)
# s_test = signal!(T)[2]
G = zeros(10)
print(Calculus.gradient(x -> fused_elastic_net(x, s_test, 1, 1), f), "       ")
print(g!(G, f, s_test, 1, 1))


f = rand(10)
s_test = signal!(10)[2]
H = zeros((10, 10))
print(Calculus.hessian(x -> fused_elastic_net(x, s_test, 0, 1), f), "       ")

print(h!(H, f, signal, 1))