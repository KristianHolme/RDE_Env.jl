struct TimeDiffNormReward{T <: AbstractFloat} <: AbstractRDEReward
    threshold::T
    threshold_reward::T
end

function TimeDiffNormReward{T}(; threshold::T = 1.1f0, threshold_reward::T = 0.3f0) where {T <: AbstractFloat}
    return TimeDiffNormReward{T}(threshold, threshold_reward)
end
function TimeDiffNormReward(; threshold::T = 1.1f0, threshold_reward::T = 0.3f0) where {T <: AbstractFloat}
    return TimeDiffNormReward{T}(threshold, threshold_reward)
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TimeDiffNormReward{T}) where {T, A, O, R, V, OBS}
    if isnothing(env.prob.sol)
        return zero(T)
    end
    N = env.prob.params.N
    us = map(v -> v[1:N], env.prob.sol.u)
    n = length(us)
    diff_norms = zeros(T, Int((n^2 - n) // 2))
    ind = 1
    ft_us = rfft.(us)
    abs_ft_us = map(v -> abs.(v), ft_us)
    for i in 1:n, j in (i + 1):n
        diff_norms[ind] = RDE.turbo_diff_norm(abs_ft_us[i], abs_ft_us[j]) / sqrt(N)
        ind += 1
    end
    max_norm = RDE.turbo_maximum(diff_norms)
    a = log(rt.threshold_reward) / rt.threshold
    return exp(a * max_norm)
end

reward_value_type(::Type{T}, ::TimeDiffNormReward) where {T} = T
