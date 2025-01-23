mutable struct CompositeReward <: AbstractRDEReward
    target_shock_count::Int
    cache::Vector{Float32}
    lowest_action_magnitude_reward::Float32 #reward will be \in [lowest_action_magnitude_reward, 1]
    function CompositeReward(;target_shock_count::Int=4, lowest_action_magnitude_reward::Float32=1f0)
        return new(target_shock_count, zeros(Float32, 512), lowest_action_magnitude_reward)
    end
end

function Base.show(io::IO, rt::CompositeReward)
    print(io, "CompositeReward(target_shock_count=$(rt.target_shock_count), lowest_action_magnitude_reward=$(rt.lowest_action_magnitude_reward))")
end

function set_reward!(env::AbstractRDEEnv{T}, rt::CompositeReward) where T
    N = env.prob.params.N
    dx = env.prob.x[2] - env.prob.x[1]
    L = env.prob.params.L
    u = env.state[1:N]
    target_shock_count = rt.target_shock_count  
    
    target_span = 2f0 .- 0.3f0.*target_shock_count
    span = maximum(u) - minimum(u)
    span_reward = 1f0 - (max(target_span - span, 0f0)/target_span)

    abs_span_punishment_threshold = 0.08f0
    abs_span_reward = RDE.smooth_g(span/abs_span_punishment_threshold)

    if target_shock_count > 1
        errs = zeros(T, target_shock_count-1)
        cache = rt.cache
        shift_steps = N ÷ target_shock_count
        for i in 1:(target_shock_count-1)
            cache .= u
            RDE.apply_periodic_shift!(cache, u, shift_steps * i)
            errs[i] = norm(u - cache)/sqrt(N)
        end
        maxerr = maximum(errs)
        periodicity_reward = 1f0 - (max(maxerr-0.08f0, 0f0)/sqrt(3f0))
    else
        periodicity_reward = 1f0
    end

    shock_inds = RDE.shock_indices(u, dx)
    shocks = T(length(shock_inds))
    shock_reward = 1f0 - (abs(shocks - target_shock_count)/target_shock_count)
    if shocks > 0
        # optimal_spacing = L/shocks
        optimal_spacing = L/target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = 1f0 - maximum(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
    else
        shock_spacing_reward = 1f0
    end

    @debug "span_reward: $span_reward" typeof(span_reward)
    @debug "abs_span_reward: $abs_span_reward" typeof(abs_span_reward)
    @debug "periodicity_reward: $periodicity_reward" typeof(periodicity_reward)
    @debug "shock_reward: $shock_reward" typeof(shock_reward)
    @debug "shock_spacing_reward: $shock_spacing_reward" typeof(shock_spacing_reward)

    reward = abs_span_reward*exp(span_reward + periodicity_reward + shock_reward + shock_spacing_reward - 4f0)
    if rt.lowest_action_magnitude_reward < 1f0
        action_magnitude_inv = 1f0 - maximum(abs.(env.cache.action))
        α = rt.lowest_action_magnitude_reward
        reward *= α + (1f0 - α)*action_magnitude_inv
    end
    env.reward = reward
    nothing
end


struct ConstantTargetReward <: AbstractRDEReward
    target::Float32
    function ConstantTargetReward(;target::Float32=0.64f0)
        return new(target)
    end
end

function set_reward!(env::AbstractRDEEnv{T}, rt::ConstantTargetReward) where T
    env.reward = -abs(rt.target - mean(env.prob.cache.u_p_current)) + T(1.0)
    nothing
end
