function reset_reward!(rt::AbstractRDEReward)
    return nothing
end

function set_reward!(env::AbstractRDEEnv, rt::AbstractRDEReward)
    env.reward = compute_reward(env, rt)
    return nothing
end

include("rewards/reward_utils.jl")
include("rewards/shock_span.jl")
include("rewards/shock_preserving.jl")
include("rewards/shock_preserving_symmetry.jl")
include("rewards/periodicity.jl")
include("rewards/composite.jl")
include("rewards/time_diff_norm.jl")
include("rewards/period_minimum.jl")
include("rewards/period_minimum_variation.jl")
include("rewards/multi_section.jl")
include("rewards/multi_section_period_minimum.jl")
include("rewards/wrappers.jl")
include("rewards/constant_target.jl")
