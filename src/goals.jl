"""
    FixedTargetGoal(target_shock_count::Int)
A goal strategy that sets the target shock count to a fixed value. Uses NoCache.
"""
struct FixedTargetGoal <: AbstractGoalStrategy
    target_shock_count::Int
end

update_goal!(::NoCache, ::AbstractGoalStrategy, ::AbstractRDEEnv) = nothing
get_target_shock_count(goal_strat::FixedTargetGoal, ::AbstractRDEEnv) = goal_strat.target_shock_count
function set_target_shock_count!(::FixedTargetGoal, env::AbstractRDEEnv, v::Int)
    env.goal_strat = FixedTargetGoal(v)
    return nothing
end

"""
    RandomTargetGoal(min_value::Int = 1, max_value::Int = 4)
A goal strategy that sets the target shock count to a random value between min_value and max_value. Uses GoalCache.
"""
@kwdef struct RandomTargetGoal <: AbstractGoalStrategy
    min_value::Int = 1
    max_value::Int = 4
end

function initialize_cache(goal_strat::RandomTargetGoal, ::Int, ::Type{T}) where {T}
    return GoalCache(rand(goal_strat.min_value:goal_strat.max_value))
end

function update_goal!(cache::GoalCache, goal_strat::RandomTargetGoal, ::AbstractRDEEnv)
    cache.target_shock_count = rand(goal_strat.min_value:goal_strat.max_value)
    return nothing
end

function get_target_shock_count(goal_strat::RandomTargetGoal, env::AbstractRDEEnv)
    return env.cache.goal_cache.target_shock_count
end
function set_target_shock_count!(goal_strat::RandomTargetGoal, env::AbstractRDEEnv, v::Int)
    error("Cannot set target shock count with RandomTargetGoal, 
use FixedTargetGoal instead.
")
    return nothing
end

# ----------------------------------------------------------------------------
# EvalCycleTargetGoal
# ----------------------------------------------------------------------------
@kwdef struct EvalCycleTargetGoal <: AbstractGoalStrategy
    repetitions_per_config::Int = 4
end
mutable struct EvalCycleTargetCache <: AbstractGoalCache
    target_shocks::Vector{Int}
    current_config::Int
end

function initialize_cache(goal_strat::EvalCycleTargetGoal, ::Int, ::Type{T}) where {T}
    total_configs = 4 * 3 * goal_strat.repetitions_per_config
    target_shocks = Vector{Int}(undef, total_configs)
    i = 1
    target_shocks = repeat(1:4, inner = goal_strat.repetitions_per_config * 3)
    return EvalCycleTargetCache(target_shocks, 1)
end

function update_goal!(cache::EvalCycleTargetCache, ::EvalCycleTargetGoal, ::AbstractRDEEnv)
    cache.current_config += 1
    if cache.current_config > length(cache.target_shocks)
        cache.current_config = 1
    end
    return nothing
end

function get_target_shock_count(::EvalCycleTargetGoal, env::AbstractRDEEnv)
    return env.cache.goal_cache.target_shocks[env.cache.goal_cache.current_config]
end
