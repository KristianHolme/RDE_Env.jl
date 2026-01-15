"""
    NoGoal

Default goal strategy that performs no updates.
"""
struct NoGoal <: AbstractGoalStrategy end

function on_reset!(::NoCache, ::NoGoal, ::AbstractRDEEnv)
    return nothing
end
