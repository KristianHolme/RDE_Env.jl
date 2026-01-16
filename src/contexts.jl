"""
    NoContextStrategy

Default context strategy that performs no updates.
"""
struct NoContextStrategy <: AbstractContextStrategy end

function on_reset!(::NoCache, ::NoContextStrategy, ::AbstractRDEEnv)
    return nothing
end

function on_step!(::NoCache, ::NoContextStrategy, ::AbstractRDEEnv)
    return nothing
end
