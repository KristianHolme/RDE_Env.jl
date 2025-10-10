struct ConstantTargetReward{T <: AbstractFloat} <: AbstractRDEReward
    target::T
    function ConstantTargetReward{T}(; target::T = T(0.64)) where {T <: AbstractFloat}
        return new{T}(target)
    end
    function ConstantTargetReward(; target::T = 0.64f0) where {T <: AbstractFloat}
        return ConstantTargetReward{T}(target = target)
    end
end

function Base.show(io::IO, rt::ConstantTargetReward)
    return print(io, "ConstantTargetReward(target=$(rt.target))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ConstantTargetReward)
    println(io, "ConstantTargetReward:")
    return println(io, "  target: $(rt.target)")
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ConstantTargetReward{T}) where {T, A, O, R, V, OBS}
    return -abs(rt.target - mean(env.prob.method.cache.u_p_current)) + one(T)
end
