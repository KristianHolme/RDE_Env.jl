function sigmoid(x::AbstractFloat)
    return one(x)/(one(x)+exp(-x))
end

function reward_sigmoid(x::AbstractFloat)
    a = 5*one(x)
    b = convert(typeof(x), 0.5)
    return sigmoid(a*(x-b))
end

function sigmoid_to_linear(x::AbstractFloat)
    cutoff = convert(typeof(x), 0.5)
    if x < cutoff
        return reward_sigmoid(x)
    else
        return reward_sigmoid(cutoff)/cutoff * x
    end
end

function linear_to_sigmoid(x::AbstractFloat)
    cutoff = convert(typeof(x), 0.5)
    if x < cutoff
        return x
    else
        return reward_sigmoid(x)
    end
end
