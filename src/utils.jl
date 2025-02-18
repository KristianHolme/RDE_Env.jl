function sigmoid(x::AbstractFloat)
    return one(x)/(one(x)+exp(-x))
end

function reward_sigmoid(x::AbstractFloat)
    a = 5*one(x)
    b = convert(typeof(x), 0.5)
    return sigmoid(a*(x-b))
end

function sigmoid_to_linear(x::AbstractFloat)
    if x < convert(typeof(x), 0.2)
        return reward_sigmoid(x)
    else
        return x
    end
end
