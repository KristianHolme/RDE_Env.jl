module RDE_EnvPOMDPsExt

using RDE_Env
using POMDPs

function POMDPs.initialobs(RLEnvPOMDP, s)
    return [_observe(RLEnvPOMDP.env)]
end

end