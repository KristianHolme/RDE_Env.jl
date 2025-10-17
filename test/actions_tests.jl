@testitem "direct action" begin
    actions = [0.0f0, 1.0f0, -1.0f0]
    known_targets = [0.6f0, 1.2f0, 0.0f0]
    u_p_max = 1.2f0
    c_prev = rand()
    for i in eachindex(actions)
        action = actions[i]
        known_target = known_targets[i]
        target = RDE_Env.direct_action_to_control(action, c_prev, u_p_max, 0.0f0)
        @test target = known_target
    end
end
