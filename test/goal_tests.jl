using Test
using RDE_Env

@testitem "FixedTargetGoal Basic Functionality" begin
    using RDE
    goal_strat = FixedTargetGoal(3)
    env = RDEEnv(RDE.RDEParam{Float32}(N = 32), goal_strat = goal_strat)
    @test goal_strat.target_shock_count == 3

    # Test get_target_shock_count
    @test get_target_shock_count(goal_strat, env) == 3
end

@testitem "RandomTargetGoal Basic Functionality" begin
    goal_strat = RandomTargetGoal(min_value = 1, max_value = 4)
    @test goal_strat.min_value == 1
    @test goal_strat.max_value == 4

    # Test cache initialization
    cache = initialize_cache(goal_strat, 32, Float32)
    @test cache isa RDE_Env.GoalCache
    @test 1 ≤ cache.target_shock_count ≤ 4
end

@testitem "EvalCycleTargetGoal Initialization" begin
    goal_strat = EvalCycleTargetGoal(repetitions_per_config = 4)
    @test goal_strat.repetitions_per_config == 4

    # Test cache initialization
    cache = initialize_cache(goal_strat, 32, Float32)
    @test cache isa RDE_Env.EvalCycleTargetCache
    @test cache.current_config == 1
    @test length(cache.target_shocks) == 48  # 4 goals × 3 non-goal shocks × 4 repetitions

    # Verify the pattern: 12×1, 12×2, 12×3, 12×4
    expected_pattern = repeat(1:4, inner = 12)
    @test cache.target_shocks == expected_pattern
end

@testitem "EvalCycleTargetGoal Custom Repetitions" begin
    goal_strat = EvalCycleTargetGoal(repetitions_per_config = 2)
    cache = initialize_cache(goal_strat, 32, Float32)

    @test length(cache.target_shocks) == 24  # 4 goals × 3 non-goal shocks × 2 repetitions

    # Verify the pattern: 6×1, 6×2, 6×3, 6×4
    expected_pattern = vcat([repeat([i], 6) for i in 1:4]...)
    @test cache.target_shocks == expected_pattern
end

@testitem "EvalCycleTargetGoal Goal Sequence" begin
    using RDE
    goal_strat = EvalCycleTargetGoal(repetitions_per_config = 4)
    env = RDEEnv(RDE.RDEParam{Float32}(N = 32), goal_strat = goal_strat)
    # Test first 12 resets should all be goal 1
    expected_targets = repeat(1:4, inner = 12)

    goal_cache = env.cache.goal_cache
    @test goal_cache.target_shocks == expected_targets

    in_syncs = Bool[]
    for i in 1:48
        push!(in_syncs, goal_cache.current_config == i)
        _reset!(env)
    end

    #should have wrapped
    @test goal_cache.current_config == 1
end

@testitem "EvalCycleTargetGoal Wrap Around" begin
    using RDE
    goal_strat = EvalCycleTargetGoal(repetitions_per_config = 2)
    env = RDEEnv(RDE.RDEParam{Float32}(N = 32), goal_strat = goal_strat)
    goal_cache = env.cache.goal_cache

    # Go through all 24 configurations
    for i in 1:24
        RDE_Env.update_goal!(goal_cache, goal_strat, env)
    end

    # Should have wrapped around to config 1 (goal 1)
    @test goal_cache.current_config == 1
    @test get_target_shock_count(goal_strat, env) == 1
end

@testitem "EvalCycleTargetGoal and EvalCycleShockReset Synchronization" begin
    using RDE

    # Create both strategies with same repetitions
    goal_strat = EvalCycleTargetGoal(repetitions_per_config = 4)
    reset_strat = EvalCycleShockReset(4)

    env = RDEEnv(; goal_strat, reset_strategy = reset_strat)
    goal_cache = env.cache.goal_cache
    ok_steps = Bool[]
    for i in 1:48
        step_ok = reset_strat.current_config == i && goal_cache.current_config == i
        push!(ok_steps, step_ok)
        _reset!(env)
    end
    @test all(ok_steps)

    # Should have wrapped around to the beginning
    @test goal_cache.current_config == 1
    @test reset_strat.current_config == 1
    @test get_target_shock_count(goal_strat, env) == 1
end

@testitem "EvalCycleTargetGoal and EvalCycleShockReset Synchronization" begin
    using RDE
    reset_strat = EvalCycleShockReset(4)
    goal_strat = EvalCycleTargetGoal(repetitions_per_config = 4)
    env = RDEEnv(goal_strat = goal_strat, reset_strategy = reset_strat)
    # Test synchronized behavior over 48 resets
    expected_inits = vcat([repeat(setdiff(1:4, i), inner = 4) for i in 1:4]...)
    expected_targets = repeat(1:4, inner = 12)

    @test reset_strat.current_config == 1


    in_sync = [true]
    for i in 1:48
        reset_strat_current = reset_strat.current_config
        if reset_strat_current != i
            @error "Reset strategy current config $reset_strat_current does not match expected $i"
            in_sync[1] = false
        end
        if get_target_shock_count(goal_strat, env) != expected_targets[i]
            @error "i = $i, Target shock count $(get_target_shock_count(goal_strat, env)) does not match expected $(expected_targets[i])"
            in_sync[1] = false
        end
        if reset_strat.init_shocks[i] != expected_inits[i]
            @error "i = $i, Init shock $(reset_strat.init_shocks[i]) does not match expected $(expected_inits[i])"
            in_sync[1] = false
        end
        _reset!(env)
    end
    @test in_sync[1]
    @test env.cache.goal_cache.current_config == 1
    @test reset_strat.current_config == 1
    @test get_target_shock_count(goal_strat, env) == 1
end
