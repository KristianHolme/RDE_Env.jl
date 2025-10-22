@testitem "RDEEnv Initialization" begin
    using RDE

    @test begin
        params = RDEParam(; N = 16, tmax = 0.01)
        prob = RDEProblem(params)
        sum(abs.(prob.u0)) > 0.01
    end

    @test begin
        env = RDEEnv()
        true
    end
end

@testitem "RDEEnv Policies" begin
    using RDE

    @test begin
        env = RDEEnv(RDEParam(; N = 512, tmax = 0.1), reset_strategy = NShock(1))
        ConstPolicy = ConstantRDEPolicy(env)
        data = run_policy(ConstPolicy, env)
        data isa PolicyRunData
    end
end

@testitem "Observation Strategies" begin
    using RDE

    N = 16  # Number of spatial points for testing
    params = RDEParam(; N = N, tmax = 0.1)

    @testset "Fourier Observation" begin
        fft_terms = 4
        env = RDEEnv(params = params, observation_strat = FourierObservation(fft_terms))
        _reset!(env)
        # Test initialization
        @test length(env.observation) == 2fft_terms + 2  # 2 * fft_terms + s_scaled + u_p_scaled

        # Test observation
        obs = _observe(env)
        @test length(obs) == 2fft_terms + 2
        @test all(-1 .<= obs[1:2fft_terms] .<= 1)  # FFT coefficients should be normalized
        @test -1 <= obs[2fft_terms + 1] <= 1  # Normalized s_scaled
        @test -1 <= obs[2fft_terms + 2] <= 1  # Normalized u_p_scaled
    end

    @testset "State Observation" begin
        env = RDEEnv(params = params, observation_strat = StateObservation())
        _reset!(env)
        # Test initialization
        @test length(env.observation) == 2N + 2  # u and λ states + s_scaled + u_p_scaled

        # Test observation
        obs = _observe(env)
        @test length(obs) == 2N + 2
        @test all(-1 .<= obs[1:(end - 2)] .<= 1)  # State components should be normalized
        @test 0 <= obs[end - 1] <= 1  # Normalized s
        @test 0 <= obs[end] <= 1  # Normalized u_p

        # Test that first N components are normalized u and next N are normalized λ
        u_obs = obs[1:N]
        λ_obs = obs[(N + 1):2N]
        @test length(u_obs) == length(λ_obs) == N
    end

    @testset "Sampled Observation" begin
        n_samples = 8
        env = RDEEnv(params = params, observation_strat = SampledStateObservation(n_samples))
        _reset!(env)
        # Test initialization
        @test length(env.observation) == 2n_samples + 1  # sampled u and λ + time

        # Test observation
        obs = _observe(env)
        @test length(obs) == 2n_samples + 1
        @test all(-1 .<= obs[1:(end - 1)] .<= 1)  # Sampled values should be normalized
        @test 0 <= obs[end] <= 1  # Normalized time

        # Test sampling points
        u_samples = obs[1:n_samples]
        λ_samples = obs[(n_samples + 1):2n_samples]
        @test length(u_samples) == length(λ_samples) == n_samples
    end

    @testset "Observation Consistency" begin
        # Test that observations remain consistent after reset
        for strategy in [
                FourierObservation(4),
                StateObservation(),
                SampledStateObservation(8),
            ]
            env = RDEEnv(params = params, observation_strat = strategy)
            _reset!(env)
            obs1 = _observe(env)
            _reset!(env)
            obs2 = _observe(env)
            @test length(obs1) == length(obs2)
            @test all(isfinite.(obs1))
            @test all(isfinite.(obs2))
        end
    end

    @testset "Observation Type Consistency" begin
        # Test for Float32
        @test begin
            env = RDEEnv(params = RDEParam{Float32}())
            _reset!(env)
            obs = _observe(env)
            eltype(obs) == Float32
        end

        # Test for Float64
        @test begin
            env = RDEEnv(RDEParam{Float64}())
            _reset!(env)
            obs = _observe(env)
            eltype(obs) == Float64
        end

        # Test type consistency across different observation strategies
        for T in [Float32, Float64]
            for strategy in [
                    FourierObservation(4),
                    StateObservation(),
                    SampledStateObservation(8),
                ]
                @test begin
                    env = RDEEnv(RDEParam{T}(), observation_strat = strategy)
                    _reset!(env)
                    obs = _observe(env)
                    eltype(obs) == T
                end
            end
        end
    end
end
