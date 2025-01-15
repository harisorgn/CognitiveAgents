mutable struct CategoryLearnEnv <: AbstractEnv
    const S::Matrix{Float64}
    const CAT::Vector{Int}
    const N_trials::Int
    current_trial::Int
    
    function CategoryLearnEnv(stimuli::DataFrame)
        S = transpose(Matrix(stimuli[!, Not(:category)]))
        C = stimuli[!, :category]

        new(S, C, length(C), 1)
    end
end

function CommonRLInterface.reset!(env::CategoryLearnEnv)
    env.current_trial = 1
end

CommonRLInterface.actions(::CategoryLearnEnv) = (1, 2)

CommonRLInterface.observe(env::CategoryLearnEnv) = env.S[:, env.current_trial]

CommonRLInterface.terminated(env::CategoryLearnEnv) = env.current_trial > env.N_trials

CommonRLInterface.act!(env::CategoryLearnEnv, action) = action == env.C[env.current_trial]
