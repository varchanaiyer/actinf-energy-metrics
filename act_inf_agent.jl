# Smart Grid Active Inference Agent using RxInfer.jl
# Julia implementation of the active inference pipeline

using RxInfer
using GraphPPL
using Distributions 
using DataFrames
using CSV
using Plots
using JSON3
using Statistics
using SQLite
using DBInterface
using LinearAlgebra  # For cosine similarity in BERTScore
using HTTP  # Add HTTP for API calls

# Include LLM interface at the top level to avoid world age issues
include("llm_interface.jl")

# Define a realistic probabilistic model for energy demand with state evolution
@model function energy_demand_model(y, n)
    # State evolution with process noise
    x_0 ~ Normal(42.0, 5.0)  # Initial state prior (mean, std)
    x = Vector{Real}(undef, n)
    x[1] = x_0
    
    for t in 2:n
        # State transition with drift and noise
        x[t] ~ Normal(x[t-1], 2.0)  # Process noise (mean, std)
    end
    
    # Observation model
    for t in 1:n
        y[t] ~ Normal(x[t], 1.0)  # Observation noise (mean, std)
    end
end

# Smart Grid Active Inference Agent
mutable struct SmartGridActiveInferenceAgent
    T::Int  # Time horizon
    state_history::Vector{Dict{String, Any}}  # Full history of all agent states
    action_history::Vector{String}  # History of actions taken
    
    function SmartGridActiveInferenceAgent(T::Int)
        agent = new()
        agent.T = T
        agent.state_history = Dict{String, Any}[]
        agent.action_history = String[]
        return agent
    end
end

# Run inference over time series with state tracking
# Simple EFE-based action selector (scaled units: thousands MW)
function select_action_by_efe(belief_entry::Dict{String,Any}, data_context::Union{Dict,Nothing}=nothing;
                             possible_actions=["increase_generation","decrease_generation","maintain"],
                             action_effects=Dict("increase_generation"=>1.5, "decrease_generation"=>-1.5, "maintain"=>0.0),
                             desired_target::Union{Float64,Nothing}=nothing)
    # Use load forecast as a preferred target if available
    if desired_target === nothing && data_context !== nothing && haskey(data_context, "load_forecast")
        desired_target = data_context["load_forecast"] / 1000.0  # Scale to same units
    elseif desired_target === nothing
        desired_target = belief_entry["end_state"]  # Default: keep current belief
    end

    best_action = possible_actions[1]
    best_efe = Inf

    prior_uncertainty = 2.0  # Proxy prior std (tuneable)
    for a in possible_actions
        effect = get(action_effects, a, 0.0)
        # Predict next-state mean after applying action
        predicted_state = belief_entry["end_state"] + effect
        # Predict observation ~= predicted_state (simple model)
        predicted_obs = predicted_state
        # Expected utility: negative absolute deviation from desired target
        expected_divergence = abs(predicted_obs - desired_target)
        # Epistemic value proxy: actions that change state more give more info
        epistemic_value = abs(effect)  # Simple proxy; could use expected variance reduction
        efe = expected_divergence - 0.5 * epistemic_value  # Weight epistemic term (tuneable)
        if efe < best_efe
            best_efe = efe
            best_action = a
        end
    end

    return best_action, best_efe
end

# Extract data context for a specific timestep
function extract_data_context(data::DataFrame, timestep::Int)
    if timestep < 1 || timestep > nrow(data)
        return nothing
    end
    
    # Return context states from grid data
    return Dict{String,Any}(
        "load_forecast" => data[timestep, :load_forecast],       # Expected demand in MW
        "renewable_forecast" => data[timestep, :renewable_forecast], # Expected renewable generation in MW
        "solar_generation" => data[timestep, :solar_generation], # Current solar output in MW
        "wind_generation" => data[timestep, :wind_generation]    # Current wind output in MW
    )
end

# Update run_inference! to accept data context
function run_inference!(agent::SmartGridActiveInferenceAgent, observations::Vector{Float64},
                       actions::Union{Vector{String}, Nothing}=nothing, data_context::Union{DataFrame, Nothing}=nothing;
                       use_rxinfer::Bool=false)  # Set to false by default to avoid warnings
    posteriors = Float64[]
    states = Dict{String, Any}[]
    
    # Run inference on all observations at once
    n = length(observations)
    
    if use_rxinfer
        try
            # Run inference using RxInfer (v4+ API)
            result = infer(
                model = energy_demand_model,
                data = (y = observations, n = n)
            )
        
        # Extract posterior means for each timestep
        x_posteriors = result.posteriors[:x]
        
        for t in 1:n
            # Get posterior for timestep t
            if haskey(x_posteriors, t)
                posterior_dist = x_posteriors[t]
                posterior_mean = mean(posterior_dist)
            else
                # Fallback if inference fails
                posterior_mean = t > 1 ? posteriors[t-1] : 42.0
            end
            
            push!(posteriors, posterior_mean)
            
            # Determine action based on state change
            initial_state = t > 1 ? posteriors[t-1] : 42.0
            end_state = posterior_mean
            
            if actions !== nothing && t <= length(actions)
                action = actions[t]
            else
                # Improved action inference based on realistic thresholds
                if t > 1
                    state_change = end_state - initial_state
                    if state_change > 1.0  # > 1,000 MW increase
                        action = "increase_generation"
                    elseif state_change < -1.0  # > 1,000 MW decrease
                        action = "decrease_generation"
                    else
                        action = "maintain"  # Small changes
                    end
                else
                    action = "initialize"  # First timestep
                end
            end
            
            # Store state transition
            state_transition = Dict{String, Any}(
                "timestep" => t,                # Sequential counter (time state)
                "initial_state" => initial_state, # Prior belief before seeing new data (belief state)
                "action" => action,              # Action taken: increase_generation, decrease_generation, or maintain (action state)
                "end_state" => end_state,        # Posterior belief after seeing new data (belief state)
                "observation" => observations[t] # Actual measured energy demand (observation state)
            )
            push!(states, state_transition)
            push!(agent.action_history, action)
        end
        
        catch e
            println("Warning: RxInfer failed, using simple Bayesian update: $e")
            use_rxinfer = false  # Fall back to simple method
        end
    end
    
    # Use simple Bayesian update if RxInfer disabled or failed
    if !use_rxinfer
        # Fallback to simple Bayesian updating
        prior_mean = 42.0
        prior_var = 25.0
        obs_var = 1.0
        
        for t in 1:n
            obs = observations[t]
            
            # Bayesian update: posterior = (prior * likelihood) / evidence
            posterior_var = 1.0 / (1.0/prior_var + 1.0/obs_var)
            posterior_mean = posterior_var * (prior_mean/prior_var + obs/obs_var)
            
            push!(posteriors, posterior_mean)
            
            # Determine action based on state change
            initial_state = t > 1 ? posteriors[t-1] : 42.0
            end_state = posterior_mean
            
            if actions !== nothing && t <= length(actions)
                action = actions[t]
            else
                if t > 1
                    # Create a belief entry for EFE calculation
                    belief_entry = Dict{String,Any}(
                        "timestep" => t,
                        "initial_state" => initial_state,
                        "end_state" => end_state,
                        "observation" => obs
                    )

                    # Get data context for this timestep if available
                    step_context = data_context !== nothing ? extract_data_context(data_context, t) : nothing

                    # Select action using EFE
                    action, _ = select_action_by_efe(belief_entry, step_context)
                else
                    action = "initialize"
                end
            end
            
            # Store state transition
            state_transition = Dict{String, Any}(
                "timestep" => t,                # Sequential counter (time state)
                "initial_state" => initial_state, # Prior belief before seeing new data (belief state)
                "action" => action,              # Action taken: increase_generation, decrease_generation, or maintain (action state)
                "end_state" => end_state,        # Posterior belief after seeing new data (belief state)
                "observation" => observations[t] # Actual measured energy demand (observation state)
            )
            push!(states, state_transition)
            push!(agent.action_history, action)
            
            # Update prior for next iteration
            prior_mean = posterior_mean
            prior_var = posterior_var + 4.0  # Add process noise
        end
    end
    
    agent.state_history = states
    return posteriors
end

# Extract belief trace with state transitions for LLM interpretation
function extract_belief_trace(agent::SmartGridActiveInferenceAgent, timestamps::Vector{String})
    belief_trace = Dict{String, Any}[]
    
    for (t, state_info) in enumerate(agent.state_history)
        if t <= length(timestamps)
            trace_entry = Dict{String, Any}(
                "time" => timestamps[t],          # Current date/time (time state)
                "timestep" => t,                  # Sequential counter (time state)
                "initial_state" => state_info["initial_state"], # Prior belief (belief state)
                "action" => state_info["action"], # Action taken (action state)
                "end_state" => state_info["end_state"], # Posterior belief (belief state)
                "observation" => state_info["observation"], # Measured demand (observation state)
                "description" => "At t=$(timestamps[t]): " *
                               "Initial state: $(round(state_info["initial_state"], digits=2)) MW, " *
                               "Action taken: $(state_info["action"]), " *
                               "End state: $(round(state_info["end_state"], digits=2)) MW, " *
                               "Observed demand: $(round(state_info["observation"], digits=2)) MW"
            )
            push!(belief_trace, trace_entry)
        end
    end
    
    return belief_trace
end

# Prompt human for explanation of agent action intent (every 50th timestep)
function prompt_for_human_explanation(belief_entry::Dict{String, Any}, data_context::Union{Dict, Nothing}, timestep::Int)
    println("\n" * "="^60)
    println("ACTION INTENT EXPLANATION - Timestep $timestep")
    println("="^60)
    
    # Display agent state information
    initial_state = belief_entry["initial_state"]
    end_state = belief_entry["end_state"]
    action = belief_entry["action"]
    observation = belief_entry["observation"]
    
    println("Agent State Transition:")
    println("  • Initial State: $(round(initial_state, digits=2)) MW")
    println("  • Action Taken: $action")
    println("  • End State: $(round(end_state, digits=2)) MW")
    println("  • Actual Observation: $(round(observation, digits=2)) MW")
    println("  • State Change: $(round(end_state - initial_state, digits=2)) MW")
    println("  • Prediction Error: $(round(abs(end_state - observation), digits=2)) MW")
    
    # Display additional context if available
    if data_context !== nothing
        println("\nGrid Context:")
        if haskey(data_context, "renewable_forecast")
            println("  • Renewable Generation: $(round(data_context["renewable_forecast"], digits=2)) MW")
        end
        if haskey(data_context, "load_forecast")
            println("  • Load Forecast: $(round(data_context["load_forecast"], digits=2)) MW")
        end
    end
    
    println("\nPlease provide a concise explanation for WHY the agent took this action.")
    println("Focus on the agent's intent and reasoning:")
    println("  1. What factors influenced the agent's decision?")
    println("  2. How accurate was the agent's prediction?")
    println("  3. Was this action appropriate for the grid state?")
    println("  4. What was the agent trying to achieve?")
    
    print("\nYour explanation of agent intent: ")
    human_explanation = readline()
    
    # Validate input
    while isempty(strip(human_explanation))
        print("Please provide a non-empty explanation: ")
        human_explanation = readline()
    end
    
    return strip(human_explanation)
end

# Create comprehensive dataset with LLM predictions and human explanations
function create_comprehensive_dataset(agent::SmartGridActiveInferenceAgent, data::DataFrame, llm_interface, explanation_interval::Int=50; interactive::Bool=true)
    println("Creating comprehensive dataset with LLM action predictions and human intent explanations...")
    println("• Every timestep: LLM will predict the agent's NEXT action")
    if interactive
        println("• Every $(explanation_interval)th timestep: You'll explain the agent's intent")
        println("Press Enter to continue...")
        readline()
    end
    
    action_predictions = Dict{String, Any}[]
    intent_explanations = Dict{String, Any}[]
    
    for t in 1:length(agent.state_history)
        belief_entry = agent.state_history[t]
        
        # Get additional context from data if available
        data_context = nothing
        if t <= nrow(data)
            data_context = Dict(
                "renewable_forecast" => data[t, :renewable_forecast],
                "load_forecast" => data[t, :load_forecast]
            )
        end
        
        # Get NEXT timestep's data context (needed for predicting next action)
        next_data_context = nothing
        if t + 1 <= nrow(data)
            next_data_context = Dict(
                "renewable_forecast" => data[t + 1, :renewable_forecast],
                "load_forecast" => data[t + 1, :load_forecast]
            )
        end

        # Collect LLM action prediction for every timestep
        if t < length(agent.state_history)  # Only if there's a next action to predict
            prediction_result = llm_predict_next_action(belief_entry, data_context, t, agent.state_history, llm_interface; next_data_context=next_data_context)
            
            prediction_entry = Dict{String, Any}(
                "timestep" => t,
                "time" => haskey(belief_entry, "time") ? belief_entry["time"] : "t$t",
                "predicted_action" => prediction_result["predicted_action"],
                "actual_next_action" => prediction_result["actual_next_action"],
                "prediction_correct" => prediction_result["prediction_correct"],
                "initial_state" => belief_entry["initial_state"],
                "end_state" => belief_entry["end_state"],
                "observation" => belief_entry["observation"],
                "current_action" => belief_entry["action"],
                "llm_response" => prediction_result["llm_response"]
            )
            push!(action_predictions, prediction_entry)
        end
        
        # Collect intent explanation for every nth timestep (only if interactive)
        if interactive && t % explanation_interval == 1  # Every 50th timestep starting from 1
            intent_explanation = prompt_for_human_explanation(belief_entry, data_context, t)
            
            explanation_entry = Dict{String, Any}(
                "timestep" => t,
                "time" => haskey(belief_entry, "time") ? belief_entry["time"] : "t$t",
                "action" => belief_entry["action"],
                "intent_explanation" => intent_explanation,
                "initial_state" => belief_entry["initial_state"],
                "end_state" => belief_entry["end_state"],
                "observation" => belief_entry["observation"],
                "human_generated" => true
            )
            push!(intent_explanations, explanation_entry)
        end
    end
    
    return action_predictions, intent_explanations
end

# Load smart grid data from SQLite database
function load_smart_grid_data(source::String="DE", limit::Int=48)
    try
        # Connect to SQLite database
        db = SQLite.DB("time_series.sqlite")
        
        # Query real energy data from Germany (DE)
        query = """
        SELECT 
            utc_timestamp,
            $(source)_load_actual_entsoe_transparency as load_actual,
            $(source)_load_forecast_entsoe_transparency as load_forecast,
            $(source)_solar_generation_actual as solar_generation,
            $(source)_wind_generation_actual as wind_generation
        FROM time_series_60min_singleindex 
        WHERE $(source)_load_actual_entsoe_transparency IS NOT NULL 
        AND $(source)_load_forecast_entsoe_transparency IS NOT NULL
        ORDER BY utc_timestamp 
        LIMIT $limit
        """
        
        # Execute query and get DataFrame
        df = DBInterface.execute(db, query) |> DataFrame
        
        # Close database connection
        SQLite.close(db)
        
        # Clean and process data
        if nrow(df) == 0
            error("No data found for source $source")
        end
        
        # Remove rows with missing values and convert to proper types
        df = dropmissing(df)
        
        # Create renewable forecast (solar + wind)
        df.renewable_forecast = coalesce.(df.solar_generation, 0.0) .+ coalesce.(df.wind_generation, 0.0)
        
        # Rename columns for consistency
        rename!(df, :utc_timestamp => :timestamp, :load_actual => :load_demand)
        
        println("Loaded $(nrow(df)) records from $source energy data")
        return df
        
    catch e
        println("Error loading from SQLite database: $e")
        println("Falling back to synthetic data...")
        return load_synthetic_data(limit)
    end
end

# Fallback function for synthetic data
function load_synthetic_data(n_points::Int=48)
    timestamps = [string("2025-08-01 ", lpad(i, 2, "0"), ":00") for i in 1:n_points]
    load_demand = 40000.0 .+ 5000.0 * sin.(2π * (1:n_points) / 24) .+ 1000.0 * randn(n_points)
    renewable_forecast = 5000.0 .+ 2000.0 * cos.(2π * (1:n_points) / 24) .+ 500.0 * randn(n_points)
    
    return DataFrame(
        timestamp = timestamps,
        load_demand = load_demand,
        load_forecast = load_demand .* (1 .+ 0.05 * randn(n_points)),
        renewable_forecast = renewable_forecast
    )
end

# Save belief trace for LLM interpretation
function save_belief_trace(belief_trace::Vector{Dict{String, Any}}, filename::String="belief_trace.json")
    open(filename, "w") do file
        JSON3.write(file, belief_trace)
    end
    println("Belief trace saved to $filename ($(length(belief_trace)) entries)")
end

# Save action predictions to JSON
function save_action_predictions(action_predictions::Vector{Dict{String, Any}}, filename::String="action_predictions.json")
    open(filename, "w") do file
        JSON3.write(file, action_predictions)
    end
    println("Action predictions saved to $filename ($(length(action_predictions)) entries)")
end

# Save intent explanations to JSON
function save_intent_explanations(intent_explanations::Vector{Dict{String, Any}}, filename::String="intent_explanations.json")
    open(filename, "w") do file
        JSON3.write(file, intent_explanations)
    end
    println("Intent explanations saved to $filename ($(length(intent_explanations)) entries)")
end

# Plot results function
function plot_results(posteriors::Vector{Float64}, timestamps::Vector{String}, observations::Vector{Float64})
    try
        # Create time indices for plotting
        time_indices = 1:length(posteriors)
        
        # Create the plot
        p = plot(time_indices, observations, label="Observations", linewidth=2, color=:blue)
        plot!(p, time_indices, posteriors, label="Posterior Mean", linewidth=2, color=:red, linestyle=:dash)
        
        # Customize the plot
        xlabel!(p, "Time Step")
        ylabel!(p, "Energy Demand (MW)")
        title!(p, "Active Inference Agent: Observations vs Posterior Beliefs")
        
        # Add grid and legend
        plot!(p, grid=true, legend=:topright)
        
        return p
    catch e
        println("Warning: Plotting failed: $e")
        println("Continuing without plot...")
        return nothing
    end
end

# Evaluate action prediction accuracy
function evaluate_action_predictions(action_predictions::Vector{Dict{String, Any}})
    if isempty(action_predictions)
        println("No action predictions to evaluate")
        return Dict("accuracy" => 0.0, "total" => 0)
    end

    # Count how many deterministic predictions matched (should be ~100%)
    correct_predictions = sum(pred["prediction_correct"] for pred in action_predictions)
    total_predictions = length(action_predictions)
    accuracy = correct_predictions / total_predictions

    println("Evaluation Summary:")
    println("=" ^ 40)
    println("   • Deterministic replay match: $correct_predictions/$total_predictions ($(round(accuracy * 100, digits=1))%)")
    println("   • LLM explanations collected: $(sum(1 for p in action_predictions if haskey(p, "llm_response") && !isempty(p["llm_response"])))")

    return Dict(
        "accuracy" => accuracy,
        "correct" => correct_predictions,
        "total" => total_predictions
    )
end

# Generate evaluation report
function generate_evaluation_report(summary_scores::Dict{String, Float64}, 
                                  detailed_results::Vector{Dict{String, Any}})
    report = """
    LLM Active Inference Interpretation Evaluation Report
    ===================================================
    
    Overall Performance:
    • Composite Score: $(round(summary_scores["composite"], digits=3))/1.0
    • Fidelity (Accuracy): $(round(summary_scores["fidelity"], digits=3))/1.0
    • Understandability: $(round(summary_scores["understandability"], digits=3))/1.0
    • Time Efficiency: $(round(summary_scores["time"], digits=3))/1.0
    • Conciseness: $(round(summary_scores["conciseness"], digits=3))/1.0
    
    Detailed Analysis:
    • Total Evaluations: $(length(detailed_results))
    • Best Composite Score: $(length(detailed_results) > 0 ? round(maximum([r["composite_score"] for r in detailed_results]), digits=3) : "N/A")
    • Worst Composite Score: $(length(detailed_results) > 0 ? round(minimum([r["composite_score"] for r in detailed_results]), digits=3) : "N/A")
    
    Recommendations:
    """
    
    if summary_scores["fidelity"] < 0.7
        report *= "• Improve mathematical accuracy and action prediction\n"
    end
    if summary_scores["understandability"] < 0.6
        report *= "• Enhance explanation clarity and semantic alignment\n"
    end
    if summary_scores["time"] < 0.5
        report *= "• Optimize generation speed for real-time requirements\n"
    end
    if summary_scores["conciseness"] < 0.6
        report *= "• Balance explanation length for better conciseness\n"
    end
    
    return report
end

# Main pipeline function
function run_pipeline()
    println("Smart Grid Active Inference Pipeline with RxInfer.jl")
    println("=" ^ 60)
    
    # Step 1: Load data from SQLite database
    println("Loading smart grid data from SQLite database...")
    data = load_smart_grid_data("DE", 200)  # Load 200 data points
    obs = data.load_demand
    timestamps = data.timestamp
    
    # Normalize data to reasonable scale for inference (convert MW to scaled units)
    obs_scaled = obs ./ 1000.0  # Scale down to thousands

    # Step 2: Initialize agent and run inference with EFE-based action selection
    println("Initializing Active Inference Agent...")
    agent = SmartGridActiveInferenceAgent(length(obs_scaled))

    println("Running inference with EFE-based action selection...")
    posteriors = run_inference!(agent, obs_scaled, nothing, data)
    
    # Scale posteriors back up for display
    posteriors_scaled = posteriors .* 1000.0
    
    # Step 3: Extract beliefs with state transitions
    println("Extracting belief trace...")
    belief_trace = extract_belief_trace(agent, timestamps)
    
    # Scale back the belief trace values for display
    for entry in belief_trace
        entry["initial_state"] *= 1000.0
        entry["end_state"] *= 1000.0
        entry["observation"] *= 1000.0
        # Update description with scaled values
        entry["description"] = "At t=$(entry["time"]): " *
                              "Initial state: $(round(entry["initial_state"], digits=0)) MW, " *
                              "Action taken: $(entry["action"]), " *
                              "End state: $(round(entry["end_state"], digits=0)) MW, " *
                              "Observed demand: $(round(entry["observation"], digits=0)) MW"
    end
    
    # Step 4: Initialize LLM interface for action predictions
    println("Initializing LLM interface for action predictions...")
    
    # Use auto-detection to select available LLM
    llm_interface = create_llm_interface_auto()
    println("LLM interface initialized successfully.")
    
    # Step 5: Create comprehensive dataset with LLM predictions and human explanations
    println("Creating comprehensive evaluation dataset...")
    action_predictions, intent_explanations = create_comprehensive_dataset(agent, data, llm_interface, 50; interactive=true)
    
    # Scale back values for saving
    for pred in action_predictions
        pred["initial_state"] *= 1000.0
        pred["end_state"] *= 1000.0
        pred["observation"] *= 1000.0
    end
    
    for exp in intent_explanations
        exp["initial_state"] *= 1000.0
        exp["end_state"] *= 1000.0
        exp["observation"] *= 1000.0
    end
    
    # Save datasets
    save_action_predictions(action_predictions)
    save_intent_explanations(intent_explanations)
    save_belief_trace(belief_trace)
    
    # Evaluate action prediction accuracy
    pred_results = evaluate_action_predictions(action_predictions)
    
    # Step 5: Generate example LLM prompts
    println("Generating example LLM prompts...")
    if !isempty(belief_trace)
        example_prompt = generate_llm_prompt(belief_trace[1])
        println("Example prompt for LLM:")
        println("-" ^ 40)
        println(example_prompt)
        println("-" ^ 40)
    end
    
    # Step 6: Show example intent explanation
    if !isempty(intent_explanations)
        println("Example intent explanation:")
        println("-" ^ 40)
        println("Timestep $(intent_explanations[1]["timestep"]): $(intent_explanations[1]["intent_explanation"])")
        println("-" ^ 40)
    end
    
    # Step 7: Plot results
    println("Plotting results...")
    p = plot_results(posteriors_scaled[1:min(50, length(posteriors_scaled))], 
                    timestamps[1:min(50, length(timestamps))], 
                    obs[1:min(50, length(obs))])
    display(p)
    
    # Summary
    println("\nPipeline Summary:")
    println("   • Time horizon: $(agent.T) steps")
    println("   • Mean posterior: $(round(mean(posteriors_scaled), digits=0)) MW")
    println("   • Mean observation: $(round(mean(obs), digits=0)) MW")
    println("   • State transitions tracked: $(length(agent.state_history))")
    println("   • Action predictions collected: $(length(action_predictions))")
    println("   • Intent explanations collected: $(length(intent_explanations))")
    println("   • Action prediction accuracy: $(round(pred_results["accuracy"] * 100, digits=1))%")
    println("   • Evaluation system: Composite scoring with 4 metrics")
    println("   • Data source: Real German energy data from SQLite")
    
    return agent, posteriors_scaled, belief_trace, action_predictions, intent_explanations
end

# Export main functions
export SmartGridActiveInferenceAgent, run_inference!, extract_belief_trace,
       load_smart_grid_data, run_pipeline,
       create_comprehensive_dataset, prompt_for_human_explanation,
       save_action_predictions, save_intent_explanations, evaluate_action_predictions,
       select_action_by_efe, extract_data_context,
       deterministic_predict_next_action

# Main execution block - will run when file is executed
function main()
    try
        println("Starting Active Inference Pipeline...")
        agent, posteriors, belief_trace, action_predictions, intent_explanations = run_pipeline()
        println("Pipeline completed successfully!")
        return agent, posteriors, belief_trace, action_predictions, intent_explanations
    catch e
        println("Error running pipeline: $e")
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        rethrow(e)
    end
end

# Run the pipeline if script is executed directly OR if we're in an interactive session
if (abspath(PROGRAM_FILE) == @__FILE__) || Base.isinteractive()
    main()
end