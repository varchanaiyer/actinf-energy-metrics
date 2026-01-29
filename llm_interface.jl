#This file prompts the LLM to interpret the active inference agent's behavior
# 
# Multi-Provider LLM Support:
# - OpenAI GPT-3.5 Turbo (default) - requires OPENAI_API_KEY
# - Claude 3.5 Haiku - requires ANTHROPIC_API_KEY  
# - Gemini 2.0 Flash - requires GOOGLE_API_KEY
#
# The system will auto-detect available API keys and use the first available provider
# in priority order: OpenAI > Claude > Gemini

using HTTP
using JSON3


function load_env_file()
    env_vars = Dict{String, String}()
    env_file = joinpath(dirname(@__FILE__), ".env")
    
    if isfile(env_file)
        for line in eachline(env_file)
            line = strip(line)
            if !isempty(line) && !startswith(line, "#")
                if occursin("=", line)
                    key, value = split(line, "=", limit=2)
                    env_vars[strip(key)] = strip(value)
                end
            end
        end
    end
    
    return env_vars
end

# Load environment variables
env_vars = load_env_file()
api_key = get(env_vars, "OPENAI_API_KEY", "")


# LLM Interface Structure
mutable struct LLMInterface
    api_key::String
    model_name::String
    base_url::String
    provider::String
    max_tokens::Int
    temperature::Float64
end

# Constructor functions for different providers
# GPT (default)
function LLMInterface(api_key::String; model_name::String="gpt-3.5-turbo", max_tokens::Int=500, temperature::Float64=0.0)
    return LLMInterface(api_key, model_name, "https://api.openai.com/v1/chat/completions", "openai", max_tokens, temperature)
end

# Claude 3.5 Haiku
function create_claude_interface(api_key::String; model_name::String="claude-3-5-haiku-20241022", max_tokens::Int=500, temperature::Float64=0.0)
    return LLMInterface(api_key, model_name, "https://api.anthropic.com/v1/messages", "anthropic", max_tokens, temperature)
end

# Gemini 2.0 Flash
function create_gemini_interface(api_key::String; model_name::String="gemini-2.0-flash-exp", max_tokens::Int=500, temperature::Float64=0.0)
    return LLMInterface(api_key, model_name, "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent", "google", max_tokens, temperature)
end

# Generate interpretation prompt from belief trace
function create_interpretation_prompt(belief_trace::Vector{Dict{String, Any}}, num_recent::Int=5)
    recent_actions = belief_trace[max(1, length(belief_trace)-num_recent+1):end]
    
    # Format the recent actions into readable text
    action_summary = ""
    for (i, action) in enumerate(recent_actions)
        action_summary *= """
        Step $(action["timestep"]): 
        - Time: $(action["time"])
        - Initial State: $(round(action["initial_state"], digits=2)) MW
        - Action: $(action["action"])
        - End State: $(round(action["end_state"], digits=2)) MW
        - Observed: $(round(action["observation"], digits=2)) MW
        
        """
    end
    
    #need to add gold standard response for k-shot prompting below
    prompt = """
    You are an AI interpreter analyzing an active inference agent's behavior in an energy grid system.
    
    The agent has taken the following recent actions:
    
    $action_summary
    
    Based on each action performed by the agent, please provide a natural language interpretation of:
    
    1. What the agent believes about current grid conditions
    2. Why the agent chose these specific actions
    3. How well the prediction aligned with the observation
    4. Any potential concerns or unusual behavior you notice
    
    Keep your explanation clear and accessible to someone managing an energy grid.
    Focus on the agent's reasoning and decision-making process.
    """
    
    return prompt
end

# Query the OpenAI API
function query_openai(llm::LLMInterface, prompt::String)
    headers = [
        "Authorization" => "Bearer $(llm.api_key)",
        "Content-Type" => "application/json"
    ]
    
    body = JSON3.write(Dict(
        "model" => llm.model_name,
        "messages" => [
            Dict("role" => "system", "content" => "You are an expert energy grid analyst and AI interpreter."),
            Dict("role" => "user", "content" => prompt)
        ],
        "max_tokens" => llm.max_tokens,
        "temperature" => llm.temperature
    ))
    
    try
        response = HTTP.post(llm.base_url, headers, body)
        if response.status == 200
            response_data = JSON3.read(String(response.body))
            return response_data.choices[1].message.content
        else
            error("API request failed with status $(response.status): $(String(response.body))")
        end
    catch e
        error("Failed to query OpenAI API: $e")
    end
end

# Query the Claude API
function query_claude(llm::LLMInterface, prompt::String)
    headers = [
        "x-api-key" => llm.api_key,
        "Content-Type" => "application/json",
        "anthropic-version" => "2023-06-01"
    ]
    
    body = JSON3.write(Dict(
        "model" => llm.model_name,
        "max_tokens" => llm.max_tokens,
        "temperature" => llm.temperature,
        "messages" => [
            Dict("role" => "user", "content" => prompt)
        ],
        "system" => "You are an expert energy grid analyst and AI interpreter."
    ))
    
    try
        response = HTTP.post(llm.base_url, headers, body)
        if response.status == 200
            response_data = JSON3.read(String(response.body))
            return response_data.content[1].text
        else
            error("API request failed with status $(response.status): $(String(response.body))")
        end
    catch e
        error("Failed to query Claude API: $e")
    end
end

# Query the Gemini API
function query_gemini(llm::LLMInterface, prompt::String)
    # Gemini uses API key as query parameter
    url_with_key = "$(llm.base_url)?key=$(llm.api_key)"
    
    headers = [
        "Content-Type" => "application/json"
    ]
    
    body = JSON3.write(Dict(
        "contents" => [
            Dict(
                "parts" => [
                    Dict("text" => "You are an expert energy grid analyst and AI interpreter. $prompt")
                ]
            )
        ],
        "generationConfig" => Dict(
            "temperature" => llm.temperature,
            "maxOutputTokens" => llm.max_tokens
        )
    ))
    
    try
        response = HTTP.post(url_with_key, headers, body)
        if response.status == 200
            response_data = JSON3.read(String(response.body))
            return response_data.candidates[1].content.parts[1].text
        else
            error("API request failed with status $(response.status): $(String(response.body))")
        end
    catch e
        error("Failed to query Gemini API: $e")
    end
end

# Main function to get interpretation
function get_agent_interpretation(llm::LLMInterface, belief_trace::Vector{Dict{String, Any}}; num_recent::Int=5)
    prompt = create_interpretation_prompt(belief_trace, num_recent)
    
    # Route to appropriate API based on provider
    if llm.provider == "openai"
        return query_openai(llm, prompt)
    elseif llm.provider == "anthropic"
        return query_claude(llm, prompt)
    elseif llm.provider == "google"
        return query_gemini(llm, prompt)
    else
        error("Unsupported provider: $(llm.provider). Supported providers: openai, anthropic, google")
    end
end

# Helper function to create LLM interface with auto-detection of available API keys
function create_llm_interface_auto()
    env_vars = load_env_file()
    
    # Try OpenAI first (default)
    openai_key = get(env_vars, "OPENAI_API_KEY", "")
    if !isempty(openai_key)
        println("Using OpenAI GPT-3.5 Turbo (default)")
        return LLMInterface(openai_key)
    end
    
    # Try Claude
    claude_key = get(env_vars, "ANTHROPIC_API_KEY", "")
    if !isempty(claude_key)
        println("Using Claude 3.5 Haiku")
        return create_claude_interface(claude_key)
    end
    
    # Try Gemini
    gemini_key = get(env_vars, "GOOGLE_API_KEY", "")
    if !isempty(gemini_key)
        println("Using Gemini 2.0 Flash")
        return create_gemini_interface(gemini_key)
    end
    
    error("No API keys found. Please add one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY to your .env file")
end

# Combined function: LLM predicts next action, then explains agent's actual choice
function llm_predict_and_explain_action(belief_entry::Dict{String, Any}, data_context::Union{Dict, Nothing}, timestep::Int, agent_state_history::Vector{Dict{String, Any}}, llm_interface::LLMInterface)
    # Get actual next action for comparison
    actual_next_action = "unknown"
    next_observation = nothing
    if timestep < length(agent_state_history)
        actual_next_action = agent_state_history[timestep + 1]["action"]
        next_observation = agent_state_history[timestep + 1]["observation"]
    end
    
    # STEP 1: Ask LLM to predict next action
    prediction_prompt = create_action_prediction_prompt(belief_entry, data_context, timestep, agent_state_history)
    
    # Get prediction from LLM
    if llm_interface.provider == "openai"
        prediction_response = query_openai(llm_interface, prediction_prompt)
    elseif llm_interface.provider == "anthropic"
        prediction_response = query_claude(llm_interface, prediction_prompt)
    elseif llm_interface.provider == "google"
        prediction_response = query_gemini(llm_interface, prediction_prompt)
    else
        error("Unsupported provider: $(llm_interface.provider)")
    end
    
    # Extract predicted action
    response_lower = lowercase(strip(prediction_response))
    predicted_action = if occursin("increase_generation", response_lower)
        "increase_generation"
    elseif occursin("decrease_generation", response_lower)
        "decrease_generation"
    elseif occursin("maintain", response_lower)
        "maintain"
    else
        "maintain"  # default fallback
    end
    
    # STEP 2: Tell LLM the actual choice and ask for explanation
    explanation_prompt = create_action_explanation_prompt(belief_entry, data_context, timestep, agent_state_history, predicted_action, actual_next_action)
    
    # Get explanation from LLM
    if llm_interface.provider == "openai"
        explanation_response = query_openai(llm_interface, explanation_prompt)
    elseif llm_interface.provider == "anthropic"
        explanation_response = query_claude(llm_interface, explanation_prompt)
    elseif llm_interface.provider == "google"
        explanation_response = query_gemini(llm_interface, explanation_prompt)
    else
        error("Unsupported provider: $(llm_interface.provider)")
    end
    
    # Terminal feedback
    accuracy_symbol = predicted_action == actual_next_action ? "✓" : "✗"
    if next_observation !== nothing
        println("T$timestep: Current=$(round(belief_entry["end_state"], digits=0))MW | LLM→'$predicted_action' | Actual→'$actual_next_action' | NextObs=$(round(next_observation, digits=0))MW | $accuracy_symbol")
    else
        println("T$timestep: LLM→'$predicted_action' | Actual→'$actual_next_action' | $accuracy_symbol")
    end
    
    return Dict(
        "predicted_action" => predicted_action,
        "actual_next_action" => actual_next_action,
        "prediction_correct" => predicted_action == actual_next_action,
        "prediction_response" => prediction_response,
        "action_explanation" => explanation_response,
        "combined_llm_response" => "PREDICTION: $prediction_response\n\nEXPLANATION: $explanation_response"
    )
end

# Create prompt for action prediction (Step 1)
function create_action_prediction_prompt(belief_entry::Dict{String, Any}, data_context::Union{Dict, Nothing}, timestep::Int, agent_state_history::Vector{Dict{String, Any}})
    prompt = """
    You are analyzing an active inference agent managing energy generation in a smart grid.

    CURRENT SITUATION (Timestep $timestep):
    • The agent believed demand would be: $(round(belief_entry["initial_state"], digits=0)) MW
    • The agent took action: $(belief_entry["action"])
    • After seeing actual demand of $(round(belief_entry["observation"], digits=0)) MW
    • The agent updated its belief to: $(round(belief_entry["end_state"], digits=0)) MW
    
    ANALYSIS:
    • Prediction error: $(round(abs(belief_entry["end_state"] - belief_entry["observation"]), digits=0)) MW
    • State change: $(round(belief_entry["end_state"] - belief_entry["initial_state"], digits=0)) MW
    """
    
    # Add grid context
    if data_context !== nothing
        prompt *= """
        
        GRID CONTEXT:
        • Renewable generation forecast: $(round(data_context["renewable_forecast"], digits=0)) MW
        • Load forecast: $(round(data_context["load_forecast"], digits=0)) MW
        """
    end
    
    # Add recent pattern context
    if timestep >= 3
        recent_actions = []
        for i in max(1, timestep-2):timestep
            if i <= length(agent_state_history)
                push!(recent_actions, agent_state_history[i]["action"])
            end
        end
        prompt *= """
        
        RECENT PATTERN: $(join(recent_actions, " → "))
        """
    end
    
    prompt *= """
    
    Based on the agent's current belief state ($(round(belief_entry["end_state"], digits=0)) MW) and decision-making pattern, predict what action the agent will take NEXT.
    
    Available actions:
    - increase_generation (expects significant demand increase)
    - decrease_generation (expects significant demand decrease)  
    - maintain (expects stable demand)
    
    Respond with ONLY the action name: increase_generation, decrease_generation, or maintain
    """
    
    return prompt
end

# Create prompt for action explanation (Step 2)
function create_action_explanation_prompt(belief_entry::Dict{String, Any}, data_context::Union{Dict, Nothing}, timestep::Int, agent_state_history::Vector{Dict{String, Any}}, predicted_action::String, actual_action::String)
    prompt = """
    You previously predicted the agent would take: "$predicted_action"
    But the agent actually chose: "$actual_action"
    
    SITUATION RECAP (Timestep $timestep):
    • Agent's belief: $(round(belief_entry["end_state"], digits=0)) MW
    • Current action taken: $(belief_entry["action"])
    • Prediction error: $(round(abs(belief_entry["end_state"] - belief_entry["observation"]), digits=0)) MW
    """
    
    if data_context !== nothing
        prompt *= """
        • Renewable forecast: $(round(data_context["renewable_forecast"], digits=0)) MW
        • Load forecast: $(round(data_context["load_forecast"], digits=0)) MW
        """
    end
    
    if predicted_action == actual_action
        prompt *= """
        
        ✓ Your prediction was CORRECT! 
        
        Explain WHY the agent chose "$actual_action" at this timestep. What factors led to this decision?
        """
    else
        prompt *= """
        
        ✗ Your prediction was incorrect.
        
        Explain WHY the agent actually chose "$actual_action" instead of "$predicted_action". What factors did you miss in your prediction?
        """
    end
    
    prompt *= """
    
    Consider:
    - Grid conditions and forecasts
    - Recent prediction errors
    - Energy demand patterns
    - Agent's risk management strategy
    
    Provide a 2-3 sentence explanation of the agent's reasoning.
    """
    
    return prompt
end

# Deterministic prediction of agent's next action — replays exact EFE logic
function deterministic_predict_next_action(next_end_state::Float64, next_load_forecast::Float64)
    desired = next_load_forecast / 1000.0
    action_effects = Dict("increase_generation" => 1.5, "decrease_generation" => -1.5, "maintain" => 0.0)

    best_action = "maintain"
    best_efe = Inf
    for (a, effect) in action_effects
        predicted = next_end_state + effect
        divergence = abs(predicted - desired)
        epistemic = abs(effect)
        efe = divergence - 0.5 * epistemic
        if efe < best_efe
            best_efe = efe
            best_action = a
        end
    end

    gap = desired - next_end_state
    return best_action, gap
end

# Build a prompt asking the LLM to EXPLAIN (not predict) the agent's action
function build_explanation_prompt(belief_entry::Dict{String, Any}, action::String, gap::Float64, next_load_forecast::Float64)
    end_state = belief_entry["end_state"]
    obs = belief_entry["observation"]

    return """
    You are an expert energy grid analyst interpreting an active inference agent's decisions.

    The agent just updated its belief about energy demand:
    • Previous belief: $(round(belief_entry["initial_state"] * 1000, digits=0)) MW
    • Observed actual demand: $(round(obs * 1000, digits=0)) MW
    • Updated belief: $(round(end_state * 1000, digits=0)) MW
    • Next hour's load forecast: $(round(next_load_forecast, digits=0)) MW
    • Gap between forecast and expected next belief: $(round(gap, digits=3)) (thousands MW)
    • Agent's chosen action: $action

    In ONE concise sentence, explain why the agent chose "$action" and what it means for the energy grid.
    Focus on the practical grid implications, not the math.
    """
end

# Predict next action deterministically + get LLM explanation
function llm_predict_next_action(belief_entry::Dict{String, Any}, data_context::Union{Dict, Nothing}, timestep::Int, agent_state_history::Vector{Dict{String, Any}}, llm_interface; next_data_context::Union{Dict, Nothing}=nothing)
    # Get actual next action for comparison
    actual_next_action = "unknown"
    if timestep < length(agent_state_history)
        actual_next_action = agent_state_history[timestep + 1]["action"]
    end

    # Use the ACTUAL next end_state (not an estimate) for exact EFE replay
    next_end_st = timestep < length(agent_state_history) ? agent_state_history[timestep + 1]["end_state"] : belief_entry["end_state"]

    # Get next load forecast
    if next_data_context !== nothing
        next_lf = next_data_context["load_forecast"]
    elseif data_context !== nothing
        next_lf = data_context["load_forecast"]
    else
        next_lf = next_end_st * 1000.0
    end

    # Deterministic prediction — replays agent's exact EFE logic with actual values
    predicted_action, gap = deterministic_predict_next_action(next_end_st, next_lf)

    # LLM explanation (optional — gracefully handle failures)
    llm_explanation = ""
    try
        explanation_prompt = build_explanation_prompt(belief_entry, predicted_action, gap, next_lf)

        if llm_interface.provider == "openai"
            llm_explanation = query_openai(llm_interface, explanation_prompt)
        elseif llm_interface.provider == "anthropic"
            llm_explanation = query_claude(llm_interface, explanation_prompt)
        elseif llm_interface.provider == "google"
            llm_explanation = query_gemini(llm_interface, explanation_prompt)
        else
            error("Unsupported provider: $(llm_interface.provider)")
        end
    catch e
        llm_explanation = "(LLM explanation error: $e)"
    end

    println("Timestep $timestep: Predicted '$predicted_action' | Actual '$actual_next_action' | gap=$(round(gap, digits=3)) | $(predicted_action == actual_next_action ? "✓" : "✗")")

    return Dict(
        "predicted_action" => predicted_action,
        "actual_next_action" => actual_next_action,
        "prediction_correct" => predicted_action == actual_next_action,
        "gap" => gap,
        "llm_response" => llm_explanation
    )
end

# Generate LLM prompt for state transition interpretation
function generate_llm_prompt(belief_entry::Dict{String, Any})
    prompt = """
    You are an AI interpreter analyzing an active inference agent's behavior in an energy grid system.

    The agent observed the following state transition:

    Time: $(belief_entry["time"])
    Initial State: Energy demand was estimated at $(round(belief_entry["initial_state"], digits=2)) MW
    Action Taken: $(belief_entry["action"])
    End State: Energy demand became $(round(belief_entry["end_state"], digits=2)) MW
    Actual Observed Demand: $(round(belief_entry["observation"], digits=2)) MW

    Based on this action performed by the agent, please provide a natural language interpretation of:

    1. What the agent believes about current grid conditions
    2. Why the agent chose this specific action
    3. How well the prediction aligned with the observation
    4. Any potential concerns or unusual behavior you notice

    Keep your explanation clear and accessible to someone managing an energy grid.
    Focus on the agent's reasoning and decision-making process.
    """
    
    return prompt
end

# Function to ask user which LLM to use
function prompt_for_llm_selection()
    # Display available options
    println("\n=== Select a Large Language Model ===")
    println("1. OpenAI GPT-3.5 Turbo (Default)")
    println("2. OpenAI GPT-4o")
    println("3. Claude 3.5 Haiku")
    println("4. Claude 3 Opus")
    println("5. Gemini 2.0 Flash")
    println("6. OpenAI GPT-5")
    println("A. Auto-detect from available API keys")
    
    # Get user input
    print("\nEnter your choice (1-6 or A): ")
    choice = lowercase(strip(readline()))
    
    # Load environment variables
    env_vars = load_env_file()
    
    # Process based on choice
    if choice == "1"
        api_key = get(env_vars, "OPENAI_API_KEY", "")
        if isempty(api_key)
            error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        end
        println("Using OpenAI GPT-3.5 Turbo")
        return LLMInterface(api_key, model_name="gpt-3.5-turbo")
        
    elseif choice == "2"
        api_key = get(env_vars, "OPENAI_API_KEY", "")
        if isempty(api_key)
            error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        end
        println("Using OpenAI GPT-4o")
        return LLMInterface(api_key, model_name="gpt-4o")
        
    elseif choice == "3"
        api_key = get(env_vars, "ANTHROPIC_API_KEY", "")
        if isempty(api_key)
            error("Anthropic API key not found. Please add ANTHROPIC_API_KEY to your .env file.")
        end
        println("Using Claude 3.5 Haiku")
        return create_claude_interface(api_key, model_name="claude-3-5-haiku-20241022")
        
    elseif choice == "4"
        api_key = get(env_vars, "ANTHROPIC_API_KEY", "")
        if isempty(api_key)
            error("Anthropic API key not found. Please add ANTHROPIC_API_KEY to your .env file.")
        end
        println("Using Claude 3 Opus")
        return create_claude_interface(api_key, model_name="claude-3-opus-20240229")
        
    elseif choice == "5"
        api_key = get(env_vars, "GOOGLE_API_KEY", "")
        if isempty(api_key)
            error("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
        end
        println("Using Gemini 2.0 Flash")
        return create_gemini_interface(api_key)

    elseif choice == "6"
        api_key = get(env_vars, "OPENAI_API_KEY", "")
        if isempty(api_key)
            error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        end
        println("Using OpenAI GPT-5")
        return LLMInterface(api_key, model_name="gpt-5")
        
    elseif choice == "a"
        println("Auto-detecting available API keys...")
        return create_llm_interface_auto()
        
    else
        println("Invalid choice. Defaulting to auto-detection.")
        return create_llm_interface_auto()
    end
end

# Export functions
export LLMInterface, get_agent_interpretation, create_interpretation_prompt,
       create_claude_interface, create_gemini_interface, create_llm_interface_auto,
       query_openai, query_claude, query_gemini, load_env_file,
       llm_predict_and_explain_action, create_action_prediction_prompt, create_action_explanation_prompt,
       llm_predict_next_action, generate_llm_prompt, prompt_for_llm_selection,
       deterministic_predict_next_action, build_explanation_prompt