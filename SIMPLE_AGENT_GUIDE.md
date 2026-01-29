# Smart Grid Agent Guide: Simple Version

## What This Agent Does

This agent manages electricity in a power grid. It decides when to increase, decrease, or maintain power generation based on what people need.

## How It Works

1. **Watches electricity demand**: Sees how much power people are using
2. **Makes predictions**: Guesses what people will need next
3. **Takes actions**: Chooses to increase, decrease, or maintain power generation
4. **Learns from results**: Updates its beliefs based on what actually happened

## The Data It Uses

- **Current demand**: How much electricity people are using right now
- **Load forecast**: Prediction of future electricity needs
- **Renewable energy**: Available solar and wind power
- **Time of day**: Morning, afternoon, evening, or night patterns

## How It Makes Decisions

The agent has three possible actions:
- **"increase_generation"**: Make more electricity
- **"decrease_generation"**: Make less electricity
- **"maintain"**: Keep electricity production the same

It decides based on:
1. **Change in belief**: Difference between what it thought before and what it thinks now
2. **Future target**: Where it thinks demand is heading
3. **Renewable availability**: How much solar and wind power is available

## Understanding the Decision Math

```
If forecast > current belief → increase_generation
If forecast < current belief → decrease_generation
If forecast ≈ current belief → maintain
```

The agent uses "Expected Free Energy" (EFE) to pick the best action:
```
EFE = |Predicted State After Action - Forecast| - 0.5 × Information Value
```
It chooses the action with the lowest EFE score.

## Practical Examples

### Morning Rush (7:00 AM)
```
Current belief: 40,000 MW
Load forecast: 44,000 MW
Renewables: Low (5,000 MW)
```
**Action**: increase_generation
**Why**: People waking up need more power, and solar isn't available yet

### Sunny Afternoon (1:00 PM)
```
Current belief: 48,000 MW
Load forecast: 47,000 MW
Renewables: High (15,000 MW solar)
```
**Action**: decrease_generation
**Why**: Demand slightly lower and lots of solar power available

### Evening Peak (7:00 PM)
```
Current belief: 51,000 MW
Load forecast: 54,000 MW
Renewables: Low (3,000 MW)
```
**Action**: increase_generation
**Why**: Evening peak demand with low renewable energy

## Predicting What It Will Do Next

### Time of Day Patterns
- **Morning (6-9 AM)**: Usually increases generation
- **Midday (10 AM-2 PM)**: Often maintains or decreases with high solar
- **Evening (5-8 PM)**: Usually increases for peak demand
- **Night (9 PM-5 AM)**: Usually decreases or maintains

### Renewable Energy Factors
- High solar/wind → More likely to decrease generation
- Low renewables → More likely to increase generation

### Looking at Trends
- Rising demand over hours → Expect increase_generation
- Falling demand → Expect decrease_generation
- Stable demand → Expect maintain

## How to Practice Prediction

1. Look at the last few hours of data
2. Note the current state, forecast, and renewable availability
3. Predict what the agent will do next
4. Check what it actually did
5. Learn from any differences

## Simple Way to Learn Action Prediction

Follow this easy 3-step method:

### Step 1: Look at the Gap
Compare the agent's current belief with the load forecast:
```
Gap = Load Forecast - Agent's Current Belief (end_state)
```

### Step 2: Apply the Simple Rules
- **If Gap > +2,000 MW**: Predict "increase_generation"
- **If Gap < -2,000 MW**: Predict "decrease_generation"  
- **If Gap is between -2,000 and +2,000 MW**: Predict "maintain"

### Step 3: Check for Special Cases
Adjust your prediction if:
- **High renewables (>10,000 MW)**: More likely to decrease or maintain
- **Low renewables (<3,000 MW)**: More likely to increase
- **Morning hours (6-9 AM)**: Bias toward increase
- **Evening hours (5-8 PM)**: Bias toward increase
- **Night hours (10 PM-5 AM)**: Bias toward decrease

### Practice Example
```
Agent's current belief: 48,000 MW
Load forecast: 52,000 MW
Renewables: 8,000 MW
Time: 7:00 AM
```

**Step 1**: Gap = 52,000 - 48,000 = +4,000 MW (large positive gap)
**Step 2**: Gap > +2,000 MW → predict "increase_generation"
**Step 3**: Morning time + large gap confirms → "increase_generation"
**Prediction**: increase_generation ✓

### Quick Reference Card
Print this out and keep it handy:

| Situation | Gap Size | Renewables | Time | Likely Action |
|-----------|----------|------------|------|---------------|
| Morning rush | Any positive | Any | 6-9 AM | increase_generation |
| Solar peak | Small | High (>10k) | 11 AM-2 PM | maintain/decrease |
| Evening peak | Large positive | Low (<5k) | 5-8 PM | increase_generation |
| Night time | Any negative | Any | 10 PM-5 AM | decrease_generation |
| Stable period | Small gap | Medium | Other | maintain |

Start with these patterns, then add complexity as you get better!

## Common Mistakes to Avoid

1. Forgetting the time of day matters
2. Ignoring renewable energy availability
3. Looking at single points instead of trends
4. Expecting perfect predictions (the grid is complex!)
5. Forgetting that MW means megawatts (big numbers are normal)

## All States in the Agent

The agent maintains several states to track its beliefs and observations:

### 1. Belief States
- **Initial State** (`initial_state`): What the agent thought demand was before seeing new data
- **End State** (`end_state`): What the agent thinks demand is after seeing measurements
- **Process Noise**: Fixed at 2,000 MW - how much demand naturally varies between timesteps
- **Observation Noise**: Fixed at 1,000 MW - how reliable the agent considers measurements

### 2. Observation State
- **Observation** (`observation`): The actual measured electricity demand from the grid

### 3. Action State
- **Action** (`action`): One of "increase_generation", "decrease_generation", or "maintain"

### 4. Context States (from grid data)
- **Renewable Forecast**: Expected renewable generation in MW
- **Load Forecast**: Expected demand in MW
- **Solar Generation**: Current solar power production in MW
- **Wind Generation**: Current wind power production in MW

### 5. Time States
- **Timestamp** (`time`): The current date and time (e.g., "2015-01-02T07:00:00Z")
- **Timestep** (`timestep`): Sequential counter (1, 2, 3...) of decision points

### Memory Structure
These states are stored in:
```
agent.state_history  # Complete history of all states
agent.action_history # Just the sequence of actions
```

Each entry in `state_history` looks like:
```
{
  "timestep" => 25,
  "time" => "2015-01-02T07:00:00Z",
  "initial_state" => 50,400 MW,
  "action" => "increase_generation",
  "end_state" => 53,380 MW,
  "observation" => 54,131 MW
}
```

**Important**: The agent doesn't analyze patterns across many timesteps. It only considers the current state and context when deciding what to do next.

---

This guide helps you understand how the agent makes decisions to keep electricity flowing in the power grid.
