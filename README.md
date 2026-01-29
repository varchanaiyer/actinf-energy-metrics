# LLM-Active Inference Integration Project

This project connects an Active Inference agent (using RxInfer.jl) with an LLM to provide real-time explanations and interpretations of the agent's belief states based on its actions.

## 🎯 Project Overview

The system works as follows:
1. **Active Inference Agent** makes decisions about energy grid management
2. **Action Logger** records what the agent does and why
3. **LLM Interpreter** analyzes the agent's actions and provides human-readable explanations
4. **Real-time Feedback** helps grid operators understand AI decision-making

## 📁 File Structure

- `act_inf_agent.jl` - Active Inference agent implementation
- `llm_interface.jl` - LLM integration module with multi-provider support
- `setup.jl` - Package installation script
- `AGENT_INTERPRETATION_GUIDE.md` - Comprehensive guide for understanding agent behavior
- `time_series.sqlite` - German energy grid database
- `README.md` - This file

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
julia setup.jl
```

### Step 2: Get an API Key
- **OpenAI**: Visit https://platform.openai.com/api-keys
- **Anthropic**: Visit https://console.anthropic.com/
- **Google**: Visit https://console.cloud.google.com/apis/credentials
- Copy your API key

### Step 3: Configure API Key
Create a `.env` file in the project directory:
```bash
# Choose one or more providers
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-claude-key-here  
GOOGLE_API_KEY=your-gemini-key-here
```

### Step 4: Run the Pipeline
```bash
julia act_inf_agent.jl
```

## 🔧 How It Works

### Active Inference Agent
- Uses RxInfer.jl for probabilistic inference
- Models energy grid dynamics
- Makes decisions to minimize "surprise" (unexpected outcomes)
- Tracks belief states and actions over time

### LLM Interface
- Connects to OpenAI GPT or Anthropic Claude
- Formats agent actions into natural language prompts
- Requests interpretations of agent behavior
- Provides human-readable explanations

### Integration Pipeline
1. Agent processes energy grid data
2. Agent makes decisions and updates beliefs
3. Actions are logged with context
4. LLM analyzes recent actions
5. LLM provides interpretation of agent's reasoning

## 📊 Example Output

```
🤖 LLM Interpretation:
----------------------------------------
Based on the agent's recent actions, I can see that:

1. **Grid Conditions**: The agent believes energy demand is 
   increasing and becoming more volatile, as evidenced by 
   its decision to increase generation capacity.

2. **Action Reasoning**: The agent chose to increase generation 
   because it observed a significant gap between expected and 
   actual demand (97.8 MW vs 98.1 MW estimated).

3. **Strategy Pattern**: The agent is following a conservative 
   approach, maintaining slightly higher generation capacity 
   than strictly necessary to handle demand uncertainty.

4. **Potential Concerns**: The agent's estimates are quite 
   close to observations, suggesting good model calibration.
----------------------------------------
```

## 🛠️ Customization

### Modify Agent Behavior
Edit `act_inf_agent` to:
- Change the energy grid model
- Adjust decision-making parameters
- Add new action types
- Modify belief state structure

### Customize LLM Prompts
Edit `llm_interface.jl` to:
- Change interpretation style
- Add specific analysis questions
- Modify prompt structure
- Support different LLM providers

### Add New Features
- Real-time data streaming
- Multiple agent coordination
- Advanced visualization
- Performance metrics

## 🔍 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
- Run `julia setup.jl` to install packages

**"API request failed"**
- Check your API key is correct
- Ensure you have API credits
- Verify internet connection

**"SSL certificate error"**
- On macOS, run: `/Applications/Python\ 3.x/Install\ Certificates.command`
- Or upgrade certifi: `pip3 install --upgrade certifi`

### Getting Help
- Check the error messages for specific issues
- Verify all packages are installed correctly
- Ensure your API key has sufficient credits
- Test with a simple API call first

## 📈 Next Steps

### Immediate Improvements
1. Add real-time data streaming
2. Implement confidence metrics
3. Add anomaly detection
4. Create web dashboard

### Advanced Features
1. Multi-agent coordination
2. Predictive explanations
3. Interactive debugging
4. Performance optimization

## 🤝 Contributing

This is a research project. Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Share use cases

## 📚 References

- [Active Inference](https://en.wikipedia.org/wiki/Active_inference)
- [RxInfer.jl Documentation](https://biaslab.github.io/RxInfer.jl/)
- [Julia Programming Language](https://julialang.org/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)

---

**Happy coding! 🎉** 