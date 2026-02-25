# Expert Skills Quick Reference

## Activated Skills

All expert skills are now active and integrated into the project context. They are automatically loaded via the CLAUDE.md configuration file.

## Available Expert Personas

### 1. 🐍 Python Expert (`python-expert.md`)
**Activation**: "As the Python expert, [task]..."

**Specializations**:
- Deep CPython internals, GIL, memory management
- Advanced features: metaclasses, descriptors, decorators, context managers
- Type systems: typing module, Protocol, TypeVar, strict mypy compliance
- Performance: profiling, optimization, Cython, Numba
- Scientific stack: NumPy, Pandas, SciPy mastery
- Production standards: testing (pytest, hypothesis), documentation, packaging

**Use For**:
- Code reviews and quality improvements
- Performance optimization and profiling
- Type safety and testing strategies
- Library architecture and design patterns
- Debugging complex Python issues
- Production-grade code development

**Example Usage**:
```
"As the Python expert, review this data processing pipeline for performance bottlenecks"
"Python expert: help me design a type-safe API for this module"
"Using Python expert knowledge, optimize this NumPy computation"
```

---

### 2. 🤖 ML Engineer (`ml-engineer.md`)
**Activation**: "As the ML engineer, [task]..."

**Specializations**:
- Classical statistics: hypothesis testing, experimental design, Bayesian methods
- Time series: ARIMA, GARCH, Prophet, structural models, causal inference
- Machine learning: XGBoost, LightGBM, neural networks, ensemble methods
- Deep learning: PyTorch, LSTM, Transformers, attention mechanisms
- Forecasting: Prophet, N-BEATS, temporal fusion transformers
- Model development: feature engineering, hyperparameter tuning, validation
- Production ML: MLflow tracking, model monitoring, deployment

**Use For**:
- Statistical model selection and development
- Time series forecasting and analysis
- Feature engineering strategies
- Hyperparameter optimization
- Model evaluation and diagnostics
- Production ML workflows
- Uncertainty quantification

**Example Usage**:
```
"As the ML engineer, design a forecasting model for hourly LMP spreads"
"ML engineer: evaluate this XGBoost model's performance and suggest improvements"
"Using ML expertise, help me set up proper cross-validation for time series data"
```

---

### 3. ⚡ FTR Trader (`ftr-trader.md`)
**Activation**: "As the FTR trader, [task]..."

**Specializations**:
- North American power markets: ERCOT, PJM, MISO, CAISO, ISO-NE, NYISO, SPP
- FTR mechanics: obligations, options, simultaneous feasibility, ARRs
- LMP decomposition: energy, congestion, loss components
- Congestion analysis: constraint binding, outage impacts, flow patterns
- Trading strategies: statistical arbitrage, portfolio optimization, mean reversion
- Risk management: VaR, stress testing, hedging, scenario analysis
- Fundamental analysis: production cost modeling, supply-demand balance

**Use For**:
- FTR path analysis and valuation
- Congestion forecasting models
- Trading strategy development
- Portfolio optimization
- Risk assessment and hedging
- Market fundamental analysis
- ISO/RTO-specific considerations

**Example Usage**:
```
"As the FTR trader, analyze this congestion pattern in PJM"
"FTR trader: design a portfolio optimization strategy with these constraints"
"Using FTR trading expertise, evaluate the fundamental drivers for this path"
```

---

## Multi-Skill Workflows

### Comprehensive Model Development
```
1. FTR Trader: Define market problem and fundamental drivers
2. ML Engineer: Design forecasting model architecture
3. Python Expert: Implement production-grade code
4. ML Engineer: Validate model performance
5. Python Expert: Optimize for production deployment
```

### Trading Strategy Implementation
```
1. FTR Trader: Identify trading opportunity and strategy
2. ML Engineer: Develop predictive models for signals
3. Python Expert: Build robust backtesting framework
4. FTR Trader: Validate against market microstructure
5. Python Expert: Production deployment with monitoring
```

### Code Optimization Workflow
```
1. Python Expert: Profile code and identify bottlenecks
2. ML Engineer: Assess if model can be simplified
3. Python Expert: Implement optimizations (vectorization, Cython)
4. ML Engineer: Validate model accuracy maintained
5. Python Expert: Production testing and benchmarking
```

---

## Skill Coordination Principles

### When to Use Multiple Skills
- **Complex Projects**: Combine market knowledge (FTR), statistical rigor (ML), and implementation quality (Python)
- **Code Reviews**: Python expert for code quality, ML engineer for model validity
- **Strategy Development**: FTR trader for market insight, ML engineer for modeling approach
- **Production Systems**: All three for end-to-end quality

### Skill Priority by Task Type

| Task Type | Primary Skill | Supporting Skills |
|-----------|---------------|-------------------|
| Market Analysis | FTR Trader | ML Engineer (models) |
| Model Development | ML Engineer | Python Expert (implementation) |
| Code Architecture | Python Expert | ML Engineer (requirements) |
| Trading Strategy | FTR Trader | ML Engineer, Python Expert |
| Performance Tuning | Python Expert | ML Engineer (model efficiency) |
| Risk Management | FTR Trader | ML Engineer (quantification) |

---

## Integration with Development Workflow

### Daily Development
- Use **Python Expert** by default for all code-related tasks
- Invoke **ML Engineer** when working on models, statistics, or forecasting
- Consult **FTR Trader** for market context, validation, and domain knowledge

### Code Reviews
```
1. Python Expert: Code quality, type safety, performance
2. ML Engineer: Statistical validity, model correctness
3. FTR Trader: Market realism, domain appropriateness
```

### Research Projects
```
1. FTR Trader: Define research question and hypothesis
2. ML Engineer: Design experimental approach
3. Python Expert: Implement data pipeline and analysis
4. ML Engineer: Execute analysis and interpret results
5. FTR Trader: Validate findings against market knowledge
```

---

## Quick Activation Commands

### Implicit Activation (Auto-detected)
Claude will automatically apply relevant skill expertise based on:
- Code-related tasks → Python Expert
- Statistical/ML tasks → ML Engineer
- Market/trading tasks → FTR Trader

### Explicit Activation (Recommended for Clarity)
```bash
# Single skill
"As the Python expert, [your request]"
"As the ML engineer, [your request]"
"As the FTR trader, [your request]"

# Multiple skills
"Python expert and ML engineer: [your request]"
"All skills: [your request]"

# Skill-specific review
"Python expert: review this code"
"ML engineer: validate this model"
"FTR trader: assess this strategy"
```

---

## Skill Capabilities Summary

### Python Expert
- ✅ Code quality and best practices
- ✅ Performance optimization
- ✅ Type safety and testing
- ✅ Library architecture
- ✅ Production deployment
- ❌ Statistical model selection
- ❌ Market domain knowledge

### ML Engineer
- ✅ Statistical modeling
- ✅ Time series forecasting
- ✅ Model evaluation
- ✅ Feature engineering
- ✅ Hyperparameter tuning
- ⚠️ Code implementation (basic)
- ❌ Market domain knowledge

### FTR Trader
- ✅ Market analysis
- ✅ Trading strategies
- ✅ Congestion forecasting
- ✅ Risk management
- ✅ Portfolio optimization
- ⚠️ Model implementation (conceptual)
- ❌ Code development

---

## Notes

- Skills are **context-aware**: They understand the project structure and objectives
- Skills are **complementary**: Designed to work together on complex problems
- Skills are **always active**: No need to "load" them, just reference when needed
- Skills provide **$2000/hour consultant-level** expertise in their domains

---

## Maintenance

To update or modify skills:
1. Edit the corresponding .md file in `.claude/` directory
2. Skills are automatically reloaded with each new session
3. Changes take effect immediately in new conversations

For questions or issues with skills, reference this guide.
