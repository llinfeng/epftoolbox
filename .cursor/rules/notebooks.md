# Jupyter Notebook Rules

## Scope
Applies to: `*.ipynb` files and Python files in `notebooks/` or `Notebooks/` directories.

## Core Rules

### 1. Minimal Edits
- Make targeted, local changes to specific cells
- Preserve notebook structure and flow
- Avoid rewriting large code blocks

### 2. Data Exploration: Tabulation Only
- Use pandas tabulation methods: `df.info()`, `df.describe()`, `df.value_counts()`, `df.groupby().agg()`
- No visualizations unless explicitly requested

### 3. No Plotting
- Do not import or use matplotlib, seaborn, plotly, etc. unless user requests visualization
- Suggest visualization if helpful, but do not create it

### 4. Cell Size Limit
- Keep cells ≤ 20 lines
- One purpose per cell
- Split complex operations across multiple cells

### 5. No External Files
- Keep code in notebook cells only
- Do not create separate modules, scripts, or helper files

## Examples

### ✅ Tabulation
```python
df.info()
df.describe()
df['column'].value_counts()
df.groupby('category')['value'].agg(['mean', 'std', 'count'])
```

### ❌ Plotting (unless requested)
```python
import matplotlib.pyplot as plt
df.plot(kind='hist')
plt.show()
```

### ✅ Short, focused cell
```python
df.isnull().sum()
```

### ❌ Long, multi-purpose cell
```python
# Avoid 30+ line cells doing multiple things
```

