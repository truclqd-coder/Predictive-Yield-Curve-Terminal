# Yield Curve Predictive Terminal

An interactive **Fixed Income Relative Value (RV)** tool built with Python and Scikit-Learn.

## 🚀 Features
- **Predictive Engine:** Uses Multiple Linear Regression to calculate 10Y Treasury Fair Value.
- **Scenario Analysis:** Adjust Fed Policy and Inflation shocks to see real-time yield predictions.
- **Bloomberg UI:** High-contrast terminal aesthetic for professional information density.
- **Sensitivity Matrix:** Visualizes the "Risk Surface" across multiple yield curve shifts.

## 🛠️ Tech Stack
- **AI/ML:** Scikit-Learn (Linear Regression)
- **Data:** YFinance API (Treasury Yields)
- **Frontend:** Streamlit
- **Math:** NumPy / Pandas

## 📈 Methodology
The model solves for:  
**$Yield_{10Y} = \beta_0 + \beta_1(Yield_{3M}) + \beta_2(Yield_{30Y}) + \epsilon$**

Where $\epsilon$ represents the market dislocation (Rich/Cheap signal).

--
Developed by Truc Nguyen, Product Support Specialist, AI Knowledge Lead at Moody's Analytics, Inc.
