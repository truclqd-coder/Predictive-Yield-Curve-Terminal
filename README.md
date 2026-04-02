# 🛰️ QuantYield: Predictive Yield Curve Terminal `<EXEC>`

**QuantYield** is a high-density analytical terminal designed to synthesize historical interest rate data with user-defined macro "shocks." Powered by the **OLS_REGRESSION_V1** engine, the tool identifies the "Fair Value" of the US 10-Year Treasury Note by analyzing the structural relationship between short-term policy rates and long-term growth expectations.

---

## 🛠️ System Architecture: OLS_REGRESSION_V1

The engine utilizes **Ordinary Least Squares (OLS) Regression** to model the "belly" of the yield curve (10Y) as a function of the "wings" (3-Month and 30-Year yields). 

* **The Prior:** 5 years of historical daily Treasury data via Yahoo Finance.
* **The Synthesis:** A multi-dimensional correlation matrix mapping how the 10Y maturity historically reacts to shifts in the 3M (Fed Policy) and 30Y (Inflation/Growth).
* **The Posterior:** A real-time predictive output that calibrates the historical model against user-defined basis point (**bps**) shocks.

---

## 🚀 Key Features

### 1. **Live Market Diagnostics**
Real-time tracking of the US 10Y Treasury vs. the AI-derived **Fair Value**. Includes a live delta indicator to show if the market is currently "Rich" or "Cheap" relative to historical anchors.

### 2. **<SENS> Projection Matrix**
A dynamic sensitivity grid that visualizes 25 potential 10Y yield outcomes based on simultaneous shifts in the 3M and 30Y yields. This allows for instant "at-a-glance" risk assessment.

### 3. **Manual Shock Synthesis (AI Calibration)**
A dedicated sidebar instrument that allows users to "stress-test" the curve:
* **Short-Rate Shocks:** Simulate Federal Reserve hawkish/dovish pivots.
* **Long-Bond Shocks:** Simulate shifts in long-term inflation or terminal growth.
* **Calibration Guide:** Integrated instructions to guide users through the synthesis process.

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/QuantYield-Terminal.git](https://github.com/your-username/QuantYield-Terminal.git)
   cd QuantYield-Terminal

--
**Developed by Truc Nguyen, Product Support Specialist, AI Knowledge Lead at Moody's Analytics, Inc.**
