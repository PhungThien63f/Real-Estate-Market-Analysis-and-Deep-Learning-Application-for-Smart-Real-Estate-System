# Real Estate Market Analysis and Deep Learning Application for Smart Real Estate System

## Team Members :
- 20280041: Nguyễn Đình Hưng
- 20280047: Nguyễn Lê Diệu Huyền
- 20280075: Phạm Thiên Phụng

## Information: 

  ### Step 1:
- Using the Selenium library, data was crawled from **Batdongsan** and **Chotot** websites.
  ### Step 2:
- Using Power BI to visualize the data
  ![Dahboard-1](https://github.com/PhungThien63f/Real-Estate-Market-Analysis-and-Deep-Learning-Application-for-Smart-Real-Estate-System/blob/main/Dashboard/Dash-1.png)

  #### **First Dashboard: Overview of the Real Estate Market in Ho Chi Minh City (2023)**  

This dashboard provides an **overview of the real estate situation in Ho Chi Minh City** for the year 2023, primarily from **September to early December**.  

- **Types of Real Estate:** The most common property types are **Apartments and Alley Houses**.  
- **Price Distribution:** The majority of properties fall within the **1 to 10 billion VND** range.  
- **District Distribution:** **Thu Duc District** has the highest number of real estate listings, followed by **District 7** and **Tan Phu District**.  
- **Legal Status:** The majority of properties have either a **Red Book (Ownership Certificate) or an Unknown status**.

  ![Dahboard-2](https://github.com/PhungThien63f/Real-Estate-Market-Analysis-and-Deep-Learning-Application-for-Smart-Real-Estate-System/blob/main/Dashboard/Dash-2.png)
  
  #### **Real Estate Price Trends**  

- Although **Villas and Townhouses** are less common, they have **very high prices**. **Apartments** tend to have the **lowest prices** among all property types.  
- **Districts near the city center (Ben Thanh area)**, such as **District 1 and District 3**, have **higher average prices**, while **outlying districts** like **Binh Chanh, Hoc Mon, and Cu Chi** have **significantly lower average prices**.  
- Given the **government's tightening legal regulations on real estate**, properties with **clear legal status (Ownership Certificate available)** tend to have **higher prices**.  
- **Furnished properties** are generally priced **higher** than **unfurnished** ones.  

### Step 3:
- The dataset was then analyzed, and both **Machine Learning (ML)** and **Deep Learning (DL)** models were applied to predict housing prices in **Ho Chi Minh City**.
  
  **RESULT**
  
  | MODEL                         | SCORE  | MSE     | RMSE    | MAE    |
|-------------------------------|--------|---------|---------|--------|
| LINEAR REGRESSION             | 0.6364 | 10.9652 | 3.3113  | 2.2173 |
| RANDOM FOREST REGRESSION      | 0.8576 | 4.2944  | 2.0723  | 1.1346 |
| K-NEAREST NEIGHBOR REGRESSION | 0.8231 | 5.3372  | 2.3102  | 1.2392 |
| XGBOOST REGRESSION            | 0.8660 | 4.0401  | 2.0100  | 1.1483 |
| REGRESSION NEURAL NETWORK     | **0.8801** | **3.2117** | **1.6831** | **0.9224** |

  
### Step 4:
- Finally, the best-performing model is **Regression Neural Network** was selected to develop a **house price prediction application**.

  ![House Prediction](https://github.com/PhungThien63f/Real-Estate-Market-Analysis-and-Deep-Learning-Application-for-Smart-Real-Estate-System/blob/main/Code/assets/price_prediction.png)
  
