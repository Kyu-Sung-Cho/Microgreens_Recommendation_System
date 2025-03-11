# Microgreens_Recommendation_System

## Project Overview
An advanced recommendation system for Koppert Cress microgreens, leveraging machine learning techniques to provide personalized product suggestions to chefs. This project demonstrates the effective combination of TF-IDF and SVD approaches while maintaining interpretable recommendations through a weighted scoring system.

## Business Problem
Chefs often struggle to discover new microgreen varieties that match their culinary style. This project aims to:
- Develop accurate recommendation models for microgreen products
- Increase product adoption among chefs
- Provide relevant suggestions based on chef preferences and seasonal trends
- Improve chef satisfaction through personalized recommendations

## Data and Features
**Source**: Koppert Cress database of chef profiles, products, and purchase history  
**Size**: Database with multiple interconnected tables covering chefs, microgreens, and interactions  
**Target Variable**: Recommendation relevance determined by chef adoption and feedback  

### Feature Categories:
#### Chef Data
- Cuisine Type
- Flavor Preferences
- Purchase History
- Feedback Scores
- Geographic Location

#### Product Data
- Flavor Profiles
- Nutritional Values
- Seasonal Availability
- Stock Quantities
- Price Points

#### Interaction Data
- Purchase Recency
- Purchase Frequency
- Feedback Scores
- Pairing Information
- Seasonal Recommendations

---

## Preprocessing and Feature Engineering

### Data Cleaning
- Handled missing flavor profiles
- Normalized feedback scores
- Structured chef preference data

### Feature Engineering
- TF-IDF vectorization of flavor profiles
- Chef-microgreen interaction matrix
- Recency-Frequency-Feedback (RFM) scoring
- Cuisine-microgreen pairing relevance scores

---

## Validation Strategy
- A/B testing with chef groups
- Comparison of traditional vs. enhanced models
- Product adoption rate tracking
- Chef feedback analysis

---

## Model Architecture

### Traditional Recommendation Components:
- **Purchase History Analysis (25%)**:
  - Recency, frequency, and feedback metrics
  - Normalization to 0-1 range
- **Cuisine Type Matching (25%)**:
  - Pairing scores from dish_pairings table
  - Weighted by chef's cuisine preferences
- **Basic Flavor Matching (25%)**:
  - String-based keyword matching
  - Normalization by match count
- **Seasonal Relevance (15%)**:
  - Current season determination
  - Seasonal recommendation scores
- **Stock Availability (10%)**:
  - Normalized stock quantity scores
  - Ensures recommended products are available

### Enhanced Recommendation Components:
- **Traditional Base (50%)**:
  - Combined score from traditional approach
  - Provides foundation based on business rules
- **TF-IDF Flavor Analysis (30%)**:
  - Vector representation of flavor profiles
  - Cosine similarity between chef preferences and products
  - Emphasizes distinctive flavor characteristics
- **SVD Collaborative Filtering (20%)**:
  - Matrix factorization of chef-microgreen interactions
  - Captures latent patterns and relationships
  - Addresses the cold start problem

---

## Model Performance

### Implementation Results:
| Metric | Before System | After Implementation | Improvement |
|--------|--------------|---------------------|------------|
| Product Adoption Rate | Baseline | +14% | 14% |
| Chef Satisfaction | 3.2/5.0 | 4.1/5.0 | +28% |
| Order Diversity | 2.4 products | 3.6 products | +50% |

### Model Comparison (Chef Acceptance Rate):
| Model | Acceptance Rate | Quality Feedback |
|-------|----------------|-----------------|
| Traditional | 62% | 3.8/5.0 |
| Enhanced | 76% | 4.1/5.0 |
| Improvement | +14% | +8% |

---

## Key Insights

### Critical Recommendation Factors:
- **Flavor Profile Matching** (Most influential):  
  TF-IDF significantly outperforms basic string matching for flavor relevance.
- **Previous Purchase Patterns**:  
  Chef loyalty to specific microgreens strongly indicates future purchases.
- **Cuisine-Specific Pairings**:  
  Different cuisine types show distinct microgreen preferences.

### Secondary Factors:
- Seasonal availability
- Price sensitivity varies by chef segment
- Stock availability impacts adoption of recommendations
- Regional variations in preferences

---

## Why We Chose the Enhanced Model

While the traditional recommendation model showed decent performance, we selected the **enhanced model** for the following reasons:

1. **Superior Performance Metrics**:  
   The enhanced model achieved a 14% higher adoption rate and better chef satisfaction scores.

2. **Better Text Analysis Capabilities**:  
   TF-IDF provides more nuanced understanding of flavor profiles compared to basic string matching.

3. **Discovery of Latent Patterns**:  
   SVD identifies hidden relationships between chefs and microgreens that aren't explicitly captured in the data.

4. **Cold Start Problem Handling**:  
   The enhanced model performs better with new chefs or microgreens with limited interaction history.

5. **Balanced Approach**:  
   Combining traditional business rules with advanced ML techniques provides both relevance and explainability.

---

## Limitations & Future Improvements

### Model Limitations:
- **Limited chef preference data**
- Seasonal variation effects not fully captured
- Cold start problem for brand new chefs
- Regional cuisine variations need refinement

### Future Work:
- Incorporate image-based similarity
- Add recipe-specific recommendations
- Implement real-time feedback loop
- Explore deep learning approaches
- Develop time-series analysis for trend prediction
- Add personalized explanation generation

---

## Files:

The actual code with details about this project is in "**Microgreens_Recommendation_System.py**"
