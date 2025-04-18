# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def brown_gibson_model(objective_factors, subjective_factors, factor_weights, k=1):
    """
    Implement the Brown-Gibson model for facility location selection.
    
    Parameters:
    -----------
    objective_factors : dict
        Dictionary with location names as keys and costs as values.
    
    subjective_factors : dict of dicts
        Dictionary with location names as keys and another dictionary as values.
        The nested dictionary contains factor names as keys and scores (0-1) as values.
    
    factor_weights : dict
        Dictionary with factor names as keys and weights (0-1) as values.
        Sum of weights should be 1.
    
    k : float, optional (default=1)
        Weight given to objective factors vs subjective factors.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with locations and their final preference measure.
    """
    # Validate inputs
    if not 0 <= k <= 1:
        raise ValueError("k must be between 0 and 1 inclusive")
    
    if abs(sum(factor_weights.values()) - 1.0) > 0.001:
        raise ValueError("Sum of factor weights must be 1")
    
    locations = list(objective_factors.keys())
    
    # Calculate Objective Factor Measure (OFM)
    total_reciprocal = sum(1/cost for cost in objective_factors.values())
    OFM = {loc: (1/objective_factors[loc])/total_reciprocal for loc in locations}
    
    # Calculate Subjective Factor Measure (SFM)
    SFM = {}
    for location in locations:
        location_sfm = 0
        for factor, weight in factor_weights.items():
            if factor in subjective_factors[location]:
                location_sfm += subjective_factors[location][factor] * weight
        SFM[location] = location_sfm
    
    # Calculate Location Preference Measure (LPM)
    LPM = {loc: k * OFM[loc] + (1 - k) * SFM[loc] for loc in locations}
    
    # Create result DataFrame
    results = pd.DataFrame({
        'Location': locations,
        'Objective Factor Measure': [OFM[loc] for loc in locations],
        'Subjective Factor Measure': [SFM[loc] for loc in locations],
        'Location Preference Measure': [LPM[loc] for loc in locations]
    })
    
    # Sort by preference measure (higher is better)
    results = results.sort_values('Location Preference Measure', ascending=False).reset_index(drop=True)
    
    return results

def main():
    st.set_page_config(page_title="Brown-Gibson Model Tool", layout="wide")
    
    st.title("Brown-Gibson Location Decision Model")
    st.write("""
    This tool helps you evaluate potential facility locations using the Brown-Gibson model,
    which combines objective factors (like costs) and subjective factors (like preferences)
    to determine optimal facility locations.
    """)
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("How to use this tool")
        st.write("""
        1. Enter the number of locations or suppliers and factors
        2. Fill in the objective costs for each location
        3. Rate subjective factors for each location (0-1)
        4. Set weights for subjective factors (must sum to 1)
        5. Adjust the k-value (balance between objective and subjective)
        6. View your results
        """)
        
        st.header("About the Brown-Gibson Model")
        st.write("""
        The Brown-Gibson model is a multi-criteria decision making tool used in operations research.
        It considers both quantitative (objective) and qualitative (subjective) factors to rank
        potential locations.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("1. Define Your Locations or suppliers and Factors")
        num_locations = st.number_input("Number of Locations or suppliers:", min_value=2, max_value=10, value=3, step=1)
        factors = st.text_input("Subjective Factors (comma separated):", value="proximity,labor quality,environment").split(',')
        factors = [f.strip() for f in factors]
        
        st.header("2. Objective Factors (Costs)")
        st.write("Enter the costs for each location (lower is better):")
        
        costs = {}
        for i in range(num_locations):
            location_name = st.text_input(f"Location {i+1} Name:", value=f"Location {chr(65+i)}")
            location_cost = st.number_input(f"Cost for {location_name}:", min_value=1.0, value=100.0 + i*10)
            costs[location_name] = location_cost
            st.divider()
    
    with col2:
        st.header("3. Subjective Factors")
        st.write("Rate each factor from 0.0 (worst) to 1.0 (best):")
        
        # Create a structure to hold all subjective factors
        subjective_data = {}
        
        for loc_name in costs.keys():
            subjective_data[loc_name] = {}
            st.subheader(f"Ratings for {loc_name}")
            
            for factor in factors:
                subjective_data[loc_name][factor] = st.slider(
                    f"{factor} for {loc_name}:", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5,
                    step=0.1
                )
            st.divider()
        
        st.header("4. Factor Weights")
        st.write("Allocate weights to each subjective factor (must sum to 1.0):")
        
        # Initialize weights equally
        equal_weight = 1.0 / len(factors)
        weights = {}
        total_weight = 0.0
        
        for i, factor in enumerate(factors[:-1]):  # All but the last factor
            weights[factor] = st.slider(
                f"Weight for {factor}:", 
                min_value=0.0, 
                max_value=1.0, 
                value=equal_weight,
                step=0.05
            )
            total_weight += weights[factor]
        
        # Last factor weight is computed to ensure sum is 1.0
        last_factor = factors[-1]
        last_weight = round(1.0 - total_weight, 2)
        if last_weight < 0:
            st.error(f"Weight allocations exceed 1.0 by {-last_weight}. Please reduce other weights.")
            last_weight = 0.0
        
        weights[last_factor] = st.number_input(
            f"Weight for {last_factor} (auto-calculated):", 
            min_value=0.0, 
            max_value=1.0, 
            value=last_weight,
            disabled=True
        )
        
        st.header("5. Objective-Subjective Balance")
        k_value = st.slider(
            "k value (weight for objective factors):", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            step=0.1,
            help="0 = only subjective factors, 1 = only objective factors"
        )
    
    # Calculate and show results
    st.header("Results")
    
    if st.button("Calculate Rankings"):
        try:
            results = brown_gibson_model(costs, subjective_data, weights, k_value)
            
            # Display results table
            st.subheader("Location Rankings")
            st.dataframe(results)
            
            # Create visualizations
            st.subheader("Visualization")
            
            # Bar chart of final scores
            fig1 = px.bar(
                results, 
                x='Location', 
                y='Location Preference Measure',
                title="Location Preference Measure (Higher is Better)",
                color='Location Preference Measure',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig1)
            
            # Component breakdown
            fig2 = px.bar(
                results, 
                x='Location', 
                y=['Objective Factor Measure', 'Subjective Factor Measure'],
                title="Breakdown of Objective vs Subjective Measures",
                barmode='group'
            )
            st.plotly_chart(fig2)
            
            # Recommendations
            st.subheader("Recommendation")
            best_location = results.iloc[0]['Location']
            best_score = results.iloc[0]['Location Preference Measure']
            st.success(f"Based on your inputs, **{best_location}** is the recommended location with a score of {best_score:.4f}.")
            
            # Sensitivity analysis hint
            st.info("Try adjusting the k value to see how sensitive your decision is to the balance between objective and subjective factors.")
            
        except ValueError as e:
            st.error(f"Error in calculation: {e}")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
