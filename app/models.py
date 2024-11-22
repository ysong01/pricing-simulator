import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class PricingCalculator:
    def __init__(self, data: Dict[str, Any]):
        # Convert all inputs to float
        try:
            self.initial_price = float(data['initial_price'])
            self.initial_quantity = float(data['initial_quantity'])
            self.elasticity = float(data['price_elasticity'])
            self.fixed_costs = float(data['fixed_costs'])
            self.variable_costs = float(data['variable_costs'])
            self.competitor_price = float(data['competitor_price'])
            self.market_share = float(data.get('market_share', 0.5))
            self.seasonality_factor = float(data.get('seasonality_factor', 1.0))
            self.quality_index = float(data.get('quality_index', 1.0))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input data: {str(e)}")

        # New parameters
        self.market_growth_rate = data.get('market_growth_rate', 0.02)
        self.min_margin = data.get('min_margin', 0.1)  # Minimum profit margin

        # Add to existing initializations
        self.competitor_responsiveness = data.get('competitor_responsiveness', 0.3)
        self.competitor_prices_history = data.get('competitor_prices_history', [])
        self.market_players = data.get('market_players', [])
        self.segments = data.get('market_segments', {
            'premium': {'price_sensitivity': 0.5, 'size': 0.3},
            'middle': {'price_sensitivity': 1.0, 'size': 0.5},
            'budget': {'price_sensitivity': 1.5, 'size': 0.2}
        })

        # Add to existing initializations
        self.order_cost = data.get('order_cost', 100)
        self.holding_cost_rate = data.get('holding_cost_rate', 0.1)
        self.lead_time = data.get('lead_time', 14)

        # Add to existing initializations
        self.historical_prices = data.get('historical_prices', [])
        self.historical_demands = data.get('historical_demands', [])

        # Add to existing initializations
        self.risk_tolerance = data.get('risk_tolerance', 0.5)
        self.confidence_level = data.get('confidence_level', 0.95)

    def competitor_impact(self, price: float) -> float:
        """Calculate market share adjustment based on competitor pricing"""
        try:
            price_ratio = float(price) / float(self.competitor_price)
            # Adjust market share based on price difference
            market_share = self.market_share * (2 - price_ratio)
            # Ensure market share stays between 0 and 1
            return max(0.0, min(1.0, market_share))
        except Exception as e:
            raise ValueError(f"Error in competitor impact calculation: {str(e)}")

    def seasonal_demand_adjustment(self, base_demand: float) -> float:
        """Adjust demand based on seasonality"""
        try:
            month = datetime.now().month
            seasonal_factor = 1 + self.seasonality_factor * np.sin(2 * np.pi * month / 12)
            return float(base_demand) * seasonal_factor
        except Exception as e:
            raise ValueError(f"Error in seasonal adjustment calculation: {str(e)}")

    def quality_adjusted_price(self, price: float) -> float:
        """Adjust price perception based on quality index"""
        try:
            return float(price) / float(self.quality_index)
        except Exception as e:
            raise ValueError(f"Error in quality price adjustment: {str(e)}")

    def demand_function(self, price: float) -> float:
        """Calculate quantity demanded at given price using constant elasticity demand function"""
        try:
            price = float(price)
            base_demand = float(self.initial_quantity) * (
                float(self.initial_price) / self.quality_adjusted_price(price)
            ) ** float(self.elasticity)
            seasonal_demand = self.seasonal_demand_adjustment(base_demand)
            market_share_adjustment = self.competitor_impact(price)
            return seasonal_demand * market_share_adjustment
        except Exception as e:
            raise ValueError(f"Error in demand calculation: {str(e)}")

    def revenue_function(self, price: float) -> float:
        """Calculate revenue at given price"""
        try:
            quantity = self.demand_function(float(price))
            return float(price) * quantity
        except Exception as e:
            raise ValueError(f"Error in revenue calculation: {str(e)}")

    def cost_function(self, quantity: float) -> float:
        """Calculate total costs for given quantity"""
        try:
            return float(self.fixed_costs) + (float(self.variable_costs) * float(quantity))
        except Exception as e:
            raise ValueError(f"Error in cost calculation: {str(e)}")

    def profit_function(self, price: float) -> float:
        """Calculate profit at given price"""
        try:
            quantity = self.demand_function(float(price))
            revenue = self.revenue_function(float(price))
            costs = self.cost_function(quantity)
            return revenue - costs
        except Exception as e:
            raise ValueError(f"Error in profit calculation: {str(e)}")

    def calculate_price_bounds(self) -> tuple:
        """Calculate price bounds ensuring minimum margin"""
        try:
            min_price = float(self.variable_costs) * 1.1  # Minimum 10% margin
            max_price = float(self.initial_price) * 3.0  # Upper limit
            return (min_price, max_price)
        except Exception as e:
            raise ValueError(f"Error in price bounds calculation: {str(e)}")

    def calculate_optimal_price(self) -> Dict[str, float]:
        """Find optimal price that maximizes profit"""
        try:
            # Get price bounds
            price_bounds = self.calculate_price_bounds()
            
            # Minimize negative profit (equivalent to maximizing profit)
            result = minimize(
                lambda p: -self.profit_function(float(p[0])), 
                x0=[float(self.initial_price)],
                bounds=[price_bounds],
                method='L-BFGS-B'
            )

            optimal_price = float(result.x[0])
            optimal_quantity = self.demand_function(optimal_price)
            
            # Calculate margin
            margin = (optimal_price - self.variable_costs) / optimal_price
            
            return {
                "optimal_price": round(optimal_price, 2),
                "optimal_quantity": round(optimal_quantity, 2),
                "revenue": round(self.revenue_function(optimal_price), 2),
                "profit": round(self.profit_function(optimal_price), 2),
                "market_share": round(self.competitor_impact(optimal_price), 3),
                "break_even_point": round(
                    float(self.fixed_costs) / (optimal_price - float(self.variable_costs)), 
                    2
                ),
                "profit_margin": round(margin, 3),
                "seasonality_impact": round(self.seasonal_demand_adjustment(1.0), 3),
                "quality_adjusted_price": round(self.quality_adjusted_price(optimal_price), 2)
            }
        except Exception as e:
            raise ValueError(f"Error in optimization: {str(e)}")