import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class PricingCalculator:
    def __init__(self, data: Dict[str, Any]):
        # Existing initializations...
        self.initial_price = data['initial_price']
        self.initial_quantity = data['initial_quantity']
        self.elasticity = data['price_elasticity']
        self.fixed_costs = data['fixed_costs']
        self.variable_costs = data['variable_costs']
        self.competitor_price = data['competitor_price']
        self.market_share = data.get('market_share', 0.5)
        
        # New parameters
        self.seasonality_factor = data.get('seasonality_factor', 1.0)
        self.quality_index = data.get('quality_index', 1.0)
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

    def seasonal_demand_adjustment(self, base_demand: float) -> float:
        """Adjust demand based on seasonality"""
        month = datetime.now().month
        seasonal_factor = 1 + self.seasonality_factor * np.sin(2 * np.pi * month / 12)
        return base_demand * seasonal_factor

    def quality_adjusted_price(self, price: float) -> float:
        """Adjust price perception based on quality index"""
        return price / self.quality_index

    def demand_function(self, price: float) -> float:
        """Enhanced demand function with quality and seasonality"""
        base_demand = self.initial_quantity * (self.initial_price / self.quality_adjusted_price(price)) ** self.elasticity
        seasonal_demand = self.seasonal_demand_adjustment(base_demand)
        market_growth_adjustment = 1 + self.market_growth_rate
        return seasonal_demand * market_growth_adjustment

    def calculate_price_bounds(self) -> tuple:
        """Calculate price bounds ensuring minimum margin"""
        min_price = self.variable_costs / (1 - self.min_margin)
        max_price = self.initial_price * 3.0  # Upper limit
        return (min_price, max_price)

    def calculate_optimal_price(self) -> Dict[str, float]:
        """Enhanced optimal price calculation"""
        bounds = [self.calculate_price_bounds()]
        
        result = minimize(
            lambda p: -self.profit_function(p[0]), 
            x0=[self.initial_price],
            bounds=bounds,
            method='L-BFGS-B'
        )

        optimal_price = result.x[0]
        optimal_quantity = self.demand_function(optimal_price)
        margin = (optimal_price - self.variable_costs) / optimal_price

        return {
            "optimal_price": round(optimal_price, 2),
            "optimal_quantity": round(optimal_quantity, 2),
            "revenue": round(self.revenue_function(optimal_price), 2),
            "profit": round(self.profit_function(optimal_price), 2),
            "market_share": round(self.competitor_impact(optimal_price), 3),
            "break_even_point": round(self.fixed_costs / (optimal_price - self.variable_costs), 2),
            "profit_margin": round(margin, 3),
            "seasonality_impact": round(self.seasonal_demand_adjustment(1.0), 3),
            "quality_adjusted_price": round(self.quality_adjusted_price(optimal_price), 2)
        }

    def sensitivity_analysis(self, variable: str, range_percent: float = 0.2) -> Dict[str, list]:
        """Perform sensitivity analysis on key variables"""
        base_value = getattr(self, variable)
        variations = np.linspace(base_value * (1-range_percent), base_value * (1+range_percent), 5)
        results = []
        
        for var in variations:
            setattr(self, variable, var)
            result = self.calculate_optimal_price()
            results.append(result)
            
        setattr(self, variable, base_value)  # Reset to original value
        return results

    def predict_competitor_response(self, price_change: float) -> Dict[str, float]:
        """Predict how competitors might react to our price changes"""
        predicted_responses = {}
        for competitor in self.market_players:
            response_likelihood = np.random.binomial(1, self.competitor_responsiveness)
            if response_likelihood:
                response_magnitude = price_change * np.random.uniform(0.5, 1.0)
                predicted_responses[competitor] = response_magnitude
        return predicted_responses

    def segment_demand(self, price: float) -> Dict[str, float]:
        """Calculate demand for different market segments"""
        segment_demands = {}
        for segment, attributes in self.segments.items():
            segment_quantity = (
                self.initial_quantity * 
                attributes['size'] * 
                (self.initial_price / price) ** 
                (self.elasticity * attributes['price_sensitivity'])
            )
            segment_demands[segment] = segment_quantity
        return segment_demands

    def calculate_optimal_order_quantity(self, demand: float) -> Dict[str, float]:
        """Calculate Economic Order Quantity (EOQ)"""
        eoq = np.sqrt((2 * demand * self.order_cost) / self.holding_cost_rate)
        reorder_point = (demand / 365) * self.lead_time
        
        return {
            "economic_order_quantity": eoq,
            "reorder_point": reorder_point,
            "annual_order_cost": (demand / eoq) * self.order_cost,
            "annual_holding_cost": (eoq / 2) * self.holding_cost_rate
        }

    def forecast_optimal_prices(self, periods: int = 12) -> List[Dict[str, float]]:
        """Generate optimal prices for future periods using time series analysis"""
        # Fit model to historical data
        model = ExponentialSmoothing(
            self.historical_demands,
            seasonal_periods=12,
            trend='add',
            seasonal='add'
        ).fit()
        
        # Forecast demand
        demand_forecast = model.forecast(periods)
        
        # Calculate optimal prices for each forecasted demand
        price_schedule = []
        for period, forecasted_demand in enumerate(demand_forecast):
            self.initial_quantity = forecasted_demand
            optimal_result = self.calculate_optimal_price()
            price_schedule.append({
                'period': period + 1,
                'forecasted_demand': forecasted_demand,
                'optimal_price': optimal_result['optimal_price'],
                'expected_profit': optimal_result['profit']
            })
            
        return price_schedule

    def calculate_risk_adjusted_price(self) -> Dict[str, float]:
        """Calculate optimal price with risk adjustment"""
        base_result = self.calculate_optimal_price()
        
        # Calculate Value at Risk (VaR)
        profit_distribution = []
        for _ in range(1000):
            # Monte Carlo simulation with random demand variations
            demand_shock = np.random.normal(1, 0.2)
            self.initial_quantity *= demand_shock
            result = self.calculate_optimal_price()
            profit_distribution.append(result['profit'])
            
        var = np.percentile(profit_distribution, (1 - self.confidence_level) * 100)
        
        # Adjust price based on risk tolerance
        risk_adjusted_price = (
            base_result['optimal_price'] * 
            (1 + (1 - self.risk_tolerance) * (var / base_result['profit']))
        )
        
        return {
            **base_result,
            "risk_adjusted_price": round(risk_adjusted_price, 2),
            "value_at_risk": round(var, 2),
            "price_adjustment": round((risk_adjusted_price - base_result['optimal_price']) / base_result['optimal_price'] * 100, 2)
        }