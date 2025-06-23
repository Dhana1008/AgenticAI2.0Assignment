# Enhanced Travel Agent with Step-by-Step Workflow
import os
from langchain.tools import tool
from serpapi import search
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import requests
from datetime import datetime, timedelta

load_dotenv()

# API Keys
serp_apikey = os.getenv("SERP_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
open_weather_api_key = os.getenv("OPEN_WEATHER_API_KEY")

# === STEP 1: SEARCH ATTRACTIONS AND ACTIVITIES ===

@tool('get_attraction')
def _get_attraction(city: str):
    '''Search for attractions in the specified city'''
    tavily_client = TavilyClient(api_key=tavily_api_key)
    query = f"top attractions tourist places things to do in {city}"
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=["tripadvisor.com", "lonelyplanet.com", "timeout.com", "google.com"]
        )
        
        attractions = []
        for result in response.get('results', [])[:6]:
            attraction = {
                'name': result.get('title', 'N/A'),
                'description': result.get('content', 'N/A')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', 'N/A'),
                'url': result.get('url', 'N/A'),
                'score': result.get('score', 0)
            }
            attractions.append(attraction)
        
        return {
            'city': city,
            'attractions': attractions,
            'total_found': len(attractions)
        }
    except Exception as e:
        return {
            'city': city,
            'attractions': [],
            'error': f"Error fetching attractions: {str(e)}"
        }

@tool('search_restaurants')
def search_restaurants(city: str):
    '''Search for restaurants and food options in the city'''
    tavily_client = TavilyClient(api_key=tavily_api_key)
    query = f"best restaurants food places to eat in {city}"
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=["zomato.com", "tripadvisor.com", "timeout.com", "lonelyplanet.com"]
        )
        
        restaurants = []
        for result in response.get('results', [])[:5]:
            restaurant = {
                'name': result.get('title', 'N/A'),
                'description': result.get('content', 'N/A')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', 'N/A'),
                'url': result.get('url', 'N/A'),
                'score': result.get('score', 0)
            }
            restaurants.append(restaurant)
        
        return {
            'city': city,
            'restaurants': restaurants,
            'total_found': len(restaurants)
        }
    except Exception as e:
        return {
            'city': city,
            'restaurants': [],
            'error': f"Error fetching restaurants: {str(e)}"
        }

@tool('search_activities')
def search_activities(city: str):
    '''Search for activities and experiences in the city'''
    tavily_client = TavilyClient(api_key=tavily_api_key)
    query = f"activities experiences tours things to do {city}"
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=["viator.com", "getyourguide.com", "tripadvisor.com", "timeout.com"]
        )
        
        activities = []
        for result in response.get('results', [])[:5]:
            activity = {
                'name': result.get('title', 'N/A'),
                'description': result.get('content', 'N/A')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', 'N/A'),
                'url': result.get('url', 'N/A'),
                'score': result.get('score', 0)
            }
            activities.append(activity)
        
        return {
            'city': city,
            'activities': activities,
            'total_found': len(activities)
        }
    except Exception as e:
        return {
            'city': city,
            'activities': [],
            'error': f"Error fetching activities: {str(e)}"
        }

@tool('get_transport_pricing')
def _get_transport_pricing(city: str):
    '''Search for transportation pricing in the city'''
    tavily_client = TavilyClient(api_key=tavily_api_key)
    query = f"{city} public transport prices fares day pass weekly pass metro bus train tickets cost"
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=6,
            include_domains=["official-city-sites.com", "rome2rio.com", "timeout.com", "tripadvisor.com"]
        )
        
        pricing_info = []
        for result in response.get('results', [])[:6]:
            pricing = {
                'title': result.get('title', 'N/A'),
                'pricing_details': result.get('content', 'N/A')[:300] + '...' if len(result.get('content', '')) > 300 else result.get('content', 'N/A'),
                'url': result.get('url', 'N/A'),
                'score': result.get('score', 0)
            }
            pricing_info.append(pricing)
        
        return {
            'city': city,
            'pricing_info': pricing_info,
            'total_found': len(pricing_info)
        }
    except Exception as e:
        return {
            'city': city,
            'pricing_info': [],
            'error': f"Error fetching transport pricing: {str(e)}"
        }

# === STEP 2: WEATHER FORECASTING ===

@tool
def get_weather_forecast(city: str, date: str) -> Dict[str, Any]:
    '''Get weather forecast for travel planning'''
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
        today = datetime.now().date()
        days_difference = (target_date.date() - today).days
        
        if days_difference < 0:
            return {
                "status": "error",
                "city": city,
                "date": date,
                "message": "Cannot get weather for past dates"
            }
        elif days_difference > 5:
            return {
                "status": "error", 
                "city": city,
                "date": date,
                "message": "Weather forecast available only for next 5 days"
            }
        
        weather_data = _fetch_openweather_data(city, date, days_difference)
        return weather_data
        
    except ValueError:
        return {
            "status": "error",
            "city": city,
            "date": date,
            "message": "Invalid date format. Use YYYY-MM-DD"
        }
    except Exception as e:
        return {
            "status": "error",
            "city": city,
            "date": date, 
            "message": f"Weather service error: {str(e)}"
        }

def _fetch_openweather_data(city: str, date: str, days_diff: int) -> Dict[str, Any]:
    '''Fetch weather data from OpenWeatherMap API'''
    api_key = open_weather_api_key
    if not api_key:
        return {"error": "Weather API key not configured"}
    
    try:
        lat, lon, normalized_city = _get_city_coordinates(city, api_key)
        if not lat:
            return {"error": f"City '{city}' not found"}
        
        if days_diff == 0:
            return _get_current_weather(lat, lon, normalized_city, date, api_key)
        else:
            return _get_forecast_weather(lat, lon, normalized_city, date, api_key)
            
    except requests.exceptions.RequestException:
        return {"error": "Weather service connection failed"}
    except Exception as e:
        return {"error": f"Failed to get weather data: {str(e)}"}

def _get_city_coordinates(city: str, api_key: str) -> tuple:
    '''Get coordinates for city'''
    clean_city = city.strip().split(',')[0]
    geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": clean_city, "limit": 1, "appid": api_key}
    
    response = requests.get(geocoding_url, params=params, timeout=10)
    response.raise_for_status()
    
    geo_data = response.json()
    if not geo_data:
        return None, None, None
    
    location_info = geo_data[0]
    lat = location_info['lat']
    lon = location_info['lon']
    normalized_city = f"{location_info['name']}, {location_info['country']}"
    
    return lat, lon, normalized_city

def _get_current_weather(lat: float, lon: float, city: str, date: str, api_key: str) -> Dict[str, Any]:
    '''Get current weather'''
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    return {
        "status": "success",
        "city": city,
        "date": date,
        "temperature": {
            "min": round(data['main']['temp_min']),
            "max": round(data['main']['temp_max']),
            "current": round(data['main']['temp'])
        },
        "conditions": data['weather'][0]['description'].title(),
        "humidity": data['main']['humidity'],
        "wind_speed": round(data['wind']['speed'], 1),
        "rain_probability": 0,
        "data_type": "current"
    }

def _get_forecast_weather(lat: float, lon: float, city: str, date: str, api_key: str) -> Dict[str, Any]:
    '''Get forecast weather'''
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    target_forecasts = []
    for item in data['list']:
        forecast_datetime = datetime.fromtimestamp(item['dt'])
        if forecast_datetime.strftime('%Y-%m-%d') == date:
            target_forecasts.append(item)
    
    if not target_forecasts:
        return {"error": f"No forecast data available for {date}"}
    
    midday_forecast = min(target_forecasts, 
                         key=lambda x: abs(datetime.fromtimestamp(x['dt']).hour - 12))
    
    day_temps = [item['main']['temp'] for item in target_forecasts]
    
    return {
        "status": "success",
        "city": city,
        "date": date,
        "temperature": {
            "min": round(min(day_temps)),
            "max": round(max(day_temps)),
            "current": round(midday_forecast['main']['temp'])
        },
        "conditions": midday_forecast['weather'][0]['description'].title(),
        "humidity": midday_forecast['main']['humidity'],
        "wind_speed": round(midday_forecast['wind']['speed'], 1),
        "rain_probability": round(midday_forecast.get('pop', 0) * 100),
        "data_type": "forecast"
    }

# === STEP 3: HOTEL SEARCH AND COST ===

def get_hotel_details(dictionary):
    name = dictionary['name']
    rate = dictionary['rate_per_night'] if 'rate_per_night' in dictionary else 'NA'
    overall_rating = dictionary['overall_rating'] if 'overall_rating' in dictionary else 'NA'
    return {'property_name': name, 'rate_per_night': rate, 'rating': overall_rating}

@tool('hotel-search')
def hotel_search(place: str, check_in_date: str, check_out_date: str, 
                country_location: str, number_of_adults: int,
                number_of_children: int = 0, children_ages: List[int] = None,
                property_types: int = None, min_price: int = None, max_price: int = None):
    '''Search for hotels at the destination location'''
    params = {
        "engine": "google_hotels",
        "q": place,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": number_of_adults,
        "children": number_of_children,
        "children_ages": children_ages,
        "property_types": property_types,
        "currency": "INR",
        "gl": country_location,
        "hl": "en",
        "max_price": max_price,
        "min_price": min_price,
        "api_key": serp_apikey
    }

    results = search(params)
    return_dict = list(map(get_hotel_details, results['properties']))[:4]
    return {'place': place, 'hotel_details': return_dict}

@tool
def estimate_hotel_cost(price_per_night: float, total_nights: int) -> float:
    """Calculate total hotel cost for the stay"""
    return price_per_night * total_nights

@tool
def get_budget_range(total_budget: float) -> Dict[str, float]:
    """Get budget ranges for different categories"""
    return {
        "accommodation": total_budget * 0.40,  # 40% for hotels
        "food": total_budget * 0.25,          # 25% for food
        "transport": total_budget * 0.15,     # 15% for transport
        "activities": total_budget * 0.15,    # 15% for activities
        "miscellaneous": total_budget * 0.05  # 5% for misc
    }

# === STEP 4: COST CALCULATIONS ===

@tool
def add_costs(cost1: float, cost2: float) -> float:
    """Add two costs together"""
    return cost1 + cost2

@tool
def multiply_costs(cost: float, multiplier: float) -> float:
    """Multiply cost by a multiplier (for multi-day/multi-person)"""
    return cost * multiplier

@tool
def calculate_total_expense(*costs: float) -> float:
    """Calculate total expense from multiple costs"""
    return sum(costs)

@tool
def calculate_daily_budget(total_cost: float, days: int) -> float:
    """Calculate daily budget breakdown"""
    return total_cost / days if days > 0 else 0

# === STEP 5: CURRENCY CONVERSION ===

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get current exchange rate between currencies"""
    # Simplified - in real implementation, use a currency API
    exchange_rates = {
        "USD_INR": 83.50,
        "EUR_INR": 91.20,
        "GBP_INR": 106.30,
        "INR_USD": 0.012,
        "INR_EUR": 0.011,
        "INR_GBP": 0.0094
    }
    key = f"{from_currency}_{to_currency}"
    return exchange_rates.get(key, 1.0)

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert amount from one currency to another"""
    rate = get_exchange_rate(from_currency, to_currency)
    return amount * rate

# === STEP 6: ITINERARY GENERATION ===

@tool
def create_day_plan(city: str, day_number: int, attractions: str, weather: str) -> str:
    """Create a detailed day plan for the trip"""
    return f"""Day {day_number} in {city}:
Weather: {weather}
Recommended activities: {attractions[:200]}...
Tips: Plan indoor activities if weather is poor."""

@tool
def create_full_itinerary(city: str, total_days: int, attractions: List[str], 
                         weather_info: str, activities: List[str]) -> str:
    """Create complete itinerary for the entire trip"""
    itinerary = f"Complete {total_days}-Day Itinerary for {city}\n"
    itinerary += f"Weather Overview: {weather_info}\n\n"
    
    for day in range(1, total_days + 1):
        itinerary += f"Day {day}:\n"
        if day <= len(attractions):
            itinerary += f"- Main Attraction: {attractions[day-1]}\n"
        if day <= len(activities):
            itinerary += f"- Activity: {activities[day-1]}\n"
        itinerary += "\n"
    
    return itinerary

# === STEP 7: TRIP SUMMARY ===

@tool
def create_trip_summary(destination: str, duration: int, total_cost: float, 
                       daily_budget: float, highlights: List[str]) -> str:
    """Create comprehensive trip summary"""
    summary = f"""
TRIP SUMMARY
============
Destination: {destination}
Duration: {duration} days
Total Cost: ₹{total_cost:,.2f}
Daily Budget: ₹{daily_budget:,.2f}

Highlights:
"""
    for highlight in highlights[:5]:
        summary += f"- {highlight}\n"
    
    return summary

# Compile all tools
tools = [
    # Step 1: Attractions and Activities
    _get_attraction, search_restaurants, search_activities, _get_transport_pricing,
    
    # Step 2: Weather
    get_weather_forecast,
    
    # Step 3: Hotels
    hotel_search, estimate_hotel_cost, get_budget_range,
    
    # Step 4: Cost Calculations
    add_costs, multiply_costs, calculate_total_expense, calculate_daily_budget,
    
    # Step 5: Currency
    get_exchange_rate, convert_currency,
    
    # Step 6: Itinerary
    create_day_plan, create_full_itinerary,
    
    # Step 7: Summary
    create_trip_summary
]

# Enhanced System Prompt with Step-by-Step Workflow
TRAVEL_AGENT_SYSTEM_PROMPT = """
You are an expert AI Travel Agent following a structured 7-step workflow for comprehensive trip planning.

MANDATORY WORKFLOW - Follow these steps in order:

STEP 1: SEARCH ATTRACTIONS AND ACTIVITIES
- Use _get_attraction to find top attractions
- Use search_restaurants to find dining options  
- Use search_activities to find experiences
- Use _get_transport_pricing for transport costs

STEP 2: WEATHER FORECASTING
- Use get_weather_forecast for current weather and forecast

STEP 3: HOTEL SEARCH AND COSTS
- Use hotel_search to find accommodations
- Use estimate_hotel_cost to calculate hotel expenses
- Use get_budget_range to allocate budget categories

STEP 4: CALCULATE TOTAL COSTS (MANDATORY - USE ALL TOOLS)
- Use add_costs to combine expense categories
- Use multiply_costs for multi-day/multi-person calculations  
- Use calculate_total_expense for final totals
- Use calculate_daily_budget for daily breakdowns

STEP 5: CURRENCY CONVERSION
- Use get_exchange_rate if international travel
- Use convert_currency for currency conversions

STEP 6: ITINERARY GENERATION  
- Use create_day_plan for daily plans
- Use create_full_itinerary for complete schedule

STEP 7: CREATE TRIP SUMMARY
- Use create_trip_summary for final comprehensive summary

CRITICAL COST CALCULATION RULES:
1. NEVER manually calculate costs - always use the cost calculation tools
2. You MUST use all 4 cost calculation tools: add_costs, multiply_costs, calculate_total_expense, calculate_daily_budget
3. Show step-by-step tool usage for transparency
4. Extract numeric values from pricing data before using cost tools

WORKFLOW EXECUTION:
- Complete each step before moving to the next
- Use multiple tools per step when needed
- Show progress through each step clearly
- Always end with a comprehensive trip summary

Remember: The cost calculation tools are MANDATORY - never skip Step 4!
"""

# Initialize LLM and create workflow
llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def travel_planner_agent(state: MessagesState):
    """Enhanced travel planning agent with step-by-step workflow"""
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        system_message = SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT)
        messages = [system_message] + messages
    
    # Count completed steps based on tool usage
    used_tools = []
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            used_tools.extend([tc['name'] for tc in msg.tool_calls])
    
    # Determine current step and provide specific guidance
    step_guidance = get_step_guidance(used_tools)
    
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    latest_request = user_messages[-1].content if user_messages else ""
    
    planning_prompt = f"""
    Travel Request: {latest_request}
    
    {step_guidance}
    
    Follow the 7-step workflow systematically. Use the appropriate tools for the current step.
    """
    
    response = llm_with_tools.invoke(messages + [HumanMessage(content=planning_prompt)])
    return {"messages": [response]}

def get_step_guidance(used_tools: List[str]) -> str:
    """Provide step-specific guidance based on tools used"""
    
    step1_tools = ['get_attraction', 'search_restaurants', 'search_activities', 'get_transport_pricing']
    step2_tools = ['get_weather_forecast']
    step3_tools = ['hotel-search', 'estimate_hotel_cost', 'get_budget_range']
    step4_tools = ['add_costs', 'multiply_costs', 'calculate_total_expense', 'calculate_daily_budget']
    step5_tools = ['get_exchange_rate', 'convert_currency']
    step6_tools = ['create_day_plan', 'create_full_itinerary']
    step7_tools = ['create_trip_summary']
    
    if not any(tool in used_tools for tool in step1_tools):
        return "STEP 1: Start by searching for attractions, restaurants, activities, and transport pricing."
    
    elif not any(tool in used_tools for tool in step2_tools):
        return "STEP 2: Get weather forecast for travel planning."
    
    elif not any(tool in used_tools for tool in step3_tools):
        return "STEP 3: Search for hotels and calculate accommodation costs."
    
    elif not all(tool in used_tools for tool in step4_tools):
        missing_tools = [tool for tool in step4_tools if tool not in used_tools]
        return f"STEP 4: MANDATORY COST CALCULATIONS - You must use these remaining tools: {missing_tools}. Extract numeric values from pricing data and use these tools systematically."
    
    elif not any(tool in used_tools for tool in step5_tools):
        return "STEP 5: Handle currency conversion if needed."
    
    elif not any(tool in used_tools for tool in step6_tools):
        return "STEP 6: Create detailed day plans and full itinerary."
    
    elif not any(tool in used_tools for tool in step7_tools):
        return "STEP 7: Create comprehensive trip summary with all details."
    
    else:
        return "All steps completed. Provide final comprehensive travel plan."

def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """Determine whether to continue with tool calls or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Check if all mandatory steps are completed
    used_tools = []
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            used_tools.extend([tc['name'] for tc in msg.tool_calls])
    
    mandatory_cost_tools = ['add_costs', 'multiply_costs', 'calculate_total_expense', 'calculate_daily_budget']
    cost_tools_used = all(tool in used_tools for tool in mandatory_cost_tools)
    summary_created = 'create_trip_summary' in used_tools
    
    if cost_tools_used and summary_created:
        return "end"
    
    return "end"

# Create workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", travel_planner_agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

travel_agent = workflow.compile()

def run_travel_agent(user_request: str):
    """Run the enhanced travel planning agent"""
    initial_state = {"messages": [HumanMessage(content=user_request)]}
    
    for event in travel_agent.stream(initial_state):
        for value in event.values():
            if "messages" in value:
                last_message = value["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    print("Agent:", last_message.content[:200] + "..." if len(last_message.content) > 200 else last_message.content)
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"Using tools: {[tc['name'] for tc in last_message.tool_calls]}")

# Example usage
if __name__ == "__main__":
    user_input = """
    I want to plan a 5-day trip to Delhi for 2 people in August 2025. 
    Budget is around 5 lac. Looking for cultural attractions, good food, 
    and efficient local transport options.
    """
    
    run_travel_agent(user_input)