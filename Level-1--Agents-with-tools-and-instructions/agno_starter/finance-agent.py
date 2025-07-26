from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

agent = Agent(  
    name="Finance Analyst",
    model=DeepSeek(
        id="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY")),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Use tables to display data. Don't include any other text.",
    markdown=True
)

#agent.print_response("What is the stock price of Apple?", stream=True)


while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("bye! 👋")
            break
            
        # Add timestamp to the response
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
        agent.print_response(user_input, stream=True)

if __name__ == "__main__":
    main()