import pytest
import os
import asyncio
from typing import List, Dict, Optional, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime
import json

from langchain_decorators import (
    llm_prompt,
    llm_function,
    GlobalSettings,
    OutputWithFunctionCall,
    StreamingContext,
    FollowupHandle,
)


@pytest.fixture(scope="session")
def setup_real_llm():
    """Setup real LLM for testing"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping real LLM tests")

    real_llm = BaseChatModel(
        temperature=0.0,  # Deterministic for testing
        model_name="gpt-3.5-turbo"
    )

    GlobalSettings.define_settings(default_llm=real_llm, verbose=False)
    return real_llm


@pytest.mark.real_llm
class TestIntegrationScenarios:
    """Test real-world integration scenarios with actual LLM calls"""

    def test_customer_support_bot_integration(self):
        """Test a complete customer support bot scenario"""

        # Database simulation
        orders_db = {
            "ORD123": {
                "status": "shipped",
                "tracking": "1Z123456789",
                "items": ["Wireless Headphones"],
            },
            "ORD456": {
                "status": "processing",
                "tracking": None,
                "items": ["Laptop", "Mouse"],
            },
            "ORD789": {
                "status": "delivered",
                "tracking": "1Z987654321",
                "items": ["Phone Case"],
            },
        }

        @llm_function
        def lookup_order(order_id: str) -> str:
            """Look up order information by order ID

            Args:
                order_id (str): Order ID to look up
            """
            order = orders_db.get(order_id.upper())
            if not order:
                return f"Order {order_id} not found"

            result = f"Order {order_id}: Status - {order['status']}, Items - {', '.join(order['items'])}"
            if order["tracking"]:
                result += f", Tracking - {order['tracking']}"
            return result

        @llm_function
        def update_order_status(order_id: str, new_status: str) -> str:
            """Update order status

            Args:
                order_id (str): Order ID to update
                new_status (str): New status for the order
            """
            if order_id.upper() not in orders_db:
                return f"Order {order_id} not found"

            orders_db[order_id.upper()]["status"] = new_status
            return f"Order {order_id} status updated to {new_status}"

        @llm_function
        def create_return_request(order_id: str, reason: str) -> str:
            """Create a return request

            Args:
                order_id (str): Order ID for return
                reason (str): Reason for return
            """
            return f"Return request RET{hash(order_id) % 1000:03d} created for order {order_id}. Reason: {reason}"

        @llm_prompt
        def customer_support_agent(
            customer_request: str, customer_name: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are a helpful customer support agent. Be polite, professional, and helpful.
            Always address the customer by name when possible.
            Use available functions to help customers with their requests.
            ```

            ```<prompt:user>
            Customer: {customer_name}
            Request: {customer_request}
            ```

            Available functions: {functions}
            """
            pass

        # Test order lookup
        result1 = customer_support_agent(
            customer_request="Hi, I need to check the status of my order ORD123",
            customer_name="John Smith",
            functions=[lookup_order, update_order_status, create_return_request],
        )

        if result1.is_function_call:
            order_info = result1.execute()
            assert "ORD123" in order_info
            assert "shipped" in order_info.lower()

        # Test return request
        result2 = customer_support_agent(
            customer_request="I want to return order ORD456 because it arrived damaged",
            customer_name="Jane Doe",
            functions=[lookup_order, update_order_status, create_return_request],
        )

        if result2.is_function_call:
            return_info = result2.execute()
            assert "return" in return_info.lower()
            assert "ORD456" in return_info

    def test_multi_agent_project_management(self):
        """Test multi-agent project management system"""

        # Project state
        project_state = {
            "tasks": {},
            "team_members": ["Alice", "Bob", "Charlie"],
            "task_counter": 0,
        }

        class TaskStatus(Enum):
            TODO = "todo"
            IN_PROGRESS = "in_progress"
            DONE = "done"

        class Task(BaseModel):
            id: str
            title: str
            assignee: Optional[str] = None
            status: TaskStatus = TaskStatus.TODO
            priority: str = "medium"

        @llm_function
        def create_task(
            title: str, assignee: Optional[str] = None, priority: str = "medium"
        ) -> str:
            """Create a new task

            Args:
                title (str): Task title
                assignee (str, optional): Team member to assign task to
                priority (str): Task priority (low, medium, high)
            """
            project_state["task_counter"] += 1
            task_id = f"TASK-{project_state['task_counter']:03d}"

            task = Task(id=task_id, title=title, assignee=assignee, priority=priority)

            project_state["tasks"][task_id] = task
            return f"Created {task_id}: '{title}' (Priority: {priority}, Assigned: {assignee or 'Unassigned'})"

        @llm_function
        def update_task_status(task_id: str, status: str) -> str:
            """Update task status

            Args:
                task_id (str): Task ID to update
                status (str): New status (todo, in_progress, done)
            """
            if task_id not in project_state["tasks"]:
                return f"Task {task_id} not found"

            try:
                new_status = TaskStatus(status.lower())
                project_state["tasks"][task_id].status = new_status
                return f"Updated {task_id} status to {status}"
            except ValueError:
                return (
                    f"Invalid status '{status}'. Valid options: todo, in_progress, done"
                )

        @llm_function
        def get_project_overview() -> str:
            """Get project overview with all tasks"""
            if not project_state["tasks"]:
                return "No tasks in project"

            overview = "Project Overview:\n"
            for task_id, task in project_state["tasks"].items():
                overview += f"- {task_id}: {task.title} [{task.status.value}] (Priority: {task.priority})\n"
                if task.assignee:
                    overview += f"  Assigned to: {task.assignee}\n"
            return overview

        @llm_prompt
        def project_manager(
            request: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are a project manager AI. Help manage tasks and coordinate team work.
            Available team members: Alice, Bob, Charlie

            Use appropriate functions based on the request:
            - Create tasks when users want to add work items
            - Update status when tasks progress
            - Show overview when asked about project status
            ```

            ```<prompt:user>
            {request}
            ```

            Available functions: {functions}
            """
            pass

        # Create initial tasks
        result1 = project_manager(
            request="Create a high priority task to implement user authentication and assign it to Alice",
            functions=[create_task, update_task_status, get_project_overview],
        )

        if result1.is_function_call:
            task_result = result1.execute()
            assert "TASK-" in task_result
            assert "authentication" in task_result.lower()

        # Update task status
        result2 = project_manager(
            request="Update TASK-001 status to in progress",
            functions=[create_task, update_task_status, get_project_overview],
        )

        if result2.is_function_call:
            update_result = result2.execute()
            assert "TASK-001" in update_result
            assert "in_progress" in update_result or "in progress" in update_result

        # Get overview
        result3 = project_manager(
            request="Show me the current project status",
            functions=[create_task, update_task_status, get_project_overview],
        )

        if result3.is_function_call:
            overview_result = result3.execute()
            assert "Project Overview" in overview_result
            assert "TASK-001" in overview_result

    @pytest.mark.asyncio
    async def test_async_streaming_chat_agent(self):
        """Test async streaming chat agent with real-time responses"""

        chat_history = []

        @llm_function
        async def remember_user_info(name: str, detail: str) -> str:
            """Remember information about the user

            Args:
                name (str): User's name
                detail (str): Information to remember
            """
            await asyncio.sleep(0.1)  # Simulate async operation
            timestamp = datetime.now().strftime("%H:%M")
            return f"Remembered about {name}: {detail} (saved at {timestamp})"

        @llm_function
        async def get_time() -> str:
            """Get current time"""
            await asyncio.sleep(0.05)
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        @llm_prompt(capture_stream=True)
        async def streaming_chat_agent(
            user_message: str, user_name: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are a friendly chat assistant. Be conversational and helpful.
            Remember information about users when they share it.
            Keep responses concise and engaging.
            ```

            ```<prompt:user>
            {user_name}: {user_message}
            ```

            Available functions: {functions}
            """
            pass

        collected_tokens = []

        def token_collector(token: str):
            collected_tokens.append(token)

        # Test streaming response
        with StreamingContext(stream_to_stdout=False, callback=token_collector):
            result = await streaming_chat_agent(
                user_message="Hi, I'm Sarah and I love hiking in the mountains",
                user_name="Sarah",
                functions=[remember_user_info, get_time],
                function_call="none"
            )

        # Verify streaming worked
        assert len(collected_tokens) > 0

        if result.is_function_call:
            function_result = await result.execute_async()
            assert "Sarah" in function_result
            assert "hiking" in function_result.lower()

        # Test time function
        result2 = await streaming_chat_agent(
            user_message="What time is it?",
            user_name="Sarah", 
            functions=[remember_user_info, get_time]
        )

        if result2.is_function_call and result2.function_name == "get_time":
            time_result = await result2.execute_async()
            assert "2024" in time_result or "2025" in time_result  # Current year

    def test_e_commerce_recommendation_system(self):
        """Test e-commerce recommendation system with product catalog"""

        # Product catalog
        products = {
            "laptop-001": {"name": "Gaming Laptop", "price": 1200, "category": "electronics", "rating": 4.5},
            "book-001": {"name": "Python Programming", "price": 40, "category": "books", "rating": 4.8},
            "headphones-001": {"name": "Wireless Headphones", "price": 150, "category": "electronics", "rating": 4.3},
            "coffee-001": {"name": "Premium Coffee Beans", "price": 25, "category": "food", "rating": 4.7}
        }

        user_preferences = {}

        @llm_function
        def search_products(query: str, category: Optional[str] = None, max_price: Optional[float] = None) -> str:
            """Search for products

            Args:
                query (str): Search query
                category (str, optional): Product category filter
                max_price (float, optional): Maximum price filter
            """
            results = []
            query_lower = query.lower()

            for product_id, product in products.items():
                # Check if product matches query
                if (
                    query_lower in product["name"].lower()
                    or query_lower in product["category"].lower()
                ):

                    # Apply filters
                    if category and product["category"] != category.lower():
                        continue
                    if max_price and product["price"] > max_price:
                        continue

                    results.append(f"{product['name']} - ${product['price']} (Rating: {product['rating']}/5)")

            if not results:
                return f"No products found for '{query}'"

            return "Found products:\n" + "\n".join(results)

        @llm_function
        def save_user_preference(preference_type: str, preference_value: str) -> str:
            """Save user preference

            Args:
                preference_type (str): Type of preference (category, budget, brand, etc.)
                preference_value (str): Preference value
            """
            user_preferences[preference_type] = preference_value
            return f"Saved preference: {preference_type} = {preference_value}"

        @llm_function
        def get_recommendations(user_context: str) -> str:
            """Get personalized recommendations

            Args:
                user_context (str): Context about what user is looking for
            """
            # Simple recommendation logic based on context
            context_lower = user_context.lower()
            recommendations = []

            if "programming" in context_lower or "coding" in context_lower:
                recommendations.append("Python Programming - $40 (Perfect for learning!)")
                recommendations.append("Gaming Laptop - $1200 (Great for development)")
            elif "music" in context_lower or "audio" in context_lower:
                recommendations.append(
                    "Wireless Headphones - $150 (Excellent sound quality)"
                )
            elif "work" in context_lower or "productivity" in context_lower:
                recommendations.append("Gaming Laptop - $1200 (High performance)")
                recommendations.append("Premium Coffee Beans - $25 (Fuel your work!)")
            else:
                # Default recommendations
                recommendations = [
                    "Gaming Laptop - $1200",
                    "Wireless Headphones - $150",
                ]

            return "Personalized recommendations:\n" + "\n".join(recommendations)

        @llm_prompt
        def shopping_assistant(
            customer_request: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are a helpful shopping assistant. Help customers find products they'll love.
            Use search to find specific products, save preferences for personalization,
            and provide recommendations based on customer context.

            Be friendly and helpful in your responses.
            ```

            ```<prompt:user>
            Customer request: {customer_request}
            ```

            Available functions: {functions}
            """
            pass

        # Test product search
        result1 = shopping_assistant(
            customer_request="I'm looking for a laptop under $1500",
            functions=[search_products, save_user_preference, get_recommendations]
        )

        if result1.is_function_call:
            search_result = result1.execute()
            assert "laptop" in search_result.lower()

        # Test recommendation request
        result2 = shopping_assistant(
            customer_request="I'm a software developer who loves music, what would you recommend?",
            functions=[search_products, save_user_preference, get_recommendations]
        )

        if result2.is_function_call:
            rec_result = result2.execute()
            assert "recommendations" in rec_result.lower()

    def test_educational_tutor_system(self):
        """Test educational tutor with progress tracking"""

        student_progress = {
            "topics_covered": [],
            "quiz_scores": {},
            "current_level": "beginner"
        }

        quiz_questions = {
            "python_basics": [
                {"question": "What keyword is used to define a function in Python?", "answer": "def"},
                {"question": "How do you create a list in Python?", "answer": "[]"},
            ],
            "data_types": [
                {"question": "What data type represents whole numbers?", "answer": "int"},
                {"question": "What data type represents text?", "answer": "str"},
            ]
        }

        @llm_function
        def create_quiz(topic: str, difficulty: str = "beginner") -> str:
            """Create a quiz for a specific topic
            
            Args:
                topic (str): Topic for the quiz
                difficulty (str): Difficulty level (beginner, intermediate, advanced)
            """
            if topic.lower().replace(" ", "_") in quiz_questions:
                questions = quiz_questions[topic.lower().replace(" ", "_")]
                quiz_text = f"Quiz: {topic.title()} ({difficulty})\n"
                for i, q in enumerate(questions, 1):
                    quiz_text += f"{i}. {q['question']}\n"
                return quiz_text
            else:
                return f"No quiz available for topic '{topic}'"

        @llm_function
        def record_progress(topic: str, score: float) -> str:
            """Record student progress
            
            Args:
                topic (str): Topic studied
                score (float): Quiz score (0-100)
            """
            student_progress["topics_covered"].append(topic)
            student_progress["quiz_scores"][topic] = score

            # Update level based on average score
            avg_score = sum(student_progress["quiz_scores"].values()) / len(student_progress["quiz_scores"])
            if avg_score >= 80:
                student_progress["current_level"] = "advanced"
            elif avg_score >= 60:
                student_progress["current_level"] = "intermediate"

            return f"Progress recorded for {topic}: {score}%. Current level: {student_progress['current_level']}"

        @llm_function
        def get_learning_path(current_topic: str) -> str:
            """Get suggested learning path
            
            Args:
                current_topic (str): Current topic being studied
            """
            learning_paths = {
                "python_basics": ["data_types", "control_structures", "functions"],
                "data_types": ["variables", "strings", "lists"],
                "control_structures": ["loops", "conditionals", "error_handling"]
            }

            next_topics = learning_paths.get(current_topic.lower().replace(" ", "_"), ["advanced_topics"])
            return f"After {current_topic}, consider studying: {', '.join(next_topics)}"

        @llm_prompt
        def educational_tutor(
            student_request: str,
            functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are an educational tutor specializing in programming.
            Help students learn by providing quizzes, tracking progress, and suggesting learning paths.
            
            Be encouraging and adapt your teaching style to the student's level.
            Use available functions to create interactive learning experiences.
            ```
            
            ```<prompt:user>
            Student request: {student_request}
            ```
            
            Available functions: {functions}
            """
            pass

        # Test quiz creation
        result1 = educational_tutor(
            student_request="I want to practice Python basics with a quiz",
            functions=[create_quiz, record_progress, get_learning_path]
        )

        if result1.is_function_call and result1.function_name == "create_quiz":
            quiz_result = result1.execute()
            assert "Quiz:" in quiz_result
            assert "Python" in quiz_result or "python" in quiz_result

        # Test progress recording
        result2 = educational_tutor(
            student_request="I completed the Python basics quiz and got 85%",
            functions=[create_quiz, record_progress, get_learning_path]
        )

        if result2.is_function_call and result2.function_name == "record_progress":
            progress_result = result2.execute()
            assert "85" in progress_result
            assert "Progress recorded" in progress_result

        # Test learning path suggestion
        result3 = educational_tutor(
            student_request="What should I study next after Python basics?",
            functions=[create_quiz, record_progress, get_learning_path]
        )

        if result3.is_function_call and result3.function_name == "get_learning_path":
            path_result = result3.execute()
            assert "After" in path_result
            assert "consider studying" in path_result

if __name__ == "__main__":
    asyncio.run(TestIntegrationScenarios().test_async_streaming_chat_agent())
