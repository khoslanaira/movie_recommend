#!/usr/bin/env python3
"""
Conversational AI Agent for Movie Recommendations
Handles natural language interactions and provides intelligent responses
"""

import re
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

class ConversationalAgent:
    def __init__(self, recommendation_models, data_collector, movies_df):
        self.models = recommendation_models
        self.data_collector = data_collector
        self.movies_df = movies_df
        self.conversation_history = {}
        self.user_preferences = {}
        
        # Intent patterns
        self.intent_patterns = {
            'recommend': [
                r'recommend', r'suggest', r'what should i watch', r'what movie',
                r'give me', r'find me', r'help me find', r'what do you suggest'
            ],
            'explain': [
                r'why', r'explain', r'how did you', r'what made you',
                r'tell me about', r'why did you recommend'
            ],
            'rate': [
                r'rate', r'rating', r'i rate', r'i give', r'stars',
                r'i think this is', r'this movie is'
            ],
            'search': [
                r'search', r'find', r'look for', r'looking for',
                r'genre', r'type of movie', r'kind of movie'
            ],
            'similar': [
                r'similar', r'like', r'same as', r'comparable',
                r'movies like', r'similar to'
            ],
            'feedback': [
                r'feedback', r'comment', r'thoughts', r'opinion',
                r'i think', r'i feel', r'my opinion'
            ],
            'greeting': [
                r'hello', r'hi', r'hey', r'good morning', r'good afternoon',
                r'good evening', r'greetings'
            ],
            'help': [
                r'help', r'what can you do', r'how do you work',
                r'what are you', r'capabilities'
            ]
        }
        
        # Response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm your movie recommendation assistant. How can I help you today?",
                "Hi there! I can help you find great movies to watch. What are you in the mood for?",
                "Hey! Ready to discover some amazing movies? Just tell me what you're looking for!"
            ],
            'help': [
                "I can help you with movie recommendations! Here's what I can do:\n"
                "• Recommend movies based on your preferences\n"
                "• Explain why I recommended specific movies\n"
                "• Find movies similar to ones you like\n"
                "• Help you rate movies and learn from your feedback\n"
                "• Search for movies by genre or keywords\n\n"
                "Just tell me what you'd like to do!",
                
                "I'm your AI movie assistant! I can:\n"
                "🎬 Recommend personalized movies\n"
                "💡 Explain my recommendations\n"
                "🔍 Find similar movies\n"
                "⭐ Help you rate movies\n"
                "🎭 Search by genre or keywords\n\n"
                "What would you like to do?"
            ],
            'no_recommendations': [
                "I'd be happy to recommend movies for you! Could you tell me a bit about what you like?",
                "I need to know more about your preferences to give good recommendations. What genres do you enjoy?",
                "Let me help you find great movies! What type of movies do you usually watch?"
            ],
            'rating_collected': [
                "Thanks for the rating! I'll use this to improve my recommendations for you.",
                "Great! I've noted your rating and will learn from your preferences.",
                "Perfect! Your feedback helps me understand your taste better."
            ]
        }
    
    def process_message(self, user_id: int, message: str) -> str:
        """Process user message and generate response"""
        # Store conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'user_message': message,
            'timestamp': datetime.now(),
            'agent_response': None
        })
        
        # Detect intent
        intent = self.detect_intent(message)
        
        # Generate response based on intent
        response = self.generate_response(user_id, message, intent)
        
        # Store agent response
        self.conversation_history[user_id][-1]['agent_response'] = response
        
        return response
    
    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return 'unknown'
    
    def generate_response(self, user_id: int, message: str, intent: str) -> str:
        """Generate response based on detected intent"""
        
        if intent == 'greeting':
            return random.choice(self.response_templates['greeting'])
        
        elif intent == 'help':
            return random.choice(self.response_templates['help'])
        
        elif intent == 'recommend':
            return self.handle_recommendation_request(user_id, message)
        
        elif intent == 'explain':
            return self.handle_explanation_request(user_id, message)
        
        elif intent == 'rate':
            return self.handle_rating_request(user_id, message)
        
        elif intent == 'search':
            return self.handle_search_request(user_id, message)
        
        elif intent == 'similar':
            return self.handle_similar_request(user_id, message)
        
        elif intent == 'feedback':
            return self.handle_feedback_request(user_id, message)
        
        else:
            return self.handle_unknown_intent(user_id, message)
    
    def handle_recommendation_request(self, user_id: int, message: str) -> str:
        """Handle movie recommendation requests"""
        try:
            # Get recommendations from models
            recommendations = self.get_recommendations(user_id)
            
            if not recommendations:
                return random.choice(self.response_templates['no_recommendations'])
            
            # Format recommendations
            response = "Here are some movies I think you'll enjoy:\n\n"
            
            for i, (movie_id, title, year) in enumerate(recommendations[:5], 1):
                response += f"{i}. **{title}** ({year})\n"
            
            response += "\nWould you like me to explain why I recommended any of these?"
            
            return response
            
        except Exception as e:
            return f"I'm having trouble getting recommendations right now. Error: {str(e)}"
    
    def handle_explanation_request(self, user_id: int, message: str) -> str:
        """Handle explanation requests"""
        # Extract movie title from message
        movie_title = self.extract_movie_title(message)
        
        if not movie_title:
            return "I'd be happy to explain a recommendation! Which movie would you like me to explain?"
        
        # Find movie in database
        movie = self.find_movie_by_title(movie_title)
        
        if movie is None:
            return f"I couldn't find a movie called '{movie_title}'. Could you check the spelling?"
        
        # Generate explanation
        explanation = self.generate_explanation(user_id, movie['item_id'])
        
        return explanation
    
    def handle_rating_request(self, user_id: int, message: str) -> str:
        """Handle movie rating requests"""
        # Extract movie title and rating
        movie_title, rating = self.extract_movie_and_rating(message)
        
        if not movie_title or not rating:
            return "I'd love to record your rating! Please tell me the movie name and your rating (1-5 stars)."
        
        # Find movie
        movie = self.find_movie_by_title(movie_title)
        
        if movie is None:
            return f"I couldn't find a movie called '{movie_title}'. Could you check the spelling?"
        
        # Collect rating
        self.data_collector.collect_rating(user_id, movie['item_id'], rating)
        
        # Update user preferences
        self.update_user_preferences(user_id, movie['item_id'], rating)
        
        return random.choice(self.response_templates['rating_collected'])
    
    def handle_search_request(self, user_id: int, message: str) -> str:
        """Handle movie search requests"""
        # Extract search terms
        search_terms = self.extract_search_terms(message)
        
        if not search_terms:
            return "What would you like to search for? You can search by genre, keywords, or movie titles."
        
        # Search movies
        results = self.search_movies(search_terms)
        
        if not results:
            return f"I couldn't find any movies matching '{' '.join(search_terms)}'. Try different keywords?"
        
        # Format results
        response = f"Here are movies matching '{' '.join(search_terms)}':\n\n"
        
        for i, (movie_id, title, year, genres) in enumerate(results[:10], 1):
            response += f"{i}. **{title}** ({year}) - {genres}\n"
        
        return response
    
    def handle_similar_request(self, user_id: int, message: str) -> str:
        """Handle similar movie requests"""
        movie_title = self.extract_movie_title(message)
        
        if not movie_title:
            return "I'd be happy to find similar movies! Which movie would you like me to find similar ones for?"
        
        # Find movie
        movie = self.find_movie_by_title(movie_title)
        
        if movie is None:
            return f"I couldn't find a movie called '{movie_title}'. Could you check the spelling?"
        
        # Find similar movies
        similar_movies = self.find_similar_movies(movie['item_id'])
        
        if not similar_movies:
            return f"I couldn't find similar movies to '{movie_title}'. Try a different movie?"
        
        # Format results
        response = f"Here are movies similar to **{movie_title}**:\n\n"
        
        for i, (movie_id, title, year) in enumerate(similar_movies[:5], 1):
            response += f"{i}. **{title}** ({year})\n"
        
        return response
    
    def handle_feedback_request(self, user_id: int, message: str) -> str:
        """Handle feedback requests"""
        # Extract movie and feedback
        movie_title, feedback = self.extract_movie_and_feedback(message)
        
        if movie_title and feedback:
            movie = self.find_movie_by_title(movie_title)
            if movie:
                # Collect feedback
                self.data_collector.collect_feedback(user_id, movie['item_id'], feedback)
                return "Thanks for your feedback! I'll use this to improve my recommendations."
        
        return "I'd love to hear your feedback! Tell me about a movie you watched and what you thought of it."
    
    def handle_unknown_intent(self, user_id: int, message: str) -> str:
        """Handle unknown intents"""
        return ("I'm not sure I understand. I can help you with:\n"
                "• Movie recommendations\n"
                "• Explaining why I recommended something\n"
                "• Finding similar movies\n"
                "• Recording your ratings\n"
                "• Searching for movies\n\n"
                "What would you like to do?")
    
    def get_recommendations(self, user_id: int, n: int = 5) -> List[Tuple[int, str, str]]:
        """Get movie recommendations for user"""
        try:
            # Get user's rated movies to avoid recommending them again
            user_ratings = self.data_collector.get_user_ratings(user_id)
            rated_movies = set(user_ratings['movie_id'].tolist()) if not user_ratings.empty else set()
            
            # Get recommendations from models
            recommendations = []
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'get_recommendations'):
                        recs = model.get_recommendations(user_id, n=n*2)
                        recommendations.extend(recs)
                    elif hasattr(model, 'predict_rating'):
                        # Generate recommendations by predicting ratings for all movies
                        all_movies = self.movies_df['item_id'].unique()
                        pred_ratings = []
                        
                        for movie_id in all_movies:
                            if movie_id not in rated_movies:
                                try:
                                    pred_rating = model.predict_rating(user_id, movie_id)
                                    pred_ratings.append((movie_id, pred_rating))
                                except:
                                    continue
                        
                        # Sort by predicted rating
                        pred_ratings.sort(key=lambda x: x[1], reverse=True)
                        recommendations.extend([movie_id for movie_id, _ in pred_ratings[:n*2]])
                
                except Exception as e:
                    print(f"Error getting recommendations from {model_name}: {e}")
                    continue
            
            # Remove duplicates and get movie details
            unique_recs = list(dict.fromkeys(recommendations))[:n]
            
            result = []
            for movie_id in unique_recs:
                movie_info = self.movies_df[self.movies_df['item_id'] == movie_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]['title']
                    year = movie_info.iloc[0].get('release_date', 'Unknown')
                    result.append((movie_id, title, year))
            
            return result
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def extract_movie_title(self, message: str) -> Optional[str]:
        """Extract movie title from message"""
        # Simple extraction - look for quoted text or common patterns
        quoted = re.findall(r'"([^"]*)"', message)
        if quoted:
            return quoted[0]
        
        # Look for "movie" followed by title
        movie_match = re.search(r'movie\s+([^.!?]+)', message, re.IGNORECASE)
        if movie_match:
            return movie_match.group(1).strip()
        
        return None
    
    def extract_movie_and_rating(self, message: str) -> Tuple[Optional[str], Optional[float]]:
        """Extract movie title and rating from message"""
        # Look for rating patterns
        rating_patterns = [
            r'(\d+(?:\.\d+)?)\s*stars?',
            r'rate\s+([^.!?]+?)\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*out\s+of\s+5',
            r'(\d+(?:\.\d+)?)\s*\/\s*5'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                rating = float(match.group(1) if len(match.groups()) == 1 else match.group(2))
                movie_title = match.group(1) if len(match.groups()) == 2 else None
                
                if not movie_title:
                    movie_title = self.extract_movie_title(message)
                
                return movie_title, rating
        
        return None, None
    
    def extract_search_terms(self, message: str) -> List[str]:
        """Extract search terms from message"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', message.lower())
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return terms
    
    def extract_movie_and_feedback(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract movie title and feedback from message"""
        movie_title = self.extract_movie_title(message)
        
        if movie_title:
            # Remove movie title from message to get feedback
            feedback = message.replace(f'"{movie_title}"', '').strip()
            if not feedback:
                feedback = message
        else:
            feedback = message
        
        return movie_title, feedback
    
    def find_movie_by_title(self, title: str) -> Optional[Dict]:
        """Find movie by title in database"""
        title_lower = title.lower()
        
        # Exact match
        exact_match = self.movies_df[self.movies_df['title'].str.lower() == title_lower]
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()
        
        # Partial match
        partial_match = self.movies_df[self.movies_df['title'].str.lower().str.contains(title_lower)]
        if not partial_match.empty:
            return partial_match.iloc[0].to_dict()
        
        return None
    
    def search_movies(self, terms: List[str]) -> List[Tuple[int, str, str, str]]:
        """Search movies by terms"""
        results = []
        
        for _, movie in self.movies_df.iterrows():
            title = movie['title'].lower()
            genres = str(movie.get('genres', '')).lower()
            
            # Check if any term matches
            if any(term in title or term in genres for term in terms):
                results.append((
                    movie['item_id'],
                    movie['title'],
                    movie.get('release_date', 'Unknown'),
                    movie.get('genres', 'Unknown')
                ))
        
        return results[:20]  # Limit results
    
    def find_similar_movies(self, movie_id: int) -> List[Tuple[int, str, str]]:
        """Find movies similar to given movie"""
        # This is a simplified version - you could implement more sophisticated similarity
        movie = self.movies_df[self.movies_df['item_id'] == movie_id]
        
        if movie.empty:
            return []
        
        movie_genres = str(movie.iloc[0].get('genres', '')).lower()
        
        similar = []
        for _, other_movie in self.movies_df.iterrows():
            if other_movie['item_id'] != movie_id:
                other_genres = str(other_movie.get('genres', '')).lower()
                
                # Simple genre overlap
                if any(genre in other_genres for genre in movie_genres.split()):
                    similar.append((
                        other_movie['item_id'],
                        other_movie['title'],
                        other_movie.get('release_date', 'Unknown')
                    ))
        
        return similar[:10]
    
    def generate_explanation(self, user_id: int, movie_id: int) -> str:
        """Generate explanation for recommendation"""
        # Get user's rating history
        user_ratings = self.data_collector.get_user_ratings(user_id)
        
        if user_ratings.empty:
            return "I recommended this movie based on its popularity and high ratings from other users."
        
        # Find similar users (simplified)
        similar_users = self.find_similar_users(user_id)
        
        if similar_users:
            return f"I recommended this movie because {len(similar_users)} users with similar tastes to you also rated it highly. You both seem to enjoy similar genres and styles."
        else:
            return "I recommended this movie based on its high ratings and popularity among users with similar preferences to yours."
    
    def find_similar_users(self, user_id: int) -> List[int]:
        """Find users with similar preferences (simplified)"""
        # This is a simplified version - you could implement more sophisticated user similarity
        user_ratings = self.data_collector.get_user_ratings(user_id)
        
        if user_ratings.empty:
            return []
        
        user_movies = set(user_ratings['movie_id'].tolist())
        similar_users = []
        
        # Get all other users
        all_ratings = self.data_collector.get_user_ratings()
        
        for other_user_id in all_ratings['user_id'].unique():
            if other_user_id != user_id:
                other_ratings = all_ratings[all_ratings['user_id'] == other_user_id]
                other_movies = set(other_ratings['movie_id'].tolist())
                
                # Calculate overlap
                overlap = len(user_movies.intersection(other_movies))
                if overlap > 0:
                    similar_users.append(other_user_id)
        
        return similar_users[:5]
    
    def update_user_preferences(self, user_id: int, movie_id: int, rating: float):
        """Update user preferences based on rating"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'liked_genres': set(), 'disliked_genres': set()}
        
        # Get movie genres
        movie = self.movies_df[self.movies_df['item_id'] == movie_id]
        if not movie.empty:
            genres = str(movie.iloc[0].get('genres', '')).split()
            
            if rating >= 4:
                self.user_preferences[user_id]['liked_genres'].update(genres)
            elif rating <= 2:
                self.user_preferences[user_id]['disliked_genres'].update(genres)

if __name__ == "__main__":
    # Test the conversational agent
    print("Testing Conversational Agent...")
    
    # This would be used with actual models and data collector
    # agent = ConversationalAgent(models, data_collector, movies_df)
    # response = agent.process_message(1, "Hello! Can you recommend some movies?")
    # print(f"Agent: {response}")
