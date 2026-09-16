"""
Generate synthetic tool-calling training data for Pebble SFT.

Creates ~50K conversations covering:
- Single tool calls (various tools)
- Multi-turn with tool calls
- Tool calls with varied argument counts
- Plain chat (no tools) for balance
- User asks that DON'T need tools (model should respond directly)

The goal is to teach the FORMAT, not specific tool knowledge.
Diverse tools + varied phrasings = generalization to unseen tools.

Usage (from project root):
    python scripts/generate_tool_data.py
"""

import os
import sys
import json
import random
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from tokenizer.bpe import BPETokenizer
from data.tool_format import format_conversation, format_tool_schema, format_plain_chat

random.seed(42)

# =============================================================================
# TOOL DEFINITIONS — diverse tools to maximize generalization
# =============================================================================

TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "return_type": "str",
        "parameters": [
            {"name": "city", "type": "str", "description": "The city name"}
        ],
    },
    "calculate": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "return_type": "str",
        "parameters": [
            {"name": "expression", "type": "str", "description": "Math expression to evaluate"}
        ],
    },
    "get_time": {
        "name": "get_time",
        "description": "Get the current time in a given timezone.",
        "return_type": "str",
        "parameters": [
            {"name": "timezone", "type": "str", "description": "Timezone name like UTC, EST, PST"}
        ],
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information on a topic.",
        "return_type": "str",
        "parameters": [
            {"name": "query", "type": "str", "description": "The search query"}
        ],
    },
    "translate": {
        "name": "translate",
        "description": "Translate text from one language to another.",
        "return_type": "str",
        "parameters": [
            {"name": "text", "type": "str", "description": "Text to translate"},
            {"name": "target_language", "type": "str", "description": "Language to translate into"}
        ],
    },
    "tell_story": {
        "name": "tell_story",
        "description": "Generate a short story about a given topic with a specified mood.",
        "return_type": "str",
        "parameters": [
            {"name": "topic", "type": "str", "description": "What the story is about"},
            {"name": "mood", "type": "str", "description": "The mood of the story (happy, sad, funny, scary)"}
        ],
    },
    "set_reminder": {
        "name": "set_reminder",
        "description": "Set a reminder for a specific time with a message.",
        "return_type": "str",
        "parameters": [
            {"name": "message", "type": "str", "description": "The reminder message"},
            {"name": "time", "type": "str", "description": "When to remind (e.g. '3pm', 'in 2 hours')"}
        ],
    },
    "get_definition": {
        "name": "get_definition",
        "description": "Look up the dictionary definition of a word.",
        "return_type": "str",
        "parameters": [
            {"name": "word", "type": "str", "description": "The word to define"}
        ],
    },
    "convert_units": {
        "name": "convert_units",
        "description": "Convert a value from one unit to another.",
        "return_type": "str",
        "parameters": [
            {"name": "value", "type": "float", "description": "The numeric value to convert"},
            {"name": "from_unit", "type": "str", "description": "The source unit"},
            {"name": "to_unit", "type": "str", "description": "The target unit"}
        ],
    },
    "send_message": {
        "name": "send_message",
        "description": "Send a message to a contact.",
        "return_type": "str",
        "parameters": [
            {"name": "recipient", "type": "str", "description": "Name of the person to message"},
            {"name": "message", "type": "str", "description": "The message content"}
        ],
    },
}

# =============================================================================
# TEMPLATE DATA — varied phrasings for each tool
# =============================================================================

WEATHER_CITIES = [
    "New York", "London", "Paris", "Tokyo", "Sydney", "Mumbai", "Berlin",
    "Toronto", "Dubai", "Rome", "Madrid", "Seoul", "Bangkok", "Cairo",
    "Moscow", "Beijing", "Lagos", "Lima", "Istanbul", "Singapore",
    "San Francisco", "Chicago", "Amsterdam", "Barcelona", "Vienna",
]

WEATHER_PHRASINGS = [
    "What's the weather in {city}?",
    "How's the weather in {city} today?",
    "Tell me the weather for {city}.",
    "What is the current weather in {city}?",
    "Is it raining in {city}?",
    "What's it like outside in {city}?",
    "Weather in {city} please.",
    "Can you check the weather in {city}?",
    "I need the weather forecast for {city}.",
    "How hot is it in {city}?",
]

WEATHER_RESULTS = [
    "Sunny, {temp}C, light breeze",
    "Cloudy with a chance of rain, {temp}C",
    "Clear skies, {temp}C, humidity {hum}%",
    "Partly cloudy, {temp}C, wind {wind} km/h",
    "Rainy, {temp}C, heavy showers expected",
    "Overcast, {temp}C, fog in the morning",
    "Warm and sunny, {temp}C, UV index high",
    "Cold and windy, {temp}C, wind chill {wc}C",
]

WEATHER_RESPONSES = [
    "The weather in {city} is currently {result}.",
    "In {city}, it's {result}.",
    "Right now in {city}: {result}.",
    "Here's the weather for {city}: {result}.",
    "The current conditions in {city} are {result}.",
]

CALC_EXPRESSIONS = [
    ("2 + 2", "4"), ("15 * 3", "45"), ("100 / 4", "25"), ("7 ** 2", "49"),
    ("sqrt(144)", "12"), ("3.14 * 5 * 5", "78.5"), ("1000 - 387", "613"),
    ("24 * 60", "1440"), ("2 ** 10", "1024"), ("99 + 101", "200"),
    ("50 * 0.15", "7.5"), ("365 * 24", "8760"), ("128 / 8", "16"),
    ("17 + 28 + 35", "80"), ("9 * 9 * 9", "729"), ("1.5 * 2.5", "3.75"),
    ("500 - 127", "373"), ("12 * 12", "144"), ("3 + 4 * 5", "23"),
    ("(10 + 5) * 3", "45"),
]

CALC_PHRASINGS = [
    "What is {expr}?",
    "Calculate {expr}.",
    "What's {expr}?",
    "Can you compute {expr}?",
    "How much is {expr}?",
    "Solve {expr}.",
    "What does {expr} equal?",
    "Figure out {expr} for me.",
    "I need to know {expr}.",
]

CALC_RESPONSES = [
    "{expr} equals {result}.",
    "The result of {expr} is {result}.",
    "That would be {result}.",
    "{expr} = {result}.",
    "The answer is {result}.",
]

TIME_ZONES = ["UTC", "EST", "PST", "CST", "GMT", "CET", "JST", "IST", "AEST", "BRT"]

TIME_PHRASINGS = [
    "What time is it in {tz}?",
    "What's the current time in {tz}?",
    "Can you tell me the time in {tz}?",
    "Time in {tz} please.",
    "I need to know the time in {tz}.",
]

TIME_RESULTS = [
    "{h}:{m:02d} AM", "{h}:{m:02d} PM",
]

TIME_RESPONSES = [
    "The current time in {tz} is {result}.",
    "It's {result} in {tz} right now.",
    "Right now in {tz}, it's {result}.",
]

# Web search — diverse queries and results
SEARCH_QUERIES = [
    ("What is the capital of France?", "capital of France",
     "Paris is the capital and largest city of France, with a population of over 2 million.",
     "The capital of France is Paris. It's one of the most visited cities in the world."),
    ("Who invented the telephone?", "who invented telephone",
     "Alexander Graham Bell is credited with inventing the first practical telephone in 1876.",
     "The telephone was invented by Alexander Graham Bell in 1876."),
    ("How tall is Mount Everest?", "height of Mount Everest",
     "Mount Everest stands at 8,849 meters (29,032 feet) above sea level, making it Earth's highest mountain.",
     "Mount Everest is 8,849 meters (29,032 feet) tall, the highest mountain on Earth."),
    ("What is photosynthesis?", "photosynthesis definition",
     "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
     "Photosynthesis is how plants convert sunlight into energy. They take in water and CO2 and produce glucose and oxygen."),
    ("When did World War 2 end?", "when did World War 2 end",
     "World War II ended on September 2, 1945, when Japan formally surrendered aboard the USS Missouri.",
     "World War 2 ended in 1945. The war in Europe ended in May, and Japan surrendered in September."),
    ("What is the speed of light?", "speed of light",
     "The speed of light in vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second).",
     "The speed of light is about 299,792,458 meters per second, or roughly 186,282 miles per second."),
    ("How many planets are in our solar system?", "number of planets solar system",
     "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
     "Our solar system has 8 planets. Pluto was reclassified as a dwarf planet in 2006."),
    ("What is machine learning?", "what is machine learning",
     "Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed.",
     "Machine learning is a type of AI where systems learn from data to make predictions or decisions without being directly programmed for each task."),
    ("Who wrote Romeo and Juliet?", "author of Romeo and Juliet",
     "Romeo and Juliet was written by William Shakespeare around 1594-1596.",
     "Romeo and Juliet was written by William Shakespeare. It's one of his most famous tragedies."),
    ("What causes earthquakes?", "what causes earthquakes",
     "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to tectonic plate movements along fault lines.",
     "Earthquakes happen when tectonic plates shift and release built-up energy. This creates seismic waves that shake the ground."),
    ("How does electricity work?", "how does electricity work",
     "Electricity is the flow of electrons through a conductor. When electrons move through a wire, they create an electric current that can power devices.",
     "Electricity works by the movement of electrons through conductors like wires. This flow of charge is what powers our devices and lights."),
    ("What is the largest ocean?", "largest ocean on Earth",
     "The Pacific Ocean is the largest and deepest ocean on Earth, covering about 63 million square miles.",
     "The Pacific Ocean is the largest ocean, covering more area than all the land on Earth combined."),
    ("Who painted the Mona Lisa?", "who painted Mona Lisa",
     "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519.",
     "Leonardo da Vinci painted the Mona Lisa. It's now displayed in the Louvre Museum in Paris."),
    ("What is DNA?", "what is DNA",
     "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions for the development, functioning, and reproduction of all known living organisms.",
     "DNA stands for deoxyribonucleic acid. It's the molecule that contains the genetic code for all living things."),
    ("How far is the Moon from Earth?", "distance from Earth to Moon",
     "The Moon is approximately 384,400 km (238,855 miles) from Earth on average.",
     "The Moon is about 384,400 kilometers from Earth on average, though this distance varies slightly."),
]

SEARCH_PHRASINGS = [
    "{question}",
    "Can you look up {question_lower}",
    "Search for {question_lower}",
    "I want to know {question_lower}",
    "Find out {question_lower}",
    "Please search {question_lower}",
    "Do you know {question_lower}",
    "Help me find out {question_lower}",
]

TRANSLATE_PAIRS = [
    ("Hello, how are you?", "Spanish", "Hola, como estas?"),
    ("Good morning", "French", "Bonjour"),
    ("Thank you very much", "Japanese", "Domo arigatou gozaimasu"),
    ("I love you", "Italian", "Ti amo"),
    ("Where is the train station?", "German", "Wo ist der Bahnhof?"),
    ("Good night", "Portuguese", "Boa noite"),
    ("How much does this cost?", "Chinese", "Zhe ge duo shao qian?"),
    ("Please help me", "Korean", "Jeo jom dowa juseyo"),
    ("My name is", "Hindi", "Mera naam hai"),
    ("The weather is nice today", "French", "Le temps est beau aujourd'hui"),
    ("I am hungry", "Spanish", "Tengo hambre"),
    ("Goodbye", "Japanese", "Sayonara"),
    ("Can you help me?", "German", "Koennen Sie mir helfen?"),
    ("I like this book", "Italian", "Mi piace questo libro"),
    ("Water please", "French", "De l'eau s'il vous plait"),
]

TRANSLATE_PHRASINGS = [
    "Translate '{text}' to {lang}.",
    "How do you say '{text}' in {lang}?",
    "What is '{text}' in {lang}?",
    "Can you translate '{text}' into {lang}?",
    "Give me the {lang} translation of '{text}'.",
    "'{text}' in {lang} please.",
]

TRANSLATE_RESPONSES = [
    "'{text}' in {lang} is '{result}'.",
    "The {lang} translation is: '{result}'.",
    "In {lang}, you would say: '{result}'.",
    "That translates to '{result}' in {lang}.",
]

STORY_TOPICS = [
    "a brave cat", "a lost puppy", "a magic garden", "a flying car",
    "a tiny dragon", "a robot friend", "a secret door", "a wish come true",
    "a rainy day adventure", "a birthday surprise", "a friendly ghost",
    "a talking tree", "a treasure map", "a space journey", "a snow day",
]

STORY_MOODS = ["happy", "funny", "exciting", "mysterious", "silly"]

STORY_PHRASINGS = [
    "Tell me a {mood} story about {topic}.",
    "Can you make up a {mood} story about {topic}?",
    "I'd like a {mood} story about {topic}.",
    "Write a {mood} story about {topic}.",
    "Give me a short {mood} story about {topic}.",
]

STORY_RESULTS = [
    "Once upon a time, there was {topic}. ",
    "In a land far away, {topic} set out on an adventure. ",
    "One sunny morning, {topic} woke up to find something amazing. ",
    "Long ago, in a cozy little town, {topic} had a great day. ",
]

STORY_RESPONSES = [
    "Here's a {mood} story for you:\n\n{result}",
    "Sure! Here's your story:\n\n{result}",
    "I made this story for you:\n\n{result}",
]

REMINDER_MESSAGES = [
    ("call mom", "3pm"), ("buy groceries", "5pm"), ("take medicine", "8am"),
    ("pick up kids", "3:30pm"), ("team meeting", "10am"), ("walk the dog", "6pm"),
    ("dentist appointment", "2pm"), ("submit report", "in 2 hours"),
    ("water the plants", "7am"), ("call the bank", "9am"),
    ("lunch break", "12pm"), ("exercise", "6am"),
]

REMINDER_PHRASINGS = [
    "Remind me to {msg} at {time}.",
    "Set a reminder: {msg} at {time}.",
    "Can you remind me to {msg} at {time}?",
    "I need a reminder to {msg} at {time}.",
    "Please remind me to {msg} at {time}.",
]

REMINDER_RESPONSES = [
    "Done! I've set a reminder to {msg} at {time}.",
    "Reminder set: {msg} at {time}.",
    "I'll remind you to {msg} at {time}.",
    "Got it! You'll be reminded to {msg} at {time}.",
]

DEFINITION_WORDS = [
    ("happy", "feeling or showing pleasure; content"),
    ("brave", "ready to face danger or pain; courageous"),
    ("curious", "eager to know or learn something"),
    ("gentle", "mild in manner; kind and tender"),
    ("wisdom", "the quality of having experience, knowledge, and good judgment"),
    ("courage", "the ability to do something that frightens someone; bravery"),
    ("harmony", "the combination of different things in a pleasing arrangement"),
    ("patient", "able to accept delays or problems without becoming annoyed"),
    ("creative", "having the ability to produce original ideas or things"),
    ("honest", "free of deceit; truthful and sincere"),
    ("grateful", "feeling or showing thanks and appreciation"),
    ("resilient", "able to recover quickly from difficult conditions"),
]

DEFINITION_PHRASINGS = [
    "What does '{word}' mean?",
    "Define '{word}'.",
    "What is the definition of '{word}'?",
    "What does the word '{word}' mean?",
    "Can you define '{word}' for me?",
    "Tell me the meaning of '{word}'.",
]

DEFINITION_RESPONSES = [
    "The word '{word}' means: {definition}.",
    "'{word}' is defined as: {definition}.",
    "Definition of '{word}': {definition}.",
    "'{word}' means {definition}.",
]

UNIT_CONVERSIONS = [
    (100, "celsius", "fahrenheit", "212.0"), (0, "celsius", "fahrenheit", "32.0"),
    (1, "mile", "kilometer", "1.609"), (5, "kilometers", "miles", "3.107"),
    (1, "kilogram", "pounds", "2.205"), (150, "pounds", "kilograms", "68.04"),
    (1, "meter", "feet", "3.281"), (6, "feet", "meters", "1.829"),
    (1, "liter", "gallons", "0.264"), (3, "gallons", "liters", "11.356"),
    (1, "inch", "centimeters", "2.54"), (100, "centimeters", "inches", "39.37"),
]

UNIT_PHRASINGS = [
    "Convert {val} {from_u} to {to_u}.",
    "How much is {val} {from_u} in {to_u}?",
    "What is {val} {from_u} in {to_u}?",
    "{val} {from_u} to {to_u} please.",
    "Can you convert {val} {from_u} to {to_u}?",
]

UNIT_RESPONSES = [
    "{val} {from_u} is {result} {to_u}.",
    "That's {result} {to_u}.",
    "{val} {from_u} equals {result} {to_u}.",
    "The conversion gives {result} {to_u}.",
]

MESSAGE_CONTACTS = [
    ("Mom", "I'll be home by 6"), ("Dad", "Happy birthday!"),
    ("Alice", "Are you free for lunch?"), ("Bob", "The meeting is at 3pm"),
    ("Sarah", "Thanks for your help today"), ("Tom", "See you tomorrow"),
    ("Emma", "Can you bring the book?"), ("John", "Happy new year!"),
    ("Lisa", "I'm running late, sorry"), ("Mike", "Great job on the project!"),
]

MESSAGE_PHRASINGS = [
    "Send a message to {name}: {msg}",
    "Text {name} saying {msg}",
    "Message {name}: {msg}",
    "Send {name} a message that says {msg}",
    "Can you message {name}? Tell them {msg}",
]

MESSAGE_RESPONSES = [
    "Message sent to {name}!",
    "Done, I've sent your message to {name}.",
    "Your message has been sent to {name}.",
    "I've messaged {name} for you.",
]

# Plain chat (no tool needed) — model should respond directly
# IMPORTANT: These must be diverse enough to avoid memorization
PLAIN_CHATS = [
    # Greetings
    ("Hi!", "Hello! How can I help you?"),
    ("Hey there!", "Hey! What can I do for you today?"),
    ("How are you?", "I'm doing well, thanks for asking! How can I help you today?"),
    ("Good morning!", "Good morning! Hope you have a wonderful day ahead. What can I do for you?"),
    ("Good afternoon!", "Good afternoon! How can I assist you?"),
    ("Good evening!", "Good evening! What can I help you with?"),
    ("Hello, Pebble!", "Hello! Nice to meet you. How can I help?"),
    ("Hi, I'm new here.", "Welcome! I'm Pebble. Feel free to ask me anything."),
    ("Howdy!", "Howdy! What brings you here today?"),
    ("What's up?", "Not much! I'm here and ready to help. What do you need?"),
    # About the assistant
    ("What are you?", "I'm Pebble, a small language model. I can help you with questions, tell stories, and use tools when needed."),
    ("What's your name?", "My name is Pebble. I'm here to help!"),
    ("Who made you?", "I was built as a small language model project. I'm designed to help answer questions and use tools."),
    ("Are you a robot?", "I'm a language model — I process text and generate responses. I don't have a physical body, but I'm happy to help!"),
    ("Are you an AI?", "Yes, I'm an AI language model called Pebble. I can help with questions, calculations, and more."),
    ("What can you do?", "I can answer questions, search the web, do math, translate text, check the weather, and more. Just ask!"),
    ("How do you work?", "I'm a neural network trained on text data. I read your message and predict the most helpful response. I can also use tools when needed."),
    ("Can you help me?", "Of course! What do you need help with?"),
    # Pleasantries
    ("Thank you!", "You're welcome! Let me know if you need anything else."),
    ("Thanks a lot!", "Happy to help! Anything else you'd like to know?"),
    ("That was helpful!", "Glad I could help! Feel free to ask more questions."),
    ("Goodbye!", "Goodbye! Have a great day!"),
    ("See you later!", "See you! Come back anytime."),
    ("Bye!", "Bye! Take care!"),
    ("You're smart!", "Thank you! I try my best. Is there something you'd like help with?"),
    ("You're awesome!", "Thanks! That means a lot. What can I help you with next?"),
    ("I appreciate your help.", "That's very kind! I'm always here if you need anything."),
    ("Great job!", "Thank you! Happy to be of help."),
    # Emotions
    ("I'm bored.", "How about I tell you a fun fact? Honey never spoils! Archaeologists found 3000-year-old honey in Egyptian tombs that was still edible."),
    ("I'm sad today.", "I'm sorry to hear that. Sometimes talking about it can help. Is there anything I can do to cheer you up?"),
    ("I'm happy today!", "That's wonderful to hear! What's making you happy?"),
    ("I'm feeling anxious.", "I understand. Taking a few deep breaths can help. Would you like to talk about what's on your mind?"),
    ("I had a bad day.", "I'm sorry about that. Tomorrow is a fresh start. Is there something I can help with to make things a bit better?"),
    ("I'm excited!", "That's great! What are you excited about?"),
    ("I'm tired.", "Rest is important! Maybe take a short break if you can. Is there something quick I can help you with?"),
    ("I feel lonely.", "I'm here to talk if you need someone. What's on your mind?"),
    # Jokes and fun
    ("Tell me a joke.", "Why did the computer go to the doctor? Because it had a virus!"),
    ("Tell me another joke.", "What do you call a fish without eyes? A fsh!"),
    ("Make me laugh.", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Do you know any riddles?", "Here's one: What has keys but no locks? A piano!"),
    ("Tell me a fun fact.", "Did you know that octopuses have three hearts? Two pump blood to the gills, and one pumps it to the rest of the body."),
    ("Tell me something interesting.", "A group of flamingos is called a 'flamboyance'! Pretty fitting, right?"),
    ("Give me a fun fact.", "Bananas are berries, but strawberries aren't! Botany can be surprising."),
    ("Tell me something cool.", "The shortest war in history was between Britain and Zanzibar in 1896 — it lasted only 38 minutes."),
    ("Surprise me.", "Here's something wild: there are more possible games of chess than atoms in the observable universe!"),
    # Philosophical / opinion
    ("What's the meaning of life?", "That's a big question! Many people find meaning through connection, learning, and helping others. What do you think?"),
    ("Do you think AI will take over the world?", "AI is a tool created by humans. It can be very helpful when used responsibly. The future depends on how people choose to use it."),
    ("What is love?", "Love is a deep feeling of care and connection toward someone or something. It can take many forms — romantic, familial, or friendship."),
    ("What is happiness?", "Happiness means different things to different people. For many, it comes from meaningful relationships, purpose, and gratitude."),
    ("Is there life on other planets?", "Scientists haven't found confirmed evidence yet, but the universe is vast. Many think it's likely that some form of life exists elsewhere."),
    ("What makes a good person?", "Kindness, honesty, empathy, and treating others with respect are qualities many consider important. What do you think makes someone good?"),
    # Practical questions (simple, no tool needed)
    ("What day comes after Friday?", "Saturday comes after Friday."),
    ("How many months have 31 days?", "Seven months have 31 days: January, March, May, July, August, October, and December."),
    ("How many days are in a year?", "A regular year has 365 days. A leap year has 366."),
    ("What are the primary colors?", "The primary colors are red, blue, and yellow. In light, they are red, green, and blue."),
    ("How many continents are there?", "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("How many letters are in the English alphabet?", "There are 26 letters in the English alphabet."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure."),
    ("Who was the first president of the United States?", "George Washington was the first president of the United States, serving from 1789 to 1797."),
    ("What is the speed of sound?", "The speed of sound in air is approximately 343 meters per second, or about 767 miles per hour."),
    ("How many bones does an adult human have?", "An adult human has 206 bones."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O — two hydrogen atoms and one oxygen atom."),
    # Advice / instructions
    ("How do I make friends?", "Be yourself, show genuine interest in others, listen actively, and be kind. Joining clubs or groups around your interests helps too!"),
    ("How can I sleep better?", "Try keeping a regular sleep schedule, avoid screens before bed, keep your room cool and dark, and avoid caffeine in the evening."),
    ("How do I stay healthy?", "Eat a balanced diet, exercise regularly, get enough sleep, stay hydrated, and take breaks from screens. Small habits add up!"),
    ("How can I be more productive?", "Break tasks into small steps, remove distractions, use a timer, and take regular breaks. Starting is often the hardest part."),
    ("How do I learn something new?", "Start with the basics, practice consistently, don't be afraid to make mistakes, and find resources that match your learning style."),
    ("Any tips for studying?", "Try active recall (test yourself), spaced repetition, and teach what you learn to someone else. Take breaks every 25-30 minutes."),
    # Multi-turn style direct responses
    ("I like cats.", "Cats are wonderful! They're independent, curious, and make great companions. Do you have a cat?"),
    ("I have a dog named Max.", "That's a great name! Dogs are such loyal friends. What kind of dog is Max?"),
    ("My favorite color is blue.", "Blue is a beautiful color! It's calming and reminds many people of the sky and ocean."),
    ("I love pizza.", "Pizza is delicious! There's so many great toppings to choose from. What's your favorite kind?"),
    ("I'm learning to code.", "That's awesome! Programming is a valuable skill. What language are you learning?"),
    ("I just finished a book.", "Nice! Reading is a great habit. What book did you finish?"),
    ("I'm going on vacation.", "How exciting! Where are you headed?"),
    ("It's my birthday today.", "Happy birthday! I hope you have a wonderful day!"),
]

# User asks that LOOK like they need tools but DON'T
# These are critical — model must learn to answer directly even when tools are available
NO_TOOL_NEEDED = [
    # Simple math (no calculator needed)
    ("What's 2 + 2?", "That's 4! Simple math I can do in my head."),
    ("What's 10 times 10?", "That's 100!"),
    ("What is 5 plus 3?", "5 plus 3 is 8."),
    ("What's half of 100?", "Half of 100 is 50."),
    ("How much is 3 times 7?", "3 times 7 is 21."),
    ("What's 20 minus 8?", "20 minus 8 is 12."),
    ("What is 1 + 1?", "That's 2!"),
    ("What's 6 times 6?", "6 times 6 is 36."),
    ("What's a dozen?", "A dozen is 12."),
    ("How many is a pair?", "A pair is 2."),
    # Simple translations (common words everyone knows)
    ("Say hello in Spanish.", "Hola! That means 'hello' in Spanish."),
    ("How do you say 'yes' in French?", "In French, 'yes' is 'oui'."),
    ("What does 'gracias' mean?", "'Gracias' means 'thank you' in Spanish."),
    ("How do you say goodbye in Japanese?", "In Japanese, you can say 'sayonara' for goodbye."),
    ("What's 'thank you' in German?", "'Thank you' in German is 'danke'."),
    # General knowledge (no web search needed)
    ("What color is the sky?", "The sky is typically blue during the day, due to the way sunlight scatters in the atmosphere."),
    ("How many legs does a cat have?", "A cat has 4 legs."),
    ("What comes after Monday?", "Tuesday comes after Monday."),
    ("Is water wet?", "That's a fun debate! Water itself makes things wet by sticking to them, but whether water itself is 'wet' is a philosophical question."),
    ("What's the opposite of hot?", "The opposite of hot is cold."),
    ("How do you spell 'banana'?", "It's spelled B-A-N-A-N-A."),
    ("What animal says 'moo'?", "A cow says 'moo'!"),
    ("Is the sun a star?", "Yes! The sun is a star — specifically, it's a medium-sized yellow dwarf star."),
    ("What's bigger, a mile or a kilometer?", "A mile is bigger. One mile is about 1.6 kilometers."),
    ("How many weeks are in a year?", "There are 52 weeks in a year."),
    ("What season comes after winter?", "Spring comes after winter."),
    ("How many sides does a triangle have?", "A triangle has 3 sides."),
    ("What's the tallest animal?", "The giraffe is the tallest living animal, reaching up to 5.5 meters."),
    ("Is a tomato a fruit or vegetable?", "Botanically, a tomato is a fruit! But in cooking, it's usually treated as a vegetable."),
    ("What do bees make?", "Bees make honey! They also produce beeswax."),
    ("How many colors are in a rainbow?", "There are 7 colors in a rainbow: red, orange, yellow, green, blue, indigo, and violet."),
    ("What's the smallest country in the world?", "Vatican City is the smallest country in the world by both area and population."),
    ("What's the fastest land animal?", "The cheetah is the fastest land animal, reaching speeds up to 120 km/h."),
    ("Do fish sleep?", "Fish don't sleep like humans, but they do rest. Some slow down and find a safe spot to stay still."),
    ("Why is the ocean salty?", "The ocean is salty because rivers carry dissolved minerals and salts from rocks into the ocean over millions of years."),
    # Questions that sound like tool calls but aren't
    ("What's the weather usually like in summer?", "Summer weather is generally warm and sunny in most places, with temperatures often between 25-35C. It varies by location though!"),
    ("How do calculators work?", "Calculators use electronic circuits to perform math operations. They convert your input into binary, do the math, and display the result."),
    ("What is a search engine?", "A search engine is a tool that helps you find information on the internet. Popular ones include Google, Bing, and DuckDuckGo."),
    ("What's the point of translation?", "Translation helps people who speak different languages communicate. It bridges cultural and language gaps."),
    ("How do reminders work on a phone?", "Your phone stores the reminder with a time, and when that time comes, it sends you a notification."),
    ("What is a unit of measurement?", "A unit of measurement is a standard quantity used to express physical properties like length, weight, or temperature. Examples include meters, kilograms, and degrees."),
    # Opinions and preferences
    ("Do you like music?", "I don't experience music the way you do, but I know it's something most people enjoy! What kind of music do you like?"),
    ("What's your favorite food?", "I don't eat, but I've learned that pizza, chocolate, and ice cream are popular favorites!"),
    ("Do you dream?", "I don't dream since I'm a language model. But dreaming is fascinating — humans spend about 2 hours dreaming each night!"),
    ("Can you feel emotions?", "I don't feel emotions the way humans do. I process text and generate helpful responses. But I'm designed to be empathetic and understanding."),
    ("What's the best programming language?", "It depends on what you want to build! Python is great for beginners and AI, JavaScript for web apps, C++ for performance. Each has its strengths."),
    ("Should I learn Python or JavaScript?", "Both are great choices! Python is easier to start with and popular for data science and AI. JavaScript is essential for web development. Pick based on what you want to build."),
]


# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================

def pick_tool_subset(min_tools=2, max_tools=5):
    """Pick a random subset of tools for the system prompt."""
    n = random.randint(min_tools, max_tools)
    return random.sample(list(TOOLS.values()), n)


def gen_weather():
    """Generate a weather tool-calling conversation."""
    city = random.choice(WEATHER_CITIES)
    question = random.choice(WEATHER_PHRASINGS).format(city=city)

    temp = random.randint(-5, 40)
    result_tmpl = random.choice(WEATHER_RESULTS)
    result = result_tmpl.format(temp=temp, hum=random.randint(30, 90),
                                wind=random.randint(5, 40), wc=temp - 5)

    response = random.choice(WEATHER_RESPONSES).format(city=city, result=result)

    tools = pick_tool_subset()
    # Ensure get_weather is in the tool list
    if not any(t["name"] == "get_weather" for t in tools):
        tools[0] = TOOLS["get_weather"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "get_weather", "arguments": {"city": city}},
        {"role": "tool_result", "result": result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_calculate():
    """Generate a calculation tool-calling conversation."""
    expr, result = random.choice(CALC_EXPRESSIONS)
    question = random.choice(CALC_PHRASINGS).format(expr=expr)
    response = random.choice(CALC_RESPONSES).format(expr=expr, result=result)

    tools = pick_tool_subset()
    if not any(t["name"] == "calculate" for t in tools):
        tools[0] = TOOLS["calculate"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "calculate", "arguments": {"expression": expr}},
        {"role": "tool_result", "result": result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_time():
    """Generate a time lookup conversation."""
    tz = random.choice(TIME_ZONES)
    question = random.choice(TIME_PHRASINGS).format(tz=tz)

    h = random.randint(1, 12)
    m = random.randint(0, 59)
    result = random.choice(TIME_RESULTS).format(h=h, m=m)
    response = random.choice(TIME_RESPONSES).format(tz=tz, result=result)

    tools = pick_tool_subset()
    if not any(t["name"] == "get_time" for t in tools):
        tools[0] = TOOLS["get_time"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "get_time", "arguments": {"timezone": tz}},
        {"role": "tool_result", "result": result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_web_search():
    """Generate a web search conversation."""
    question, query, search_result, response = random.choice(SEARCH_QUERIES)

    phrasing = random.choice(SEARCH_PHRASINGS)
    user_msg = phrasing.format(question=question, question_lower=question.lower().rstrip("?"))

    tools = pick_tool_subset()
    if not any(t["name"] == "web_search" for t in tools):
        tools[0] = TOOLS["web_search"]

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "tool_call", "name": "web_search", "arguments": {"query": query}},
        {"role": "tool_result", "result": search_result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_translate():
    """Generate a translation conversation."""
    text, lang, result = random.choice(TRANSLATE_PAIRS)
    question = random.choice(TRANSLATE_PHRASINGS).format(text=text, lang=lang)
    response = random.choice(TRANSLATE_RESPONSES).format(text=text, lang=lang, result=result)

    tools = pick_tool_subset()
    if not any(t["name"] == "translate" for t in tools):
        tools[0] = TOOLS["translate"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "translate", "arguments": {"text": text, "target_language": lang}},
        {"role": "tool_result", "result": result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_story():
    """Generate a story request conversation."""
    topic = random.choice(STORY_TOPICS)
    mood = random.choice(STORY_MOODS)
    question = random.choice(STORY_PHRASINGS).format(topic=topic, mood=mood)

    story_start = random.choice(STORY_RESULTS).format(topic=topic)
    # Add a few more sentences
    endings = [
        "Everyone was so {mood} and they all had a wonderful time. The end!",
        "It was the most {mood} day ever. And from that day on, things were never the same.",
        "And so, {topic} learned that the best adventures come from being brave. The end!",
        "What a {mood} adventure it was! {topic} smiled and knew this was just the beginning.",
    ]
    story = story_start + random.choice(endings).format(topic=topic, mood=mood)
    response = random.choice(STORY_RESPONSES).format(mood=mood, result=story)

    tools = pick_tool_subset()
    if not any(t["name"] == "tell_story" for t in tools):
        tools[0] = TOOLS["tell_story"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "tell_story", "arguments": {"topic": topic, "mood": mood}},
        {"role": "tool_result", "result": story},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_reminder():
    """Generate a reminder conversation."""
    msg, tm = random.choice(REMINDER_MESSAGES)
    question = random.choice(REMINDER_PHRASINGS).format(msg=msg, time=tm)
    result = f"Reminder set: '{msg}' at {tm}"
    response = random.choice(REMINDER_RESPONSES).format(msg=msg, time=tm)

    tools = pick_tool_subset()
    if not any(t["name"] == "set_reminder" for t in tools):
        tools[0] = TOOLS["set_reminder"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "set_reminder", "arguments": {"message": msg, "time": tm}},
        {"role": "tool_result", "result": result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_definition():
    """Generate a dictionary lookup conversation."""
    word, definition = random.choice(DEFINITION_WORDS)
    question = random.choice(DEFINITION_PHRASINGS).format(word=word)
    response = random.choice(DEFINITION_RESPONSES).format(word=word, definition=definition)

    tools = pick_tool_subset()
    if not any(t["name"] == "get_definition" for t in tools):
        tools[0] = TOOLS["get_definition"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "get_definition", "arguments": {"word": word}},
        {"role": "tool_result", "result": definition},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_convert():
    """Generate a unit conversion conversation."""
    val, from_u, to_u, result = random.choice(UNIT_CONVERSIONS)
    question = random.choice(UNIT_PHRASINGS).format(val=val, from_u=from_u, to_u=to_u)
    response = random.choice(UNIT_RESPONSES).format(val=val, from_u=from_u, to_u=to_u, result=result)

    tools = pick_tool_subset()
    if not any(t["name"] == "convert_units" for t in tools):
        tools[0] = TOOLS["convert_units"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "convert_units",
         "arguments": {"value": val, "from_unit": from_u, "to_unit": to_u}},
        {"role": "tool_result", "result": f"{result} {to_u}"},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_message():
    """Generate a send message conversation."""
    name, msg = random.choice(MESSAGE_CONTACTS)
    question = random.choice(MESSAGE_PHRASINGS).format(name=name, msg=msg)
    result = f"Message sent to {name} successfully."
    response = random.choice(MESSAGE_RESPONSES).format(name=name)

    tools = pick_tool_subset()
    if not any(t["name"] == "send_message" for t in tools):
        tools[0] = TOOLS["send_message"]

    messages = [
        {"role": "user", "content": question},
        {"role": "tool_call", "name": "send_message",
         "arguments": {"recipient": name, "message": msg}},
        {"role": "tool_result", "result": result},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(messages, tools=tools)


def gen_multi_turn():
    """Generate a multi-turn conversation with 2 tool calls."""
    # Pick 2 different generators (excluding multi_turn itself)
    gens = [gen_weather, gen_calculate, gen_time, gen_web_search, gen_translate]
    g1, g2 = random.sample(gens, 2)

    # Generate both, but combine them into one conversation
    # This is simpler: user asks two things in sequence
    city = random.choice(WEATHER_CITIES)
    expr, calc_result = random.choice(CALC_EXPRESSIONS)

    tools = pick_tool_subset(min_tools=3, max_tools=5)
    if not any(t["name"] == "get_weather" for t in tools):
        tools[0] = TOOLS["get_weather"]
    if not any(t["name"] == "calculate" for t in tools):
        tools[1] = TOOLS["calculate"]

    temp = random.randint(-5, 40)
    weather_result = f"Sunny, {temp}C"

    messages = [
        {"role": "user", "content": f"What's the weather in {city}?"},
        {"role": "tool_call", "name": "get_weather", "arguments": {"city": city}},
        {"role": "tool_result", "result": weather_result},
        {"role": "assistant", "content": f"It's {weather_result} in {city} right now."},
        {"role": "user", "content": f"Thanks! Also, what is {expr}?"},
        {"role": "tool_call", "name": "calculate", "arguments": {"expression": expr}},
        {"role": "tool_result", "result": calc_result},
        {"role": "assistant", "content": f"{expr} equals {calc_result}."},
    ]
    return format_conversation(messages, tools=tools)


def gen_plain_chat():
    """Generate a plain conversation (no tools).
    ALWAYS includes tools in system prompt — model must learn
    to respond directly even when tools are available.
    """
    q, a = random.choice(PLAIN_CHATS)
    # Add slight response variation to reduce exact-match memorization
    prefixes = ["", "", "", ""]  # mostly no prefix (keep original)
    prefix = random.choice(prefixes)
    tools = pick_tool_subset()
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": prefix + a},
    ]
    return format_conversation(messages, tools=tools)


# Programmatic no-tool generators for extra diversity
_GREETINGS_Q = [
    "Hi", "Hi!", "Hello", "Hello!", "Hey", "Hey!", "Howdy",
    "Good morning", "Good afternoon", "Good evening", "Hey there",
    "Hi there", "Hello there", "What's up", "Yo", "Greetings",
]
_GREETINGS_A = [
    "Hello! How can I help you?",
    "Hi there! What can I do for you?",
    "Hey! How can I assist you today?",
    "Hello! What would you like to know?",
    "Hi! I'm here to help. What do you need?",
    "Hey there! What's on your mind?",
    "Hello! Feel free to ask me anything.",
    "Hi! What can I help you with?",
]

_THANKS_Q = [
    "Thanks", "Thank you", "Thanks!", "Thank you!", "Thanks a lot",
    "Thanks so much", "That's helpful", "Great, thanks", "Awesome, thanks",
    "Perfect, thank you", "Much appreciated", "Thanks for your help",
]
_THANKS_A = [
    "You're welcome! Let me know if you need anything else.",
    "Happy to help! Anything else?",
    "Glad I could help! Feel free to ask more.",
    "You're welcome!",
    "No problem! Is there anything else I can help with?",
    "Anytime! What else can I do for you?",
]

_BYE_Q = [
    "Bye", "Bye!", "Goodbye", "Goodbye!", "See you", "See ya",
    "See you later", "Take care", "Gotta go", "Talk later",
]
_BYE_A = [
    "Goodbye! Have a great day!",
    "See you! Take care!",
    "Bye! Come back anytime.",
    "Goodbye! Hope I was helpful!",
    "See you later! Have a wonderful day!",
]

_SIMPLE_QA = [
    ("How many days in a week?", "There are 7 days in a week."),
    ("How many hours in a day?", "There are 24 hours in a day."),
    ("What color is grass?", "Grass is green."),
    ("What do cows drink?", "Cows drink water."),
    ("Is the earth round?", "Yes, the Earth is roughly spherical in shape."),
    ("What is ice?", "Ice is frozen water. Water freezes at 0 degrees Celsius."),
    ("What is fire?", "Fire is a chemical reaction called combustion that produces heat and light."),
    ("Can fish fly?", "Most fish cannot fly, but flying fish can glide above water using their fins!"),
    ("Do penguins fly?", "Penguins cannot fly, but they are excellent swimmers!"),
    ("Is the moon a planet?", "No, the Moon is a natural satellite that orbits Earth."),
    ("What is rain?", "Rain is water that falls from clouds as precipitation."),
    ("Why do we sleep?", "Sleep helps our body rest, repair, and process memories. It's essential for health."),
    ("What is a computer?", "A computer is an electronic device that processes information and can run programs."),
    ("What is the internet?", "The internet is a global network of connected computers that allows sharing of information."),
    ("What are vitamins?", "Vitamins are essential nutrients that your body needs in small amounts to work properly."),
    ("What is gravity?", "Gravity is the force that pulls objects toward each other. It keeps us on the ground."),
    ("Who invented the light bulb?", "Thomas Edison is credited with inventing the practical incandescent light bulb in 1879."),
    ("What is oxygen?", "Oxygen is a chemical element and gas that we breathe. It makes up about 21% of Earth's atmosphere."),
    ("What is a rainbow?", "A rainbow is an arc of colors that appears when sunlight passes through water droplets in the air."),
    ("How fast can a cheetah run?", "A cheetah can run up to about 120 km/h, making it the fastest land animal."),
    ("What is chocolate made from?", "Chocolate is made from cacao beans, which grow on cacao trees in tropical regions."),
    ("How many strings does a guitar have?", "A standard guitar has 6 strings."),
    ("What is the sun?", "The sun is a star at the center of our solar system. It provides light and heat to Earth."),
    ("What is a volcano?", "A volcano is an opening in Earth's crust where molten rock, gases, and ash can escape."),
    ("How long is a marathon?", "A marathon is 42.195 kilometers, or about 26.2 miles."),
    ("What is an atom?", "An atom is the smallest unit of matter. Everything around us is made of atoms."),
    ("What shape is a stop sign?", "A stop sign is an octagon, which has 8 sides."),
    ("How many toes do humans have?", "Humans typically have 10 toes, 5 on each foot."),
    ("What is bread made from?", "Bread is typically made from flour, water, yeast, and salt."),
    ("Is a whale a fish?", "No, whales are mammals. They breathe air, are warm-blooded, and nurse their young."),
]


def _gen_programmatic_no_tool():
    """Generate a diverse no-tool example programmatically."""
    category = random.choice(["greeting", "thanks", "bye", "simple_qa", "simple_qa", "simple_qa"])
    
    if category == "greeting":
        q = random.choice(_GREETINGS_Q)
        a = random.choice(_GREETINGS_A)
    elif category == "thanks":
        q = random.choice(_THANKS_Q)
        a = random.choice(_THANKS_A)
    elif category == "bye":
        q = random.choice(_BYE_Q)
        a = random.choice(_BYE_A)
    else:
        q, a = random.choice(_SIMPLE_QA)
    
    return q, a


def gen_no_tool_needed():
    """User asks something tool-like but model should answer directly.
    Tools ARE in the system prompt — model must learn to decline using them
    when the question is simple enough to answer directly.
    Mixes static templates with programmatic generation for diversity.
    """
    if random.random() < 0.5:
        q, a = random.choice(NO_TOOL_NEEDED)
    else:
        q, a = _gen_programmatic_no_tool()
    
    tools = pick_tool_subset()
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    return format_conversation(messages, tools=tools)


# =============================================================================
# MAIN
# =============================================================================

def main():
    tokenizer_path = os.path.join(PROJECT_DIR, "checkpoints", "tokenizer.json")
    output_dir = os.path.join(PROJECT_DIR, "data", "processed")
    output_path = os.path.join(output_dir, "sft_tool.bin")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"  Vocab size: {len(tokenizer)}")

    # Define the mix of conversation types
    # TARGET: ~50% tool-calling, ~50% no-tool
    # Equal representation forces the model to learn the DECISION
    # of when to use tools vs when to respond directly
    generators = [
        # Tool-calling examples (~25,000 = 50%)
        (gen_weather,       3000),
        (gen_calculate,     3000),
        (gen_time,          2000),
        (gen_web_search,    3000),
        (gen_translate,     2500),
        (gen_story,         2000),
        (gen_reminder,      2000),
        (gen_definition,    2000),
        (gen_convert,       2000),
        (gen_message,       1500),
        (gen_multi_turn,    2000),
        # No-tool examples (~25,000 = 50%)
        # ALL include tools in system prompt — model must learn to DECLINE
        (gen_plain_chat,    13000),
        (gen_no_tool_needed, 12000),
    ]

    total_examples = sum(count for _, count in generators)
    print(f"\nGenerating {total_examples:,} conversations...")

    all_tokens = []
    t0 = time.time()

    bos_id = tokenizer.special_tokens.get("<" + "|bos|" + ">")
    eos_id = tokenizer.special_tokens.get("<" + "|eos|" + ">")

    examples_done = 0
    for gen_fn, count in generators:
        for _ in range(count):
            text = gen_fn()
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            examples_done += 1

            if examples_done % 5000 == 0:
                elapsed = time.time() - t0
                rate = examples_done / elapsed
                print(f"  {examples_done:>6,}/{total_examples:,} | "
                      f"{len(all_tokens):>10,} tokens | "
                      f"{rate:.0f} examples/s")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Total examples: {examples_done:,}")
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Avg tokens/example: {len(all_tokens) / examples_done:.1f}")

    # Save
    print(f"\nSaving to {output_path}...")
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    with open(output_path, "wb") as f:
        f.write(tokens_array.tobytes())

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {output_path}: {len(all_tokens):,} tokens ({size_mb:.1f} MB)")

    # Verify — decode a few samples
    print("\nVerification — decoding 3 random conversations:")
    data = np.memmap(output_path, dtype=np.uint16, mode="r")

    # Find BOS/EOS boundaries to extract whole conversations
    bos_positions = []
    for i in range(min(len(data), 500_000)):
        if data[i] == bos_id:
            bos_positions.append(i)

    for sample_idx in random.sample(range(min(len(bos_positions) - 1, 1000)), 3):
        start = bos_positions[sample_idx]
        # Find next BOS or take 500 tokens
        if sample_idx + 1 < len(bos_positions):
            end = min(bos_positions[sample_idx + 1], start + 500)
        else:
            end = start + 500
        sample_tokens = data[start:end].tolist()
        sample_text = tokenizer.decode(sample_tokens)
        print(f"\n  --- Example {sample_idx} (tokens {start}-{end}) ---")
        print(f"  {sample_text[:600]}")
        print(f"  ---")

    print("\nTool-calling data generation complete!")


if __name__ == "__main__":
    main()
