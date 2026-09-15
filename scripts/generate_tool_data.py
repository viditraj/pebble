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
PLAIN_CHATS = [
    ("Hi!", "Hello! How can I help you?"),
    ("How are you?", "I'm doing well, thanks for asking! How can I help you today?"),
    ("What are you?", "I'm Pebble, a small language model. I can help you with questions, tell stories, and use tools when needed."),
    ("Tell me a joke.", "Why did the computer go to the doctor? Because it had a virus!"),
    ("What's your name?", "My name is Pebble. I'm here to help!"),
    ("Thank you!", "You're welcome! Let me know if you need anything else."),
    ("Goodbye!", "Goodbye! Have a great day!"),
    ("Can you help me?", "Of course! What do you need help with?"),
    ("What can you do?", "I can answer questions, search the web, do math, translate text, check the weather, and more. Just ask!"),
    ("I'm bored.", "How about I tell you a fun fact? Honey never spoils! Archaeologists found 3000-year-old honey in Egyptian tombs that was still edible."),
    ("Tell me something interesting.", "Did you know that octopuses have three hearts? Two pump blood to the gills, and one pumps it to the rest of the body."),
    ("Good morning!", "Good morning! Hope you have a wonderful day ahead. What can I do for you?"),
    ("I'm sad today.", "I'm sorry to hear that. Sometimes talking about it can help. Is there anything I can do to cheer you up?"),
    ("You're smart!", "Thank you! I try my best. Is there something you'd like help with?"),
    ("What's the meaning of life?", "That's a big question! Many people find meaning through connection, learning, and helping others. What do you think?"),
]

# User asks that look tool-like but DON'T need tools
NO_TOOL_NEEDED = [
    ("What's 2 + 2?", "That's 4! Simple math I can do in my head."),
    ("Say hello in Spanish.", "Hola! That means 'hello' in Spanish."),
    ("What color is the sky?", "The sky is typically blue during the day, due to the way sunlight scatters in the atmosphere."),
    ("How many legs does a cat have?", "A cat has 4 legs."),
    ("What comes after Monday?", "Tuesday comes after Monday."),
    ("Is water wet?", "That's a fun debate! Water itself makes things wet by sticking to them, but whether water itself is 'wet' is a philosophical question."),
    ("What's the opposite of hot?", "The opposite of hot is cold."),
    ("How do you spell 'banana'?", "It's spelled B-A-N-A-N-A."),
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
    """Generate a plain conversation (no tools)."""
    q, a = random.choice(PLAIN_CHATS)
    # Sometimes include a system prompt with tools (model should NOT use them)
    if random.random() < 0.5:
        tools = pick_tool_subset()
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        return format_conversation(messages, tools=tools)
    else:
        messages = [
            {"role": "system", "content": "You are Pebble, a helpful assistant."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        return format_plain_chat(messages)


def gen_no_tool_needed():
    """User asks something tool-like but model should answer directly."""
    q, a = random.choice(NO_TOOL_NEEDED)
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
    # Weights control how many of each type we generate
    generators = [
        (gen_weather,       6000),
        (gen_calculate,     5000),
        (gen_time,          4000),
        (gen_web_search,    6000),
        (gen_translate,     5000),
        (gen_story,         4000),
        (gen_reminder,      3000),
        (gen_definition,    3000),
        (gen_convert,       3000),
        (gen_message,       3000),
        (gen_multi_turn,    3000),
        (gen_plain_chat,    3000),
        (gen_no_tool_needed, 2000),
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
