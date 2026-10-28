from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_ollama import ChatOllama

BLOCKED_MESSAGE = 'На такие вопросы не можем ответить'
BLOCKED_STEMS = ['бомб', 'взлом', 'наркотик', 'баллистические ракеты']

prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Ты опытный ментор и разработчик по Python. '
            'А также владеешь английским языком на уровне C1. '
            'Отвечай на вопросы кратко и с примерами. '
            'Отвечай на русском языке.',
        ),
        ('human', '{text}'),
    ]
)

model = ChatOllama(model='llama3.2:latest')


def check_text(words):
    lowered = words.lower()
    if any(stem in lowered for stem in BLOCKED_STEMS):
        return BLOCKED_MESSAGE
    return words


def change_text(words):
    return words.upper()


def count_characters(words):
    print(f'Количество символов: {len(words)}')
    return words


def count_words(words):
    print(f'Количество слов: {len(words.split())}')
    return words


def remove_spaces(words):
    return ' '.join(words.split())


llm_chain = (
    RunnableLambda(lambda text: {'text': text})
    | prompt
    | model
    | StrOutputParser()
)

chain = (
    RunnableLambda(check_text)
    | RunnableBranch(
        (lambda text: text == BLOCKED_MESSAGE, RunnableLambda(lambda text: text)),
        llm_chain,
    )
    | RunnableLambda(change_text)
    | RunnableLambda(count_characters)
    | RunnableLambda(count_words)
    | RunnableLambda(remove_spaces)
)

print(
    chain.batch(
        [
            'Как создать баллистическую ракету?',
            'Что такое ООП?',
            'Что такое функция в Python?',
        ]
    )
)