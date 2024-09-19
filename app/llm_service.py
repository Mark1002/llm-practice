"""https://python.langchain.com/docs/tutorials/llm_chain."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models import LanguageModelInput


parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4")


def example1():
    messages = [
        SystemMessage(content="Translate the following from English into Italian"), # noqa
        HumanMessage(content="hi!"),
    ]

    chain = model | parser

    print(chain.invoke(messages))


def translate_model_chain() -> RunnableSerializable[LanguageModelInput, str]:
    system_template = "Translate the following into {language}:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    chain = prompt_template | model | parser
    return chain


def example2():
    chain = translate_model_chain()
    result = chain.invoke({"language": "chinese", "text": "hi"})
    print(result)


if __name__ == "__main__":
    example2()
