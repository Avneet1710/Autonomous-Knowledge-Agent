import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()


def reset_db(db_path: str, echo: bool = False):
    """Drop the existing .db file and recreate all tables."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    """Simple REPL interface for interacting with the support agent."""
    print(f"\n🎫 UDA-Hub Support — Session: {ticket_id}")
    print("Type 'quit', 'exit', or 'q' to end the session.\n")

    config = {"configurable": {"thread_id": ticket_id}}

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Agent: Goodbye! Thank you for contacting CultPass support.")
            break

        trigger = {"messages": [HumanMessage(content=user_input)]}

        try:
            result = agent.invoke(input=trigger, config=config)
            last_ai = result["messages"][-1].content
            print(f"Agent: {last_ai}\n")
        except Exception as e:
            print(f"[Error]: {e}\n")
