CREATE TABLE chat_history (

    id SERIAL PRIMARY KEY,

    user_input VARCHAR,

    bot_response VARCHAR,

    model VARCHAR,

    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()

);



CREATE TABLE documents (

    id SERIAL PRIMARY KEY,

    filename VARCHAR,

    content TEXT,

    embedding TEXT,

    timestamp TIMESTAMPTZ DEFAULT now()

);