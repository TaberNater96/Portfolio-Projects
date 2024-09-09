-- Create inside the tagged_climate_corpuses database
USE tagged_climate_corpuses

-- Training Data Table
CREATE TABLE IF NOT EXISTS TrainingData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    corpus TEXT,
    classification TEXT,
    topic TEXT,
    sentiment TEXT,
    summary TEXT
)

-- Evaluation Data Table
CREATE TABLE IF NOT EXISTS EvaluationData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    corpus TEXT,
    classification TEXT,
    topic TEXT,
    sentiment TEXT,
    summary TEXT
);