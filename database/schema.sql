-- Create database if not exists
CREATE DATABASE IF NOT EXISTS crop_prediction;
USE crop_prediction;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    nitrogen FLOAT NOT NULL,
    phosphorus FLOAT NOT NULL,
    potassium FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    ph FLOAT NOT NULL,
    rainfall FLOAT NOT NULL,
    yield_prediction VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Create crop_facts table
CREATE TABLE IF NOT EXISTS crop_facts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert some sample crop facts
INSERT INTO crop_facts (title, description) VALUES
('Soil pH Matters', 'Different crops thrive in different pH levels. Most crops prefer slightly acidic soil with pH between 6.0 and 6.8.'),
('NPK Balance', 'Nitrogen (N) promotes leaf growth, Phosphorus (P) helps root and flower development, and Potassium (K) improves overall plant health.'),
('Water Management', 'While water is essential for crop growth, overwatering can be as harmful as underwatering. Good drainage is crucial.'),
('Temperature Impact', 'Each crop has an optimal temperature range. Most crops grow best between 20°C and 30°C.'),
('Humidity Effects', 'High humidity can increase the risk of fungal diseases, while low humidity can increase water loss through transpiration.');

-- Fertilizer recommendations table
CREATE TABLE IF NOT EXISTS fertilizer_recommendations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    crop_name VARCHAR(50),
    soil_type VARCHAR(50),
    n_level VARCHAR(20),
    p_level VARCHAR(20),
    k_level VARCHAR(20),
    fertilizer_name VARCHAR(100),
    application_method TEXT,
    dosage TEXT
); 