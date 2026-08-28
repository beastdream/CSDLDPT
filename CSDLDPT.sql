CREATE DATABASE IF NOT EXISTS wang_cbir CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE wang_cbir;

SELECT DATABASE();

CREATE TABLE categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE images (
    image_id INT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(500) NOT NULL UNIQUE,
    category_id INT NOT NULL,
    width INT,
    height INT,
    file_extension VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_image_category FOREIGN KEY (category_id) REFERENCES categories (category_id)
);

CREATE TABLE color_moment_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL,
    color_space VARCHAR(20) NOT NULL DEFAULT 'RGB',
    feature_type VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    r_mean DOUBLE NOT NULL,
    r_std DOUBLE NOT NULL,
    r_skew DOUBLE NOT NULL,
    g_mean DOUBLE NOT NULL,
    g_std DOUBLE NOT NULL,
    g_skew DOUBLE NOT NULL,
    b_mean DOUBLE NOT NULL,
    b_std DOUBLE NOT NULL,
    b_skew DOUBLE NOT NULL,
    CONSTRAINT fk_feature_image FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE,
    CONSTRAINT uq_image_feature UNIQUE (
        image_id,
        color_space,
        feature_type
    )
);

INSERT INTO
    categories (category_name)
VALUES ('africa'),
    ('beach'),
    ('buildings'),
    ('buses'),
    ('dinosaurs'),
    ('elephants'),
    ('flowers'),
    ('food'),
    ('horses'),
    ('mountains');

DESCRIBE categories;

DESCRIBE images;

DESCRIBE color_moment_features;