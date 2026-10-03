-- WANG/Corel-1K CBIR schema (multi color space Color Moments).
-- Fresh install: run this whole file.
-- Existing database created by the old RGB-only schema (r_mean, g_mean, ...):
-- run sql/migrations/001_multi_color_space.sql instead.

CREATE DATABASE IF NOT EXISTS wang_cbir CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE wang_cbir;

SELECT DATABASE();

CREATE TABLE IF NOT EXISTS categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS images (
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

-- Describes what channel 1/2/3 mean for each color space and their value range.
CREATE TABLE IF NOT EXISTS color_spaces (
    color_space VARCHAR(20) PRIMARY KEY,
    channel_1 VARCHAR(10) NOT NULL,
    channel_2 VARCHAR(10) NOT NULL,
    channel_3 VARCHAR(10) NOT NULL,
    description VARCHAR(255)
);

-- One row = one 9D Color Moments vector of one image in one color space.
-- c1/c2/c3 follow the channel order in color_spaces (e.g. HSV: c1=H, c2=S, c3=V).
CREATE TABLE IF NOT EXISTS color_moment_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL,
    color_space VARCHAR(20) NOT NULL DEFAULT 'RGB',
    feature_type VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    c1_mean DOUBLE NOT NULL,
    c1_std DOUBLE NOT NULL,
    c1_skew DOUBLE NOT NULL,
    c2_mean DOUBLE NOT NULL,
    c2_std DOUBLE NOT NULL,
    c2_skew DOUBLE NOT NULL,
    c3_mean DOUBLE NOT NULL,
    c3_std DOUBLE NOT NULL,
    c3_skew DOUBLE NOT NULL,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_feature_image FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE,
    CONSTRAINT fk_feature_color_space FOREIGN KEY (color_space) REFERENCES color_spaces (color_space),
    CONSTRAINT uq_image_feature UNIQUE (
        image_id,
        color_space,
        feature_type
    ),
    INDEX idx_feature_space_type (color_space, feature_type)
);

-- One row per evaluated configuration (color spaces x distance x normalization).
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    method VARCHAR(50) NOT NULL,
    distance_metric VARCHAR(20) NOT NULL,
    normalization VARCHAR(20) NOT NULL,
    feature_dim INT NOT NULL,
    num_queries INT NOT NULL,
    precision_at_5 DOUBLE, precision_at_10 DOUBLE, precision_at_20 DOUBLE,
    recall_at_5 DOUBLE, recall_at_10 DOUBLE, recall_at_20 DOUBLE,
    f1_at_5 DOUBLE, f1_at_10 DOUBLE, f1_at_20 DOUBLE,
    map_score DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_experiment_config UNIQUE (
        experiment_name,
        method,
        distance_metric,
        normalization
    )
);

CREATE TABLE IF NOT EXISTS experiment_category_results (
    run_id INT NOT NULL,
    category_id INT NOT NULL,
    num_queries INT NOT NULL,
    precision_at_5 DOUBLE, precision_at_10 DOUBLE, precision_at_20 DOUBLE,
    recall_at_5 DOUBLE, recall_at_10 DOUBLE, recall_at_20 DOUBLE,
    f1_at_5 DOUBLE, f1_at_10 DOUBLE, f1_at_20 DOUBLE,
    map_score DOUBLE,
    PRIMARY KEY (run_id, category_id),
    CONSTRAINT fk_result_run FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id) ON DELETE CASCADE,
    CONSTRAINT fk_result_category FOREIGN KEY (category_id) REFERENCES categories (category_id)
);

INSERT IGNORE INTO
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

INSERT IGNORE INTO
    color_spaces (color_space, channel_1, channel_2, channel_3, description)
VALUES ('RGB', 'R', 'G', 'B', 'uint8, each channel 0-255'),
    ('HSV', 'H', 'S', 'V', 'OpenCV float: H 0-360, S 0-1, V 0-1'),
    ('LAB', 'L', 'a', 'b', 'OpenCV float: L 0-100, a/b about -127..127');

DESCRIBE categories;

DESCRIBE images;

DESCRIBE color_spaces;

DESCRIBE color_moment_features;
