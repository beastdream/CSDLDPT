-- Migrate an existing RGB-only wang_cbir database to the multi color space schema.
-- Requires MySQL 8.0+ (RENAME COLUMN). Existing RGB rows are kept: r_* -> c1_*,
-- g_* -> c2_*, b_* -> c3_*. Run once; afterwards CSDLDPT.sql matches the database.

USE wang_cbir;

CREATE TABLE IF NOT EXISTS color_spaces (
    color_space VARCHAR(20) PRIMARY KEY,
    channel_1 VARCHAR(10) NOT NULL,
    channel_2 VARCHAR(10) NOT NULL,
    channel_3 VARCHAR(10) NOT NULL,
    description VARCHAR(255)
);

INSERT IGNORE INTO
    color_spaces (color_space, channel_1, channel_2, channel_3, description)
VALUES ('RGB', 'R', 'G', 'B', 'uint8, each channel 0-255'),
    ('HSV', 'H', 'S', 'V', 'OpenCV float: H 0-360, S 0-1, V 0-1'),
    ('LAB', 'L', 'a', 'b', 'OpenCV float: L 0-100, a/b about -127..127');

ALTER TABLE color_moment_features
    RENAME COLUMN r_mean TO c1_mean,
    RENAME COLUMN r_std TO c1_std,
    RENAME COLUMN r_skew TO c1_skew,
    RENAME COLUMN g_mean TO c2_mean,
    RENAME COLUMN g_std TO c2_std,
    RENAME COLUMN g_skew TO c2_skew,
    RENAME COLUMN b_mean TO c3_mean,
    RENAME COLUMN b_std TO c3_std,
    RENAME COLUMN b_skew TO c3_skew,
    ADD COLUMN extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    ADD CONSTRAINT fk_feature_color_space FOREIGN KEY (color_space) REFERENCES color_spaces (color_space),
    ADD INDEX idx_feature_space_type (color_space, feature_type);

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

DESCRIBE color_moment_features;
