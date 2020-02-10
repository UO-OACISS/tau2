const char * database_schema = R"(
-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema default_schema
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Table `trial`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `trial` ;

CREATE TABLE IF NOT EXISTS `trial` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `name` TEXT NULL DEFAULT NULL,
  `created` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
  );


-- -----------------------------------------------------
-- Table `thread`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `thread` ;

CREATE TABLE IF NOT EXISTS `thread` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `trial` INTEGER UNSIGNED NOT NULL,
  `node_rank` INT UNSIGNED NOT NULL,
  `context_rank` INT UNSIGNED NOT NULL,
  `thread_rank` INT UNSIGNED NOT NULL,
  CONSTRAINT `thread_trial_key`
    FOREIGN KEY (`trial`)
    REFERENCES `trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `thread_trial` ON `thread` (`trial` ASC) ;


-- -----------------------------------------------------
-- Table `metric`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `metric` ;

CREATE TABLE IF NOT EXISTS `metric` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `trial` INTEGER UNSIGNED NOT NULL,
  `name` TEXT NOT NULL,
  CONSTRAINT `metric_trial_key`
    FOREIGN KEY (`trial`)
    REFERENCES `trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `metric_trial_index` ON `metric` (`trial` ASC) ;


-- -----------------------------------------------------
-- Table `timer`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `timer` ;

CREATE TABLE IF NOT EXISTS `timer` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `trial` INTEGER UNSIGNED NOT NULL,
  `parent` INTEGER UNSIGNED NULL DEFAULT NULL,
  `name` TEXT NOT NULL,
  `short_name` TEXT NOT NULL,
  `timergroup` TEXT NULL DEFAULT NULL,
  `source_file` TEXT NULL DEFAULT NULL,
  `line_number` INT NULL DEFAULT NULL,
  `line_number_end` INT NULL DEFAULT NULL,
  `column_number` INT NULL DEFAULT NULL,
  `column_number_end` INT NULL DEFAULT NULL,
  CONSTRAINT `timer_trial_key`
    FOREIGN KEY (`trial`)
    REFERENCES `trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `timer_parent_key`
    FOREIGN KEY (`parent`)
    REFERENCES `timer` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `timer_trial_index` ON `timer` (`trial` ASC) ;

CREATE INDEX `timer_trial_key` ON `timer` (`trial` ASC) ;

CREATE INDEX `timer_parent_key_idx` ON `timer` (`parent` ASC) ;


-- -----------------------------------------------------
-- Table `timer_value`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `timer_value` ;

CREATE TABLE IF NOT EXISTS `timer_value` (
  `timer` INTEGER UNSIGNED NOT NULL,
  `metric` INTEGER UNSIGNED NOT NULL,
  `value` DOUBLE NULL DEFAULT NULL,
  `thread` INTEGER UNSIGNED NOT NULL,
  CONSTRAINT `timer_value_timer_key`
    FOREIGN KEY (`timer`)
    REFERENCES `timer` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `timer_value_metric_key`
    FOREIGN KEY (`metric`)
    REFERENCES `metric` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `timer_value_thread_key`
    FOREIGN KEY (`thread`)
    REFERENCES `thread` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `timer_value_timer_key_idx` ON `timer_value` (`timer` ASC) ;

CREATE INDEX `timer_value_metric_key_idx` ON `timer_value` (`metric` ASC) ;

CREATE INDEX `timer_value_thread_key_idx` ON `timer_value` (`thread` ASC) ;


-- -----------------------------------------------------
-- Table `counter`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `counter` ;

CREATE TABLE IF NOT EXISTS `counter` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `trial` INTEGER UNSIGNED NOT NULL,
  `name` TEXT NOT NULL,
  CONSTRAINT `counter_trial_key`
    FOREIGN KEY (`trial`)
    REFERENCES `trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `counter_trial_index` ON `counter` (`trial` ASC) ;

CREATE INDEX `counter_trial_key` ON `counter` (`trial` ASC) ;


-- -----------------------------------------------------
-- Table `counter_value`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `counter_value` ;

CREATE TABLE IF NOT EXISTS `counter_value` (
  `counter` INTEGER UNSIGNED NOT NULL,
  `timer` INTEGER UNSIGNED NULL DEFAULT NULL,
  `thread` INTEGER UNSIGNED NOT NULL,
  `sample_count` INT NULL DEFAULT NULL,
  `maximum_value` DOUBLE NULL DEFAULT NULL,
  `minimum_value` DOUBLE NULL DEFAULT NULL,
  `mean_value` DOUBLE NULL DEFAULT NULL,
  `sum_of_squares` DOUBLE NULL DEFAULT NULL,
  CONSTRAINT `counter_value_counter_key`
    FOREIGN KEY (`counter`)
    REFERENCES `counter` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `counter_value_thread_key`
    FOREIGN KEY (`thread`)
    REFERENCES `thread` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `counter_value_timer_key`
    FOREIGN KEY (`timer`)
    REFERENCES `timer` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `counter_value_index` ON `counter_value` (`counter` ASC, `thread` ASC, `timer` ASC) ;

CREATE INDEX `counter_value_thread_key` ON `counter_value` (`thread` ASC) ;

CREATE INDEX `counter_value_timer_key_idx` ON `counter_value` (`timer` ASC) ;


-- -----------------------------------------------------
-- Table `metadata`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `metadata` ;

CREATE TABLE IF NOT EXISTS `metadata` (
  `trial` INTEGER UNSIGNED NOT NULL,
  `thread` INTEGER UNSIGNED NOT NULL,
  `name` TEXT NOT NULL,
  `value` TEXT NULL DEFAULT NULL,
  CONSTRAINT `metadata_trial_key`
    FOREIGN KEY (`trial`)
    REFERENCES `trial` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `metadata_thread_key`
    FOREIGN KEY (`thread`)
    REFERENCES `thread` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);

CREATE INDEX `primary_metadata_index` ON `metadata` (`trial` ASC) ;
)";