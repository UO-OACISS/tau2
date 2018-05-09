/*NNNNNNNNNN remove whatever indexes and tables are there already, in reverse order */

/* indexes */
/* MySQL won't let us drop indexes that don't exist.
DROP INDEX counter_value_index;
DROP INDEX counter_trial_index;
DROP INDEX secondary_metadata_index;
DROP INDEX primary_metadata_index;
DROP INDEX timer_value_index;
DROP INDEX timer_group_index;
DROP INDEX timer_trial_index;
DROP INDEX timer_call_data_thread;
*/

/* perfexplorer related */
DROP TABLE IF EXISTS analysis_settings;
DROP TABLE IF EXISTS analysis_result;
/* view related */
DROP TABLE IF EXISTS taudb_view_parameter;
DROP TABLE IF EXISTS taudb_view;
/* counter related */
DROP TABLE IF EXISTS counter_value;
DROP TABLE IF EXISTS counter;
/* metadata related */
DROP TABLE IF EXISTS secondary_metadata;
DROP TABLE IF EXISTS primary_metadata;
/* timer related */
DROP TABLE IF EXISTS timer_value;
DROP TABLE IF EXISTS timer_call_data;
DROP TABLE IF EXISTS timer_callpath;
DROP TABLE IF EXISTS timer_group;
DROP TABLE IF EXISTS timer_parameter;
/* data dimensions */
DROP TABLE IF EXISTS time_range;
DROP TABLE IF EXISTS timer;
DROP TABLE IF EXISTS metric;
DROP TABLE IF EXISTS thread;
/* the top level object */
DROP TABLE IF EXISTS trial;
/* static tables */
DROP TABLE IF EXISTS derived_thread_type;
DROP TABLE IF EXISTS data_source;
DROP TABLE IF EXISTS schema_version;

/****************************/
/* CREATE THE STATIC TABLES */
/****************************/

CREATE TABLE schema_version (
 version     INT NOT NULL,
 description TEXT NOT NULL
);
/* IF THE SCHEMA IS MODIFIED, INCREMENT THIS VALUE */
/* 0 = PERFDMF (ORIGINAL) */
/* 1 = TAUDB (APRIL, 2012) */
/*VALUES (1, 'TAUdb redesign from Spring, 2012');*/
INSERT INTO schema_version (version, description) 
  VALUES (2, 'Changes after Nov. 9, 2012 release');

/* These are our supported parsers. */
CREATE TABLE data_source (
 id          INT UNIQUE NOT NULL,
 name        TEXT NOT NULL,
 description TEXT
);

INSERT INTO data_source (name,id,description) 
  VALUES ('ppk',0,'TAU Packed profiles (TAU)');
INSERT INTO data_source (name,id,description) 
  VALUES ('TAU profiles',1,'TAU profiles (TAU)');
INSERT INTO data_source (name,id,description) 
  VALUES ('DynaProf',2,'PAPI DynaProf profiles (UTK)');
INSERT INTO data_source (name,id,description) 
  VALUES ('mpiP',3,'mpiP: Lightweight, Scalable MPI Profiling (Vetter, Chambreau)');
INSERT INTO data_source (name,id,description) 
  VALUES ('HPM',4,'HPM Toolkit profiles (IBM)');
INSERT INTO data_source (name,id,description) 
  VALUES ('gprof',5,'gprof profiles (GNU)');
INSERT INTO data_source (name,id,description) 
  VALUES ('psrun',6,'PerfSuite psrun profiles (NCSA)');
INSERT INTO data_source (name,id,description) 
  VALUES ('pprof',7,'TAU pprof.dat output (TAU)');
INSERT INTO data_source (name,id,description) 
  VALUES ('Cube',8,'Cube data (FZJ)');
INSERT INTO data_source (name,id,description) 
  VALUES ('HPCToolkit',9,'HPC Toolkit profiles (Rice Univ.)');
INSERT INTO data_source (name,id,description) 
  VALUES ('SNAP',10,'TAU Snapshot profiles (TAU)');
INSERT INTO data_source (name,id,description) 
  VALUES ('OMPP',11,'OpenMP Profiler profiles (Fuerlinger)');
INSERT INTO data_source (name,id,description) 
  VALUES ('PERIXML',12,'Data Exchange Format (PERI)');
INSERT INTO data_source (name,id,description) 
  VALUES ('GPTL',13,'General Purpose Timing Library (ORNL)');
INSERT INTO data_source (name,id,description) 
  VALUES ('Paraver',14,'Paraver profiles (BSC)');
INSERT INTO data_source (name,id,description) 
  VALUES ('IPM',15,'Integrated Performance Monitoring (NERSC)');
INSERT INTO data_source (name,id,description) 
  VALUES ('Google',16,'Google profiles (Google)');
INSERT INTO data_source (name,id,description) 
  VALUES ('Cube3',17,'Cube 3D profiles (FZJ)');
INSERT INTO data_source (name,id,description) 
  VALUES ('Gyro',100,'Self-timing profiles from Gyro application');
INSERT INTO data_source (name,id,description) 
  VALUES ('GAMESS',101,'Self-timing profiles from GAMESS application');
INSERT INTO data_source (name,id,description) 
  VALUES ('Other',999,'Other profiles');

/* threads make it convenient to identify timer values.
   Special values for thread_index:
   -1 mean (nulls ignored)
   -2 total
   -3 stddev (nulls ignored)
   -4 min
   -5 max
   -6 mean (nulls are 0 value)
   -7 stddev (nulls are 0 value)
*/

CREATE TABLE derived_thread_type (
 id INT NOT NULL,
 name TEXT NOT NULL,
 description TEXT NOT NULL
);

INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-1, 'MEAN', 'MEAN (nulls ignored)');
INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-2, 'TOTAL', 'TOTAL');
INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-3, 'STDDEV', 'STDDEV (nulls ignored)');
INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-4, 'MIN', 'MIN');
INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-5, 'MAX', 'MAX');
INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-6, 'MEAN', 'MEAN (nulls are 0 value)');
INSERT INTO derived_thread_type (id, name, description) 
  VALUES (-7, 'STDDEV', 'STDDEV (nulls are 0 value)');

/**************************/
/* CREATE THE TRIAL TABLE */
/**************************/

/* trials are the top level table */

CREATE TABLE trial (
 id                  SERIAL NOT NULL PRIMARY KEY,
 name                TEXT,
 /* where did this data come from? */
 data_source         INT,
 /* number of processes */
 node_count          INT,
 /* legacy values - these are actually "max" values - i.e. not all nodes have
  * this many threads */
 contexts_per_node   INT,
 /* how many threads per node? */
 threads_per_context INT,
 /* total number of threads */
 total_threads       INT,
 /* reference to the data source table. */
 FOREIGN KEY(data_source) REFERENCES data_source(id) 
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/******************************/
/* CREATE THE DATA DIMENSIONS */
/******************************/

/* threads are the "location" dimension */

CREATE TABLE thread (
 id           SERIAL NOT NULL PRIMARY KEY,
 /* trial this thread belongs to */
 trial        BIGINT UNSIGNED NOT NULL,
 /* process rank, really */
 node_rank    INT NOT NULL,
 /* legacy value */
 context_rank INT NOT NULL,
 /* thread rank relative to the process */
 thread_rank  INT NOT NULL,
 /* thread index from 0 to N-1 */
 thread_index INT NOT NULL,
 FOREIGN KEY(trial) REFERENCES trial(id) ON DELETE 
   NO ACTION ON UPDATE NO ACTION
);

/* metrics are things like num_calls, num_subroutines, TIME, PAPI
   counters, and derived metrics. */

CREATE TABLE metric (
 id      SERIAL NOT NULL PRIMARY KEY,
 /* trial this value belongs to */
 trial   BIGINT UNSIGNED NOT NULL,
 /* name of the metric */
 name    TEXT NOT NULL,
 /* if this metric is derived by one of the tools */
 derived BOOLEAN NOT NULL DEFAULT FALSE,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timers are timers, capturing some interval value.  For callpath or
   phase profiles, the parent refers to the calling function or phase. */

CREATE TABLE timer (
 id                SERIAL NOT NULL PRIMARY KEY,
 /* trial this value belongs to */
 trial             BIGINT UNSIGNED NOT NULL,
 /* name of the timer */
 name              TEXT NOT NULL,
 /* short name of the timer - without source or parameter info */
 short_name        TEXT NOT NULL,
 /* filename */
 source_file       TEXT,
 /* line number of the start of the block of code */
 line_number       INT,
 /* line number of the end of the block of code */
 line_number_end   INT,
 /* column number of the start of the block of code */
 column_number     INT,
 /* column number of the end of the block of code */
 column_number_end INT,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timer index on the trial and name columns */
CREATE INDEX timer_trial_index on timer (trial, name(256));

/***********************************/
/* CREATE THE TIMER RELATED TABLES */
/***********************************/

/* timer groups are the groups such as TAU_DEFAULT,
   MPI, OPENMP, TAU_PHASE, TAU_CALLPATH, TAU_PARAM, etc. 
   This mapping table allows for NxN mappings between timers
   and groups */

CREATE TABLE timer_group (
 timer BIGINT UNSIGNED,
 group_name  TEXT NOT NULL,
 FOREIGN KEY(timer) REFERENCES timer(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* index for faster queries into groups */
CREATE INDEX timer_group_index on timer_group (timer, group_name(256));

/* timer parameters are parameter based profile values. 
 * an example is foo (x,y) where x=4 and y=10. In that example,
 * timer would be the index of the timer with the
 * name 'foo (x,y) <x>=<4> <y>=<10>'. This table would have two
 * entries, one for the x value and one for the y value. */

CREATE TABLE timer_parameter (
 timer     BIGINT UNSIGNED,
 parameter_name  TEXT NOT NULL,
 parameter_value TEXT NOT NULL,
 FOREIGN KEY(timer) REFERENCES timer(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timer callpath have the information about the call graph in a trial.
 * If the profile is "flat", these will all have no parents. Otherwise,
 * the parent points to a node in the callgraph, the calling timer 
 * (function). */

CREATE TABLE timer_callpath (
 id        SERIAL NOT NULL PRIMARY KEY,
 /* what timer is this? */
 timer     BIGINT UNSIGNED NOT NULL,
 /* what is the parent timer? */
 parent    BIGINT UNSIGNED,
 FOREIGN KEY(timer) REFERENCES timer(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(parent) REFERENCES timer_callpath(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* By definition, profiles have no time data. However, there are a few
 * examples where time ranges make sense, such as tracking call stacks
 * or associating metadata to a particular phase. The time_range table
 * is used to give other measurements a time context. The iteration
 * start and end can be used to indicate which loop iterations or 
 * calls to a function are relevant for this time range. */

CREATE TABLE time_range (
 id SERIAL NOT NULL PRIMARY KEY,
 /* starting iteration */
 iteration_start INT NOT NULL,
 /* ending iteration. */
 iteration_end INT,
 /* starting timestamp */
 time_start BIGINT NOT NULL,
 /* ending timestamp. */
 time_end BIGINT
);

/* timer_call_data records have the dynamic information for when a node
 * in the callgraph is visited by a thread. If you are tracking dynamic
 * callstacks, you would use the time_range field. If you are storing
 * snapshot data, you would use the time_range field. */

CREATE TABLE timer_call_data (
 id          SERIAL NOT NULL PRIMARY KEY,
 /* what callgraph node is this? */
 timer_callpath       BIGINT UNSIGNED NOT NULL,
 /* what thread is this? */
 thread      BIGINT UNSIGNED NOT NULL,
 /* how many times this timer was called */
 calls       INT,
 /* how many subroutines this timer called */
 subroutines INT,
 /* what is the time_range? this is for supporting snapshots */
 time_range  BIGINT UNSIGNED,
 FOREIGN KEY(timer_callpath) REFERENCES timer_callpath(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(thread) REFERENCES thread(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(time_range) REFERENCES time_range(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timer values have the timer of one timer
   on one thread for one metric, at one location on the callgraph. */

CREATE TABLE timer_value (
 /* what node in the callgraph and thread is this? */
 timer_call_data       BIGINT UNSIGNED NOT NULL,
 /* what metric is this? */
 metric                BIGINT UNSIGNED NOT NULL,
 /* The inclusive value for this timer */
 inclusive_value       DOUBLE PRECISION,
 /* The exclusive value for this timer */
 exclusive_value       DOUBLE PRECISION,
 /* The inclusive percent for this timer */
 inclusive_percent     DOUBLE PRECISION,
 /* The exclusive percent for this timer */
 exclusive_percent     DOUBLE PRECISION,
 /* The variance for this timer */
 sum_exclusive_squared DOUBLE PRECISION,
 FOREIGN KEY(timer_call_data) REFERENCES timer_call_data(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(metric) REFERENCES metric(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* one metric, one thread, one timer */
CREATE INDEX timer_value_index on timer_value (timer_call_data, metric);

/*************************************/
/* CREATE THE COUNTER RELATED TABLES */
/*************************************/

/* counters measure some counted value. */

CREATE TABLE counter (
 id          SERIAL      NOT NULL PRIMARY KEY,
 trial       BIGINT UNSIGNED         NOT NULL,
 name        TEXT        NOT NULL,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* counter index on the trial and name columns */
CREATE INDEX counter_trial_index on counter (trial, name(256));

CREATE TABLE counter_value (
 /* what counter is this? */
 counter            BIGINT UNSIGNED NOT NULL,
 /* where in the callgraph? */
 timer_callpath     BIGINT UNSIGNED,
 /* what thread is this? */
 thread             BIGINT UNSIGNED NOT NULL,
 /* The total number of samples */
 sample_count       INT,         
 /* The maximum value seen */
 maximum_value      DOUBLE PRECISION,
 /* The minimum value seen */
 minimum_value      DOUBLE PRECISION,
 /* The mean value seen */
 mean_value         DOUBLE PRECISION,
 /* The variance for this counter */
 standard_deviation DOUBLE PRECISION,
 FOREIGN KEY(counter) REFERENCES counter(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(timer_callpath) REFERENCES timer_callpath(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(thread) REFERENCES thread(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* one thread, one counter */
CREATE INDEX counter_value_index on counter_value (counter, thread);

/**************************************/
/* CREATE THE METADATA RELATED TABLES */
/**************************************/

/* primary metadata is metadata that is not nested, does not
   contain unique data for each thread. */

CREATE TABLE primary_metadata (
 trial    BIGINT UNSIGNED NOT NULL,
 name     TEXT NOT NULL,
 value    TEXT,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* create an index for faster queries against the primary_metadata table */
CREATE INDEX primary_metadata_index on primary_metadata (trial, name(256));

/* secondary metadata is metadata that could be nested, could
   contain unique data for each thread, and could be an array. */

CREATE TABLE secondary_metadata (
 id       VARCHAR(128) NOT NULL PRIMARY KEY,
 /* trial this value belongs to */
 trial    BIGINT UNSIGNED NOT NULL,
 /* this metadata value could be associated with a thread */
 thread   BIGINT UNSIGNED,
 /* this metadata value could be associated with a timer that happened */
 timer_callpath    BIGINT UNSIGNED,
 /* which call to the context timer was this? */
 time_range    BIGINT UNSIGNED,
 /* this metadata value could be a nested structure */
 parent   VARCHAR(128),
 /* the name of the metadata field */
 name     TEXT NOT NULL,
 /* the value of the metadata field */
 value    TEXT,
 /* this metadata value could be an array - so tokenize it */
 is_array BOOLEAN DEFAULT FALSE,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(thread) REFERENCES thread(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(timer_callpath) REFERENCES timer_callpath(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(parent) REFERENCES secondary_metadata(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(time_range) REFERENCES time_range(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* create an index for faster queries against the secondary_metadata table */
CREATE INDEX secondary_metadata_index on secondary_metadata (trial, name(256), thread, parent);

/**************************************/
/* CREATE THE METADATA RELATED TABLES */
/**************************************/

/* this is the view table, which organizes and filters trials */
create table taudb_view (
	id					SERIAL			NOT NULL	PRIMARY KEY,
	/* views can be nested */
	parent				BIGINT UNSIGNED			NULL,
	/* name of the view */
	name				TEXT	NOT NULL,
	/* view conjoin type for parameters */
	conjoin				TEXT	NOT NULL,
	FOREIGN KEY (parent) REFERENCES taudb_view(id)
	  ON DELETE CASCADE ON UPDATE CASCADE
);

create table taudb_view_parameter (
	/* the view ID */
	taudb_view			BIGINT UNSIGNED	NOT NULL,
	/* the table name for the where clause */
	table_name			TEXT	NOT NULL,
	/* the column name for the where clause.
	   If the table_name is one of the metadata tables, this is the 
	   value of the "name" column */
	column_name			TEXT	NOT NULL,
	/* the operator for the where clause */
	operator			TEXT	NOT NULL,
	/* the value for the where clause */
	value				TEXT	NOT NULL,
	FOREIGN KEY (taudb_view) REFERENCES taudb_view(id)
	  ON DELETE CASCADE ON UPDATE CASCADE
);

/* simple view of all trials */
INSERT INTO taudb_view (parent, name, conjoin) VALUES (NULL, 'All Trials', 'and');
/* must have a parameter or else the sub views for this view do not work correctly*/
INSERT INTO taudb_view_parameter (taudb_view, table_name, column_name, operator, value) VALUES (1, 'trial', 'total_threads', '>', '-1');




/* the application and experiment columns are not used in the latest schema, but
   keeping them makes the code in PerfExplorer simpler. */
create table analysis_settings (
    id                  SERIAL          NOT NULL    PRIMARY KEY,
    taudb_view          BIGINT UNSIGNED         NULL,
    application         INTEGER         NULL,
    experiment          INTEGER         NULL,
    trial               BIGINT UNSIGNED         NULL,
    metric              BIGINT UNSIGNED         NULL,
    method              VARCHAR(255)    NOT NULL,
    dimension_reduction VARCHAR(255)    NOT NULL,
    normalization       VARCHAR(255)    NOT NULL,
    FOREIGN KEY (taudb_view) REFERENCES taudb_view(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (trial) REFERENCES trial(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (metric) REFERENCES metric(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

create table analysis_result (
    id                  SERIAL          NOT NULL    PRIMARY KEY,
    analysis_settings   INTEGER         NOT NULL,
    description         VARCHAR(255)    NOT NULL,
    thumbnail_size      INTEGER         NULL,
    image_size          INTEGER         NULL,
    thumbnail           BLOB           NULL,
    image               BLOB           NULL,
    result_type         INTEGER         NOT NULL
);

 /* Performance indexes! */
create index trial_name_index on trial(name(256));
create index timer_name_index on timer(name(256));
CREATE INDEX timer_callpath_parent on timer_callpath(parent);
CREATE INDEX thread_trial on thread(trial);
CREATE INDEX timer_call_data_timer_callpath on timer_call_data(timer_callpath);
CREATE INDEX counter_name_index on counter(name(256));
CREATE INDEX timer_call_data_thread on timer_call_data(thread);

/* SHORT TERM FIX! These views make sure that charts (mostly) work... for now. */

DROP VIEW IF EXISTS interval_location_profile;
DROP VIEW IF EXISTS interval_mean_summary;
DROP VIEW IF EXISTS interval_total_summary;
DROP VIEW IF EXISTS interval_event_value;
DROP VIEW IF EXISTS interval_event;
DROP VIEW IF EXISTS atomic_location_profile;
DROP VIEW IF EXISTS atomic_mean_summary;
DROP VIEW IF EXISTS atomic_total_summary;
DROP VIEW IF EXISTS atomic_event_value;
DROP VIEW IF EXISTS atomic_event;

CREATE OR REPLACE VIEW interval_event 
(id, trial, name, group_name, source_file, line_number, line_number_end) 
AS  
SELECT tcp.id, t.trial, t.name, tg.group_name,  
t.source_file, t.line_number, t.line_number_end  
FROM timer_callpath tcp  
INNER JOIN timer t ON tcp.timer = t.id  
INNER JOIN timer_group tg ON tg.timer = t.id; 

CREATE OR REPLACE VIEW interval_event_value 
(interval_event, node, context, thread, metric, inclusive_percentage,  
inclusive, exclusive_percentage, exclusive, calls, subroutines,  
inclusive_per_call, sum_exclusive_squared) 
AS SELECT tcd.timer_callpath, t.node_rank, t.context_rank,  
t.thread_rank, tv.metric, tv.inclusive_percent,  
tv.inclusive_value, tv.exclusive_percent, tv.exclusive_value, tcd.calls, tcd.subroutines, 
tv.inclusive_value / tcd.calls, tv.sum_exclusive_squared 
FROM timer_value tv 
INNER JOIN timer_call_data tcd on tv.timer_call_data = tcd.id 
INNER JOIN thread t on tcd.thread = t.id; 

CREATE OR REPLACE VIEW interval_location_profile 
AS SELECT * from interval_event_value WHERE thread >= 0; 
 
CREATE OR REPLACE VIEW interval_total_summary 
AS SELECT * from interval_event_value WHERE thread = -2; 
 
CREATE OR REPLACE VIEW interval_mean_summary 
AS SELECT * from interval_event_value WHERE thread = -1; 
 
 
CREATE OR REPLACE VIEW atomic_event  
(id, trial, name, group_name, source_file, line_number) 
AS SELECT c.id, c.trial, c.name, NULL, NULL, NULL 
FROM counter c; 

CREATE OR REPLACE VIEW atomic_event_value 
(atomic_event, node, context, thread, sample_count, 
maximum_value, minimum_value, mean_value, standard_deviation) 
AS SELECT cv.counter, t.node_rank, t.context_rank, t.thread_rank, cv.sample_count, 
cv.maximum_value, cv.minimum_value, cv.mean_value, cv.standard_deviation 
FROM counter_value cv 
INNER JOIN thread t ON cv.thread = t.id;
 
CREATE OR REPLACE VIEW atomic_location_profile 
AS SELECT * FROM atomic_event_value WHERE thread >= 0; 
 
CREATE OR REPLACE VIEW atomic_total_summary 
AS SELECT * FROM atomic_event_value WHERE thread = -2; 
 
CREATE OR REPLACE VIEW atomic_mean_summary 
AS SELECT * FROM atomic_event_value WHERE thread >= -1; 


