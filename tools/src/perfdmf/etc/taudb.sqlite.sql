/* remove whatever indexes and tables are there already, in reverse order */

/* indexes */
DROP INDEX IF EXISTS counter_value_index;
DROP INDEX IF EXISTS counter_trial_index;
DROP INDEX IF EXISTS secondary_metadata_index;
DROP INDEX IF EXISTS primary_metadata_index;
DROP INDEX IF EXISTS timer_value_index;
DROP INDEX IF EXISTS timer_group_index;
DROP INDEX IF EXISTS timer_trial_index;

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
DROP TABLE IF EXISTS timer_callpath;
DROP TABLE IF EXISTS timer_call_data;
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
DROP TABLE IF EXISTS time_range_type;
DROP TABLE IF EXISTS derived_thread_type;
DROP TABLE IF EXISTS data_source;
DROP TABLE IF EXISTS schema_version;

/****************************/
/* CREATE THE STATIC TABLES */
/****************************/

CREATE TABLE schema_version (
 version     INT NOT NULL,
 description VARCHAR NOT NULL
);
/* IF THE SCHEMA IS MODIFIED, INCREMENT THIS VALUE */
/* 0 = PERFDMF (ORIGINAL) */
/* 1 = TAUDB (APRIL, 2012) */
  /*VALUES (1, 'TAUdb redesign from Spring, 2012');*/
INSERT INTO schema_version (version, description) 
  VALUES (2, 'Changes after Nov. 9, 2012 release (TAU 2.22)');

/* These are our supported parsers. */
CREATE TABLE data_source (
 id          INT UNIQUE NOT NULL,
 name        VARCHAR NOT NULL,
 description VARCHAR
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
 name VARCHAR NOT NULL,
 description VARCHAR NOT NULL
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
 id                  INTEGER PRIMARY KEY AUTOINCREMENT,
 name                VARCHAR,
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
 id           INTEGER PRIMARY KEY AUTOINCREMENT,
 /* trial this thread belongs to */
 trial        INT NOT NULL,
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
 id      INTEGER PRIMARY KEY AUTOINCREMENT,
 /* trial this value belongs to */
 trial   INT NOT NULL,
 /* name of the metric */
 name    VARCHAR NOT NULL,
 /* if this metric is derived by one of the tools */
 derived BOOLEAN NOT NULL DEFAULT FALSE,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timers are timers, capturing some interval value.  For callpath or
   phase profiles, the parent refers to the calling function or phase. */

CREATE TABLE timer (
 id                INTEGER PRIMARY KEY AUTOINCREMENT,
 /* trial this value belongs to */
 trial             INT NOT NULL,
 /* name of the timer */
 name              VARCHAR NOT NULL,
 /* short name of the timer - without source or parameter info */
 short_name        VARCHAR NOT NULL,
 /* filename */
 source_file       VARCHAR,
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
CREATE INDEX timer_trial_index on timer (trial, name);

/***********************************/
/* CREATE THE TIMER RELATED TABLES */
/***********************************/

/* timer groups are the groups such as TAU_DEFAULT,
   MPI, OPENMP, TAU_PHASE, TAU_CALLPATH, TAU_PARAM, etc. 
   This mapping table allows for NxN mappings between timers
   and groups */

CREATE TABLE timer_group (
 timer INT,
 group_name  VARCHAR NOT NULL,
 FOREIGN KEY(timer) REFERENCES timer(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* index for faster queries into groups */
CREATE INDEX timer_group_index on timer_group (timer, group_name);

/* timer parameters are parameter based profile values. 
 * an example is foo (x,y) where x=4 and y=10. In that example,
 * timer would be the index of the timer with the
 * name 'foo (x,y) <x>=<4> <y>=<10>'. This table would have two
 * entries, one for the x value and one for the y value. */

CREATE TABLE timer_parameter (
 timer     INT,
 parameter_name  VARCHAR NOT NULL,
 parameter_value VARCHAR NOT NULL,
 FOREIGN KEY(timer) REFERENCES timer(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timer callpath have the information about the call graph in a trial.
 * If the profile is "flat", these will all have no parents. Otherwise,
 * the parent points to a node in the callgraph, the calling timer 
 * (function). */

CREATE TABLE timer_callpath (
 id        INTEGER PRIMARY KEY AUTOINCREMENT,
 /* what timer is this? */
 timer     INT NOT NULL,
 /* what is the parent timer? */
 parent    INT,
 FOREIGN KEY(timer) REFERENCES timer(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(parent) REFERENCES timer_callpath(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* By definition, profiles have no time data. However, there are a few
 * examples where time ranges make sense, such as tracking call stacks
 * or associating metadata to a particular phase. The time_range table
 * is used to give other measurements a time context. */

CREATE TABLE time_range_type (
 id INT NOT NULL PRIMARY KEY,
 name VARCHAR NOT NULL
);

insert into time_range_type (id, name) values (1, 'TIMESTAMP');
insert into time_range_type (id, name) values (2, 'ITERATION NUMBER');
insert into time_range_type (id, name) values (3, 'CALL NUMBER');

CREATE TABLE time_range (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 /* what type of time_range is this? */
 type INT NOT NULL,
 /* starting value */
 start INT NOT NULL,
 /* ending value. Null indicates end = start */
 end INT,
 FOREIGN KEY(type) REFERENCES time_range_type(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* timer_call_data records have the dynamic information for when a node
 * in the callgraph is visited by a thread. If you are tracking dynamic
 * callstacks, you would use the time_range field. If you are storing
 * snapshot data, you would use the time_range field. */

CREATE TABLE timer_call_data (
 id          INTEGER PRIMARY KEY AUTOINCREMENT,
 /* what callgraph node is this? */
 timer_callpath       INT NOT NULL,
 /* what thread is this? */
 thread      INT NOT NULL,
 /* how many times this timer was called */
 calls       INT,
 /* how many subroutines this timer called */
 subroutines INT,
 /* what is the time_range? this is for supporting snapshots */
 time_range  INT,
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
 timer_call_data       INT NOT NULL,
 /* what metric is this? */
 metric                INT NOT NULL,
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
 id          INTEGER PRIMARY KEY AUTOINCREMENT,
 trial       INT         NOT NULL,
 name        VARCHAR        NOT NULL,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* counter index on the trial and name columns */
CREATE INDEX counter_trial_index on counter (trial, name);

CREATE TABLE counter_value (
 /* what counter is this? */
 counter            INT NOT NULL,
 /* where in the callgraph? */
 timer_callpath     INT,
 /* what thread is this? */
 thread             INT NOT NULL,
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
 trial    INT NOT NULL,
 name     VARCHAR NOT NULL,
 value    VARCHAR,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* create an index for faster queries against the primary_metadata table */
CREATE INDEX primary_metadata_index on primary_metadata (trial, name);

/* secondary metadata is metadata that could be nested, could
   contain unique data for each thread, and could be an array. */

CREATE TABLE secondary_metadata (
 id       INTEGER PRIMARY KEY AUTOINCREMENT,
 /* trial this value belongs to */
 trial    INT NOT NULL,
 /* this metadata value could be associated with a thread */
 thread   INT,
 /* this metadata value could be associated with a timer that happened */
 timer_call_data    INT,
 /* which call to the context timer was this? */
 call_number    INT,
 /* which call to the context timer was this? */
 time_range    INT,
 /* this metadata value could be a nested structure */
 parent   INT,
 /* the name of the metadata field */
 name     VARCHAR NOT NULL,
 /* the value of the metadata field */
 value    VARCHAR,
 /* this metadata value could be an array - so tokenize it */
 is_array BOOLEAN DEFAULT FALSE,
 FOREIGN KEY(trial) REFERENCES trial(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(thread) REFERENCES thread(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(timer_call_data) REFERENCES timer_call_data(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(parent) REFERENCES secondary_metadata(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION,
 FOREIGN KEY(time_range) REFERENCES time_range(id)
   ON DELETE NO ACTION ON UPDATE NO ACTION
);

/* create an index for faster queries against the secondary_metadata table */
CREATE INDEX secondary_metadata_index on secondary_metadata (trial, name, thread, parent);

/**************************************/
/* CREATE THE METADATA RELATED TABLES */
/**************************************/

/* this is the view table, which organizes and filters trials */
create table taudb_view (
	id					INTEGER PRIMARY KEY AUTOINCREMENT,
	/* views can be nested */
	parent				INTEGER			NULL,
	/* name of the view */
	name				VARCHAR	NOT NULL,
	/* view conjoin type for parameters */
	conjoin				VARCHAR	NOT NULL,
	FOREIGN KEY (parent) REFERENCES taudb_view(id)
	  ON DELETE CASCADE ON UPDATE CASCADE
);

create table taudb_view_parameter (
	/* the view ID */
	taudb_view			INTEGER	NOT NULL,
	/* the table name for the where clause */
	table_name			VARCHAR	NOT NULL,
	/* the column name for the where clause.
	   If the table_name is one of the metadata tables, this is the 
	   value of the "name" column */
	column_name			VARCHAR	NOT NULL,
	/* the operator for the where clause */
	operator			VARCHAR	NOT NULL,
	/* the value for the where clause */
	value				VARCHAR	NOT NULL,
	FOREIGN KEY (taudb_view) REFERENCES taudb_view(id)
	  ON DELETE CASCADE ON UPDATE CASCADE
);

/* simple view of all trials */
INSERT INTO taudb_view (parent, name, conjoin) VALUES (NULL, 'All Trials', 'and');
/* must have a parameter or else the sub views for this view do not work correctly*/
INSERT INTO taudb_view_parameter (taudb_view, table_name, column_name, operator, value) VALUES (1, 'trial', 'total_threads', '>', '-1');


