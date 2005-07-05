drop table analysis_result_data;
drop table analysis_result;
drop table analysis_settings;
drop table trial_in_view;
drop table trial_view;

create table analysis_settings (
	id					SERIAL			NOT NULL	PRIMARY KEY,
	application			INTEGER			NOT NULL,
	experiment			INTEGER			NOT NULL,
	trial				INTEGER			NOT NULL,
	metric				INTEGER			NULL,
	method				VARCHAR(256)	NOT NULL,
	dimension_reduction	VARCHAR(256)	NOT NULL,
	normalization		VARCHAR(256)	NOT NULL,
	FOREIGN KEY (application) REFERENCES application(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY (experiment) REFERENCES experiment(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY (trial) REFERENCES trial(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY (metric) REFERENCES metric(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
);

create table analysis_result (
	id					SERIAL			NOT NULL	PRIMARY KEY,
	analysis_settings	INTEGER			NOT NULL,
	description			VARCHAR(256)	NOT NULL,
	thumbnail_size		INTEGER			NULL,
	image_size			INTEGER			NULL,
	thumbnail			BYTEA			NULL,
	image				BYTEA			NULL,
	result_type			INTEGER			NOT NULL
);

CREATE TABLE analysis_result_data (
    interval_event		INTEGER				NOT NULL,
    metric				INTEGER				NOT NULL,
    value    			DOUBLE PRECISION	NULL,
	data_type			INTEGER				NOT NULL,
	analysis_result		INTEGER				NOT NULL,
	cluster_index		INTEGER				NULL,
    FOREIGN KEY(interval_event) REFERENCES interval_event(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY(metric) REFERENCES metric(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION,
    FOREIGN KEY(analysis_result) REFERENCES analysis_result(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
);

create table trial_view (
	id					SERIAL			NOT NULL	PRIMARY KEY,
	parent				INTEGER			NULL,
	name				VARCHAR(256)	NOT NULL,
	table_name			VARCHAR(32)		NOT NULL,
	column_name			VARCHAR(32)		NOT NULL,
	operator			VARCHAR(32)		NOT NULL,
	value				VARCHAR(256)	NOT NULL,
	FOREIGN KEY (parent) REFERENCES trial_view(id) 
		ON DELETE NO ACTION ON UPDATE NO ACTION
);

create table trial_in_view (
	trial				INTEGER,
	trial_view			INTEGER,
	FOREIGN KEY (trial) REFERENCES trial(id)
		ON DELETE NO ACTION ON UPDATE NO ACTION,
	FOREIGN KEY (trial_view) REFERENCES trial_view(id)
		ON DELETE NO ACTION ON UPDATE NO ACTION
);
